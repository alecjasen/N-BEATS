import numpy as np
import torch.nn as nn
import torch as t



# build a block

# CNN (channels are filter lengths)
# (4) FC Linear layers
# Linear layer outputting parameters of forecast; linear layer outputting parameter of backcast
# weights are shared with other blocks in stack

class Block(nn.Module):
    def __init__(self, block_type="trend", hidden_nodes=(2*np.ones((4,))).astype(int)):
        super(Block, self).__init__()
        self.filtersizes = [1,5,20,50,200]
        self.CNNLayers = [nn.Conv1d(1,1,self.filtersizes[i]) for i in range(len(self.filtersizes))]
        self.linear_processing_layers = [nn.LazyLinear(hidden_nodes[0]),
                              nn.Linear(hidden_nodes[0],hidden_nodes[1]),
                              nn.Linear(hidden_nodes[1],hidden_nodes[2]),
                              nn.Linear(hidden_nodes[2],hidden_nodes[3])]
        self.ReLULayers = [nn.ReLU(),nn.ReLU(), nn.ReLU(), nn.ReLU()]
        if block_type == "trend":
            self.forecast_basis_function = TrendBasisFunction(time_period=20)
            self.backcast_basis_function = TrendBasisFunction(time_period=240)
        else:
            self.forecast_basis_function = SeasonalBasisFunction(time_period=20)
            self.backcast_basis_function = SeasonalBasisFunction(time_period=240)
        self.linear_forecast_parameter = nn.Linear(hidden_nodes[3],self.forecast_basis_function.num_parameters)
        self.linear_backcast_parameter = nn.Linear(hidden_nodes[3],self.backcast_basis_function.num_parameters)


    def forward(self,data):
        pre_processing = [conv_layer(data).squeeze(1) for conv_layer in self.CNNLayers]
        inp = t.cat(pre_processing, dim=-1)
        for i in range(len(self.linear_processing_layers)):
            inp = self.ReLULayers[i](self.linear_processing_layers[i](inp))
        forecast_param = self.linear_forecast_parameter(inp)
        backcast_param = self.linear_backcast_parameter(inp)
        forecast = self.forecast_basis_function(forecast_param)
        backcast = self.backcast_basis_function(backcast_param)
        return forecast, backcast

class BasisFunction(nn.Module):
    def __init__(self, function=None, num_parameters=None, parameters=None, time_period=None):
        super(BasisFunction, self).__init__()
        self.function = function
        self.num_parameters = num_parameters
        self.parameters = parameters
        self.time_period = time_period

    def forward(self, x):
        self.parameters = x
        return self.function(self.parameters)


class TrendBasisFunction(BasisFunction):
    def __init__(self, function=None, num_parameters=4, parameters=None, time_period=None):
        super(TrendBasisFunction, self).__init__()
        self.num_parameters = num_parameters
        self.time_period = time_period
        self.time = t.tensor([i / time_period for i in range(time_period)])
        self.time.requires_grad = False
        if function is None:
            def polynomial(parameter_array):
                time_matrix = t.vstack([t.float_power(self.time, i) for i in range(parameter_array.shape[-1])]).float()
                return t.einsum('ij,ki->kj', time_matrix,  parameter_array)
            self.function = polynomial
        else:
            self.function = function
        self.parameters = parameters


class SeasonalBasisFunction(BasisFunction):
    def __init__(self, function=None, parameters=None, time_period=20):
        super(SeasonalBasisFunction, self).__init__()
        self.num_parameters = time_period
        self.half_params = int(self.num_parameters/2)
        self.time_period = time_period
        self.time = t.tensor([i / time_period for i in range(self.time_period)])
        self.time.requires_grad = False
        if function is None:
            def fourrier_sum(parameter_array):
                cosine_portion = t.vstack([t.cos(2*np.pi*i*self.time) for i in range(self.half_params)]).T
                sine_portion = t.vstack([t.sin(2*np.pi*i*self.time) for i in range(self.half_params)]).T
                cosine_parameter_array = parameter_array[:, :self.half_params].reshape(parameter_array.size(0), -1, 1)
                sine_parameter_array = parameter_array[:, self.half_params:].reshape(parameter_array.size(0), -1, 1)
                cosine_sum = t.matmul(cosine_portion, cosine_parameter_array).squeeze()
                sine_sum = t.matmul(sine_portion, sine_parameter_array).squeeze()
                return cosine_sum + sine_sum
            self.function = fourrier_sum
        else:
            self.function = function
        self.parameters = parameters

class Stack(nn.Module):
    def __init__(self, num_blocks=3, stack_type="trend", block_hidden_layers=None):
        super(Stack, self).__init__()
        self.num_blocks = num_blocks
        if block_hidden_layers is None:
            block_hidden_layers = (2 * np.ones((4,))).astype(int)
        self.block = Block(stack_type, block_hidden_layers)

    def forward(self, x):
        residual = x
        backcast = 0
        forecasts = t.zeros(1,1,self.block.forecast_basis_function.time_period)
        for _ in range(self.num_blocks):
            residual = residual - backcast
            forecast, backcast = self.block(residual)
            backcast = backcast.reshape(backcast.size(0), 1, backcast.size(1))
            forecasts = forecasts + forecast
        return forecasts, backcast

class NBEATS_Modified(nn.Module):
    def __init__(self, num_trend_stacks=1, num_seasonal_stacks=1, trend_hidden_layers=None, seasonal_hidden_layers=None):
        super(NBEATS_Modified, self).__init__()
        # could lift this requirement, but from an interpretability POV, it makes sense to require
        assert(num_trend_stacks == num_seasonal_stacks)
        if trend_hidden_layers is None:
            trend_hidden_layers = (2 * np.ones((4,))).astype(int)
        if seasonal_hidden_layers is None:
            seasonal_hidden_layers = (2 * np.ones((4,))).astype(int)
        self.trend_stacks = nn.ModuleList([Stack(stack_type="trend", block_hidden_layers=trend_hidden_layers) for _ in range(num_trend_stacks)])
        self.seasonal_stacks = nn.ModuleList([Stack(stack_type="seasonal", block_hidden_layers=seasonal_hidden_layers) for _ in range(num_seasonal_stacks)])

    def forward(self, x):
        backcast = x
        residual = 0
        forecasts = t.zeros(1, 1, self.trend_stacks[0].block.forecast_basis_function.time_period)
        # alternate:
        for i in range(len(self.trend_stacks)):
            residual = backcast - residual
            print(residual.shape)
            forecast, backcast = self.trend_stacks[i](residual)
            print(forecast.shape)
            forecasts += forecast
            backcast = backcast.reshape(backcast.size(0), 1, backcast.size(1))
            residual = backcast - residual
            forecast, backcast = self.seasonal_stacks[i](residual)
            print(forecast.shape)
            forecasts = forecasts + forecast
            backcast = backcast.reshape(backcast.size(0), 1, backcast.size(1))
        return forecasts
