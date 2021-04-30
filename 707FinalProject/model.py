import numpy as np
import torch.nn as nn
import torch as t



# build a block

# CNN (channels are filter lengths)
# (4) FC Linear layers
# Linear layer outputting parameters of forecast; linear layer outputting parameter of backcast
# weights are shared with other blocks in stack

class Block(nn.Module):
    def __init__(self, forecast_basis_function, backcast_basis_function, hidden_nodes):
        super(Block, self).__init__()
        self.filtersizes = [1,5,20,50,200]
        self.CNNLayers = [nn.Conv1d(1,1,self.filtersizes[i]) for i in range(len(self.filtersizes))]
        self.linear_processing_layers = [nn.LazyLinear(hidden_nodes[0]),
                              nn.Linear(hidden_nodes[0],hidden_nodes[1]),
                              nn.Linear(hidden_nodes[1],hidden_nodes[2]),
                              nn.Linear(hidden_nodes[2],hidden_nodes[3])]
        self.ReLULayers = [nn.ReLU(),nn.ReLU(), nn.ReLU(), nn.ReLU()]
        self.linear_forecast_parameter = nn.Linear(hidden_nodes[3],self.forecast_basis_function.num_parameters)
        self.linear_backcast_parameter = nn.Linear(hidden_nodes[3],self.backcast_basis_function.num_parameters)

    def forward(self,data):
        pre_processing = [self.CNNLayers[i](data) for i in range(len(self.filtersizes))]
        inp = t.cat(pre_processing, dim=-1)
        for i in range(len(self.linear_processing_layers)):
            inp = self.ReLULayers[i](self.linear_processing_layers[i](inp))
        forecast_param = self.linear_forecast_parameter(inp)
        backcast_param = self.linear_backcast_parameter(inp)
        forecast = self.forecast_basis_function(forecast_param)
        backcast = self.backcast_basis_function(backcast_param)
        return forecast, backcast

class BasisFunction(nn.Module):
    def __init__(self, function, num_parameters, parameters=None, time_period=None):
        self.function = function
        self.num_parameters = num_parameters
        self.parameters = parameters

    def forward(self, x):
        self.parameters = x
        return self.function(self.parameters, self.time_period)


class TrendBasisFunction(BasisFunction):
    def __init__(self, function=None, num_parameters=4, parameters=None, time_period=None):
        self.num_parameters = num_parameters
        self.time_period = time_period
        if function is None:
            def polynomial(parameter_array, time_period):
                t = [i/time_period for i in range(time_period)]
                time_matrix = np.vstack([[np.power(t, i) for i in range(len(parameter_array))]])
                return np.dot(time_matrix.T, parameter_array)
            self.function = polynomial
        else:
            self.function = function
        self.parameters = parameters


class SeasonalBasisFunction(BasisFunction):
    def __init__(self, function=None, num_parameters=20, parameters=None, time_period=None):
        self.num_parameters = num_parameters
        self.time_period = time_period
        if function is None:
            def fourrier_sum(parameter_array, time_period):
                t = np.array([i / time_period for i in range(time_period)])
                cosine_portion = np.array([np.cos(2*np.pi*i*t) for i in range(int(time_period/2))])
                sine_portion = np.array([np.sin(2*np.pi*i*t) for i in range(int(time_period/2))])
                cosine_parameter_array = parameter_array[:int(time_period/2)]
                sine_parameter_array = parameter_array[int(time_period/2):]
                cosine_sum = np.dot(cosine_portion.T, cosine_parameter_array)
                sine_sum = np.dot(sine_portion.T, sine_parameter_array)
                return cosine_sum + sine_sum
            self.function = fourrier_sum
        else:
            self.function = function
        self.parameters = parameters
