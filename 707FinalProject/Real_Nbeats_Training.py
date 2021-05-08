import torch as t
import numpy as np
from common.torch.ops import to_tensor
from common.sampler import TimeseriesSampler
from experiments.model import interpretable
from common.torch.losses import *
from losses import *
from SyntheticDatasets import *
from Visualization import *
from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd

# stocks = pd.read_csv("DOW.tsv", sep="\t", header=None)[0].tolist()
# yf.pdr_override()
# stock_data = pdr.get_data_yahoo(["^GSPC"], start="2016-01-01", end="2021-01-01")['Adj Close']
# data = t.tensor(stock_data.values).reshape(1,-1).float()
data, trend, seasonal = Make_Dataset()
batch_size = 64
epochs = 15000
lr_decay_step = epochs // 3
Loss = smape_2_loss
learning_rate = .001

sampler = iter(TimeseriesSampler(timeseries=data[:, :-20].detach().numpy(),
                                 insample_size=240,
                                 outsample_size=20,
                                 window_sampling_limit=data[:, :-20].detach().numpy().shape[1],
                                 batch_size=batch_size))
Nbeats_Real = interpretable(seasonality_layer_size=256,
                            seasonality_blocks=3, seasonality_layers=4,
                            trend_layer_size=256,
                            degree_of_polynomial=2, trend_blocks=3, trend_layers=4,
                            num_of_harmonics=1, input_size=240, output_size=20)

optimizer = t.optim.Adam(Nbeats_Real.parameters(), lr=learning_rate)
loss_values = np.zeros((150,))
for iterations in range(epochs):
    optimizer.zero_grad()
    Nbeats_Real.train()
    x, x_mask, y, y_mask = map(to_tensor, next(sampler))
    optimizer.zero_grad()
    forecast = Nbeats_Real(x, x_mask)
    loss = Loss(forecast, y, y_mask)
    loss.backward()
    t.nn.utils.clip_grad_norm_(Nbeats_Real.parameters(), 1.0)
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * 0.5 ** (iterations // lr_decay_step)
    if iterations % 100 == 0:
        loss_values[int(iterations//100)] = loss
        print(loss)

    if iterations % 1000 == 999:
        Nbeats_Real.eval()
        with t.no_grad():
            f = Nbeats_Real(data[:, -260:-20], t.ones(data[:, -260:-20].size()))
            validation_loss = Loss(f, data[:, -20:], t.ones(data[:, -20:].size()))
            print(f"validation loss: {validation_loss}")
            # plots_array_forecast = make_plots(Nbeats_Real.trend_predictions.detach().numpy(),
            #                                   data[:, -20:].detach().numpy())
            # img_tile(plots_array_forecast[0:10], iterations, "SP500 validation_trend")
            # plots_array_seasonal = make_plots(Nbeats_Real.seasonal_predictions.detach().numpy())
            # img_tile(plots_array_seasonal[0:10], iterations, "SP500 validation_seasonal")
            # plots_array_seasonal = make_plots(f.detach().numpy(), data[:, -20:].detach().numpy())
            # img_tile(plots_array_seasonal[0:10], iterations, "SP500 validation_full")
            plots_array_forecast = make_plots(Nbeats_Real.trend_predictions.detach().numpy(),
                                              trend[:, -20:].detach().numpy())
            img_tile(plots_array_forecast, iterations, "Synthetic validation_trend")
            plots_array_seasonal = make_plots(Nbeats_Real.seasonal_predictions.detach().numpy(),
                                              seasonal[:, -20:].detach().numpy())
            img_tile(plots_array_seasonal, iterations, "Synthetic validation_seasonal")
            plots_array_seasonal = make_plots(f.detach().numpy(),
                                              data[:, -20:].detach().numpy())
            img_tile(plots_array_seasonal, iterations, "Synthetic validation_full")