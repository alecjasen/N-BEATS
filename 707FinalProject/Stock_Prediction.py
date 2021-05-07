from pandas_datareader import data as pdr
import yfinance as yf
import pandas as pd
import numpy as np
import torch as t
import matplotlib.pyplot as plt
from losses import *
from model import *
from SyntheticDatasets import *
from Visualization import *

yf.pdr_override()
data = t.tensor(pdr.get_data_yahoo("^GSPC", start="2000-01-01", end="2021-01-01")['Adj Close']).reshape((1,-1)).float()
Loss = sMAPE
trend_hidden_layers = (256*np.ones((50,))).astype(int)
seasonal_hidden_layers = (256*np.ones((50,))).astype(int)
trend = Stack(num_blocks=3, block_type='trend',
              block_hidden_layers=trend_hidden_layers,
              trend_num_parameters=4,
              trend_basis_fn=None)
learning_rate = .0001
batch_size = 64
sampler = iter(Data_Sampler(data[:,:-20],batch_size=batch_size))
optimizer = t.optim.Adam(trend.parameters(), lr=learning_rate)
warmstart_iterations = 1
#lr_decay_step = warmstart_iterations//3
for i in range(warmstart_iterations):
    data_train, data_label = next(sampler)
    f, g = trend(data_train)
    loss = Loss(data_label.squeeze(1), f)
    if i%100==0:
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    t.nn.utils.clip_grad_norm_(trend.parameters(), 1.0)
    optimizer.step()
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)
with t.no_grad():
    f, g = trend(data[:,-260:-20][:, None, :])
    loss = Loss(data[:,-20:], f)
plots_array_forecast = make_plots(f.detach().numpy(), data[:, -20:].detach().numpy())
#plots_array_backcast = make_plots(g.squeeze(1).detach().numpy(), trend2[:, -260:-20].detach().numpy())
img_tile(plots_array_forecast, warmstart_iterations, 'Forecast Trend Warm Start')
#img_tile(plots_array_backcast, warmstart_iterations, 'Backcast Trend Warm Start')
print(f"sMAPE Loss: {loss}")
#print(f"sMAPE Loss:{sMAPE(data2[:,-20:], f)}")
print(f"vector validation loss: {sMAPE_vec(data[:, -20:], f)}")
#print(trend.total_param_forecast.detach().numpy())
# print(trend.block.backcast_basis_function.parameters.detach().numpy())
nbeats = NBEATS_Modified(trend_stacks=[trend],
                         seasonal_hidden_layers=seasonal_hidden_layers,
                         num_seasonal_stacks=1,
                         num_seasonal_blocks=3,
                         seasonal_basis_fn=None)
# nbeats = Stack(block_type="trend")
# for i in nbeats.parameters():
#     print(i)
optimizer = t.optim.Adam(nbeats.parameters(), lr=learning_rate)
diff = 1
prev = 0
iterations = 0
loss_values = np.zeros((1000,))
validation_values = np.zeros((1000,))
lr_decay_step = 5000
while diff > 1e-6:
    nbeats.train()
    data_train, data_label = next(sampler)
    f, g = nbeats(data_train)
    loss = Loss(data_label.squeeze(1), f)
    diff = t.abs(loss-prev)
    prev = loss
    if iterations % 100 == 0:
        loss_values[int(iterations//100)] = loss
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    t.nn.utils.clip_grad_norm_(nbeats.parameters(), 1.0)
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * 0.5 ** (iterations // lr_decay_step)
    iterations += 1
    if iterations % 1000 == 0:
        nbeats.eval()
        with t.no_grad():
            f, g = nbeats(data[:,-260:-20][:, None, :])
            validation_loss = Loss(data[:,-20:], f)
            print(f"validation loss: {validation_loss}")
            print(f"vector validation loss: {sMAPE_vec(data[:,-20:], f)}")
            plots_array_forecast = make_plots(nbeats.trend_predictions.detach().numpy(),
                                              data[:,-20:].detach().numpy())
            img_tile(plots_array_forecast, iterations, "validation_trend")
            plots_array_seasonal = make_plots(nbeats.seasonal_predictions.detach().numpy())
            img_tile(plots_array_seasonal, iterations, "validation_seasonal")
            plots_array_seasonal = make_plots(f.detach().numpy(),data[:,-20:].detach().numpy())
            img_tile(plots_array_seasonal, iterations, "validation_full")


print(iterations)
plt.plot(loss_values[0:int(iterations//100)])
plt.title('Training Loss at every 100th iteration')
plt.show()

