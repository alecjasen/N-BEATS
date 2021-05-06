from model import *
import numpy as np
from losses import *
from SyntheticDatasets import *
from Visualization import *
import torch as t
import pandas as pd
import matplotlib.pyplot as plt


# data = pd.read_csv("datasets/tourism/monthly_in.csv")
# data = t.tensor(np.arange(0,260)).reshape(1,1,-1).float()
# data2 = t.tensor(np.arange(0,260)+np.sin((np.pi/5)*np.arange(0,260))).reshape(1,1,-1).float()
# data2 = t.hstack((data,data2)).reshape(2,1,260)
# data = data.repeat(2, 1, 1)
# data.requires_grad = False
# data2 = data2.repeat(2, 1, 1)
# data2.requires_grad = False

Loss = sMAPE
trend_hidden_layers = (256*np.ones((4,))).astype(int)
seasonal_hidden_layers = (256*np.ones((4,))).astype(int)
learning_rate = .001

trend = Stack(num_blocks=3, block_type='trend',
              block_hidden_layers=trend_hidden_layers,
              trend_num_parameters=4,
              trend_basis_fn=None)


# mini batch size check
data2, trend2, seasonal2 = Make_Dataset()
data2.requires_grad = False
sampler = iter(Data_Sampler(data2[:,:-20], batch_size=64))
optimizer = t.optim.Adam(trend.parameters(), lr=learning_rate)
warmstart_iterations = 1000
lr_decay_step = warmstart_iterations//6
for i in range(warmstart_iterations):
    data_train, data_label = next(sampler)
    f, g = trend(data_train)
    loss = Loss(data_label.squeeze(1), f)#, data_train.squeeze(1))
    if np.isnan(loss.detach().numpy()):
        break
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    t.nn.utils.clip_grad_norm_(trend.parameters(), 1.0)
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * 0.5 ** (i // lr_decay_step)
with t.no_grad():
    f, g = trend(data2[:, -260:-20][:, None, :])
    loss = Loss(data2[:,-20:], f)
plots_array_forecast = make_plots(f.detach().numpy(), trend2[:, -20:].detach().numpy())
# plots_array_backcast = make_plots(g.squeeze(1).detach().numpy(), trend2[:, :-20].detach().numpy())
img_tile(plots_array_forecast, warmstart_iterations, 'Forecast Trend Warm Start')
# img_tile(plots_array_backcast, 1000, f'Backcast Trend Warm Start, Iterations: {warmstart_iterations}')
#print(f"RMSE Loss: {loss}")
print(f"sMAPE Loss:{sMAPE(data2[:,-20:], f)}")
print(f"vector validation loss: {sMAPE_vec(data2[:, -20:], f)}")
print(trend.total_param_forecast.detach().numpy())
# print(trend.block.backcast_basis_function.parameters.detach().numpy())
nbeats = NBEATS_Modified(trend_stacks=[trend],
                         seasonal_hidden_layers=seasonal_hidden_layers,
                         num_seasonal_stacks=1,seasonal_basis_fn=None)
#nbeats = Stack(block_type="trend")
# for i in nbeats.parameters():
#     print(i)
optimizer = t.optim.Adam(nbeats.parameters(), lr=learning_rate)
diff = 1
prev = 0
iterations = 0
loss_values = np.zeros((1000,))
lr_decay_step = 1000
while diff > 1e-6:
    data_train, data_label = next(sampler)
    f, g = nbeats(data_train)
    loss = Loss(data_label.squeeze(1),f)#,data_train.squeeze(1))
    diff = t.abs(loss-prev)
    prev = loss
    if iterations % 50 == 0:
        loss_values[int(iterations//50)] = loss
        print(loss)
    optimizer.zero_grad()
    loss.backward()
    t.nn.utils.clip_grad_norm_(trend.parameters(), 1.0)
    optimizer.step()
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * 0.5 ** (iterations // lr_decay_step)
    iterations += 1
    if iterations % 1000 == 0:
        with t.no_grad():
            f, g = nbeats(data2[:, -260:-20][:, None, :])
            validation_loss = Loss(data2[:, -20:], f)
            print(f"validation loss: {validation_loss}")
            print(f"vector validation loss: {sMAPE_vec(data2[:, -20:], f)}")
            plots_array_forecast = make_plots(nbeats.trend_predictions.detach().numpy(),
                                              trend2[:, -20:].detach().numpy())
            img_tile(plots_array_forecast, iterations, "validation_trend")
            plots_array_seasonal = make_plots(nbeats.seasonal_predictions.detach().numpy(),
                                              seasonal2[:, -20:].detach().numpy())
            img_tile(plots_array_seasonal, iterations, "validation_seasonal")


with t.no_grad():
    f, g = nbeats(data2[:, -260:-20][:, None, :])
    validation_loss = Loss(data2[:, -20:], f)
    print(f"validation loss: {validation_loss}")
    plots_array_forecast = make_plots(nbeats.trend_predictions.detach().numpy(),
                                      trend2[:, -20:].detach().numpy())
    img_tile(plots_array_forecast, iterations, "validation_trend")
    plots_array_seasonal = make_plots(nbeats.seasonal_predictions.detach().numpy(),
                                      seasonal2[:, -20:].detach().numpy())
    img_tile(plots_array_seasonal, iterations, "validation_seasonal")

print(iterations)
plt.plot(loss_values)
plt.title('Loss at every 50th iteration')
plt.show()


