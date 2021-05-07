import torch as t
import numpy as np
from common.torch.ops import to_tensor
from common.sampler import TimeseriesSampler
from experiments.model import interpretable
from common.torch.losses import *
from SyntheticDatasets import *
from Visualization import *
epochs= 15000
lr_decay_step = epochs // 3
Loss = smape_2_loss
learning_rate = .001
data2, trend2, seasonal2 = Make_Dataset()
data2.requires_grad = False
data2_train = data2[:,:-20]
sampler = iter(TimeseriesSampler(timeseries=data2_train.detach().numpy(),insample_size=240,outsample_size=20,
                            window_sampling_limit=data2_train.detach().numpy().shape[1],
                                 batch_size=64))
Nbeats_Real = interpretable(seasonality_layer_size = 256,
                            seasonality_blocks = 3,seasonality_layers = 4,
                            trend_layer_size = 256,
                            degree_of_polynomial = 2,trend_blocks = 3,trend_layers = 4,
                            num_of_harmonics = 1, input_size=240, output_size=20)
optimizer = t.optim.Adam(Nbeats_Real.parameters(),lr=learning_rate)
loss_values = np.zeros((150,))
for iterations in range(epochs):
    optimizer.zero_grad()
    Nbeats_Real.train()
    x, x_mask, y, y_mask = map(to_tensor, next(sampler))
    optimizer.zero_grad()
    forecast = Nbeats_Real(x, x_mask)
    loss = Loss(forecast,y,y_mask)
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
            f = Nbeats_Real(data2[:, -260:-20],t.ones(data2[:, -260:-20].size()))
            validation_loss = Loss(f,data2[:, -20:],t.ones(data2[:,-20:].size()))
            print(f"validation loss: {validation_loss}")
            plots_array_forecast = make_plots(Nbeats_Real.trend_predictions.detach().numpy(),
                                              trend2[:, -20:].detach().numpy())
            img_tile(plots_array_forecast, iterations, "validation_trend")
            plots_array_seasonal = make_plots(Nbeats_Real.seasonal_predictions.detach().numpy(),
                                              seasonal2[:, -20:].detach().numpy())
            img_tile(plots_array_seasonal, iterations, "validation_seasonal")
