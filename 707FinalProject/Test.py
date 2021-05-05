from model import *
import numpy as np
from losses import *
import torch as t
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("datasets/tourism/monthly_in.csv")

Loss = sMAPE
trend_hidden_layers = (256*np.ones((4,))).astype(int)
seasonal_hidden_layers = (256*np.ones((4,))).astype(int)
data = t.tensor(np.arange(0,260)).reshape(1,1,-1).float()
trend = Stack(block_type='trend',block_hidden_layers=trend_hidden_layers)

data2 = t.tensor(np.arange(0,260)+np.sin((np.pi/5)*np.arange(0,260))).reshape(1,1,-1).float()
data2 = t.hstack((data,data2)).reshape(2,1,260)
data = data.repeat(2, 1, 1)
data.requires_grad = False

#data2 = data2.repeat(2, 1, 1)
data2.requires_grad = False

# mini batch size check
data_train = data2[:, :, :-20]
data_label = data2[:, :, -20:]

optimizer = t.optim.SGD(trend.parameters(), lr=0.0001)
warmstart_iterations = 1000
for _ in range(warmstart_iterations):
    f, g = trend(data_train)
    loss = Loss(data_label.squeeze(1), f)#, data_train.squeeze(1))
    if np.isnan(loss.detach().numpy()):
        break
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(f.detach().numpy()[0])
plt.title(f'Trend Warm Start, {loss}')
plt.show()

nbeats = NBEATS_Modified(trend_stacks=[trend],
                         seasonal_hidden_layers=seasonal_hidden_layers,
                         num_seasonal_stacks=1,seasonal_basis_fn=None)
#nbeats = Stack(block_type="trend")
# for i in nbeats.parameters():
#     print(i)
optimizer = t.optim.Adam(nbeats.parameters(), lr=0.0001)
diff = 1
prev = 0
iterations = 0
loss_values = np.zeros((100,))
while diff>1e-6:
    f, g = nbeats(data_train)
    loss = Loss(data_label.squeeze(1),f)#,data_train.squeeze(1))
    diff = t.abs(loss-prev)
    prev = loss
    if iterations % 50 == 0:
        loss_values[int(iterations//50)] = loss
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    iterations+=1
    if iterations % 1000==0:
        plt.plot(nbeats.trend_predictions.detach().numpy()[0,:])
        plt.title(f'Trend, Iterations: {iterations}')
        plt.show()
        plt.plot(nbeats.seasonal_predictions.detach().numpy()[0,:])
        plt.title(f'Seasonal, Iterations: {iterations}')
        plt.show()
        break
print(f)
print(iterations)
plt.plot(loss_values)
plt.title('Loss at every 50th iteration')
plt.show()


