from model import *
import numpy as np
import torch as t

hidden_layers = (2*np.ones((4,))).astype(int)
data = t.tensor(np.arange(0,260)).reshape(1,1,-1).float()

block = Block("trend",hidden_layers)
data = data.repeat(2,1,1)
data.requires_grad = False
# mini batch size check
data_train = data[:, :, :-20]
data_label = data[:, :, -20:]
nbeats = NBEATS_Modified(num_trend_stacks=1, num_seasonal_stacks=1)
print(nbeats.parameters())
optimizer = t.optim.SGD(nbeats.parameters(), lr=0.1)
for _ in range(1):
    f = nbeats(data_train)
    loss = t.mean((f - data_label.squeeze(1))**2)
    print(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f)
