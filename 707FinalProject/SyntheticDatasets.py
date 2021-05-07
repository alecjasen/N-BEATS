import torch as t
import numpy as np

def Make_Dataset():
    Time = t.tensor((np.arange(0, 300)))
    Time = Time/t.max(Time)
    trend = t.vstack([t.float_power(Time, i) for i in range(1, 4)]+[t.float_power(2,Time)]).float()
    # trend = t.ones((Time.size()[0], 6))
    # trend[:, 1] = Time.T
    # for j in range(2, 6):
    #     trend[:, j] = 2 * Time * trend[:, j - 1] - trend[:, j - 2]
    # trend = trend[:,1:].T
    seasonal_sine = t.vstack([t.sin((2*np.pi*i)*Time/t.max(Time)) for i in range(0, 4)]).float()
    dataset = []
    for i in range(trend.shape[0]):
        for j in range(seasonal_sine.shape[0]):
            dataset.append(trend[i] + seasonal_sine[j])
    dataset = t.vstack(dataset)
    dataset.requires_grad = False
    trend1 = trend.repeat_interleave(seasonal_sine.shape[0], dim=0)
    seasonal_sine1 = seasonal_sine.repeat((trend.shape[0], 1))
    return dataset, trend1, seasonal_sine1

class Data_Sampler:
    def __init__(self, data, horizon=20, back_multiplier=12, batch_size=2):
        self.data = data.float()
        self.horizon = horizon
        self.back_multiplier = back_multiplier
        self.valid_indices = list(np.arange(back_multiplier*horizon, data.shape[1]-horizon+1))
        self.batch_size = batch_size

    def __iter__(self):
        # assume all time series have same amounts of data
        while True:
            tss = np.random.choice(range(self.data.shape[0]), size=self.batch_size)
            starts = np.random.choice(self.valid_indices, size=self.batch_size)
            data_mb = []
            label_mb = []
            for ts_index, start in zip(tss, starts):
                data_mb.append(self.data[ts_index, start - self.back_multiplier * self.horizon:start])
                label_mb.append(self.data[ts_index, start: start + self.horizon])
            data_mb = t.vstack(data_mb)
            data_mb = data_mb[:, None, :]
            label_mb = t.vstack(label_mb)
            label_mb = label_mb[:, None, :]
            yield data_mb, label_mb
        # yield self.data[i, j - self.back_multiplier*self.horizon:j + self.horizon]
class Real_NBeats_Data_Sampler:
    def __init__(self, data, horizon=20, back_multiplier=12, batch_size=2):
        self.data = data
        self.horizon = horizon
        self.back_multiplier = back_multiplier
        self.valid_indices = list(np.arange(back_multiplier*horizon, data.shape[1]-horizon+1))
        self.batch_size = batch_size

    def __iter__(self):
        # assume all time series have same amounts of data
        while True:
            tss = np.random.choice(range(self.data.shape[0]), size=self.batch_size)
            starts = np.random.choice(self.valid_indices, size=self.batch_size)
            data_mb = []
            data_mask = []
            label_mb = []
            label_mask = []
            for ts_index, start in zip(tss, starts):
                data_mb.append(self.data[ts_index, start - self.back_multiplier * self.horizon:start])
                label_mb.append(self.data[ts_index, start: start + self.horizon])
            data_mb = t.vstack(data_mb)
            data_mb = data_mb[:, None, :]
            label_mb = t.vstack(label_mb)
            label_mb = label_mb[:, None, :]
            yield data_mb, label_mb
        # yield self.data[i, j - self.back_multiplier*self.horizon:j + self.horizon]
if __name__ == '__main__':
    data, _, _= Make_Dataset()
    sampler = Data_Sampler(data)
    for b,d in sampler.next_minibatch(2):
        print('hi')
