import torch as t

def sMAPE(act_y,pred_y):
        Num = t.abs(act_y-pred_y)
        Den = (t.abs(act_y.data)+t.abs(pred_y.data))*.5
        Loss = t.mean(Num/Den)*100
        return Loss

def MASE(act_y,pred_y,x):
    Num = t.abs(act_y - pred_y)
    Den = t.mean(t.abs(x.data[:, 1:]-x.data[:, :-1]), axis=1)
    return t.mean(t.mean(Num,axis=1)/Den)