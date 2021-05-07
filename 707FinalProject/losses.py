import torch as t

def sMAPE(act_y,pred_y):
        Num = t.abs(act_y-pred_y)
        Den = (t.abs(act_y.data)+t.abs(pred_y.data))*.5
        Loss = t.mean(Num/Den)*100
        return Loss


def sMAPE_Reg(act_y, pred_y, weights):
    alpha = 500
    l1_ratio = .5
    Num = t.abs(act_y - pred_y)
    Den = (t.abs(act_y.data) + t.abs(pred_y.data)) * .5
    Loss = t.mean(Num / Den) * 100 + alpha*l1_ratio*t.mean(t.linalg.norm(weights,ord=1,dim=1))
    +.5*alpha*(1-l1_ratio)*t.mean(t.linalg.norm(weights,ord=2,dim=1))
    return Loss

def sMAPE_vec(act_y, pred_y):
    Num = t.abs(act_y - pred_y)
    Den = (t.abs(act_y.data) + t.abs(pred_y.data)) * .5
    Loss = t.mean(Num / Den,axis=1) * 100
    return Loss

def MASE(act_y,pred_y,x):
    Num = t.abs(act_y - pred_y)
    Den = t.mean(t.abs(x.data[:, 1:]-x.data[:, :-1]), axis=1)
    return t.mean(t.mean(Num,axis=1)/Den)

def RMSE(act_y,pred_y):
    Num = (act_y-pred_y)**2
    Den = (act_y)**2+(pred_y)**2
    return 2*t.mean(Num/Den)

def MAPE(act_y,pred_y):
    Num = t.abs(act_y - pred_y)
    Den = (t.abs(act_y))
    Loss = t.mean(Num / Den) * 100
    return Loss