import torch
import math as ma
#some activation functions
######################################################################

def tanh(x):
    y = torch.tanh(x).add(1).div(2)
    return y

def dtanh(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)
######################################################################

def relu(x):
    a = x.sign().add(1).div(2).long()
    return torch.mul(a,x)

def drelu(x):
    return x.sign().add(1).div(2).long()
######################################################################

#Loss
def loss(v, t):
    print('nan',v,t)
    return (v - t).pow(2)

def dloss(v, t):
    return 2 * (v - t)

#######################################################################
def forward_pass(x, w, b):
    return w.matmul(x) + b

def backward_pass(w1, b1, w2, b2, w3, b3,
                t,
                x, s1, x1, s2, x2, s3, x3,
                dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3):
    x0 = x
    dl_dx3 = dloss(x3, t)
    dl_ds3 = dtanh(s3) * dl_dx3
    dl_dx2 = w3.t().mv(dl_ds3)
    dl_ds2 = dtanh(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dtanh(s1) * dl_dx1

    dl_dw3.add_(dl_ds3.view(-1, 1).mm(x2.view(1, -1)))
    dl_db3.add_(dl_ds3)
    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)
##########################
#update parameters

def optimizer(w,b,lr,dl_dw,dl_db):
    return w - lr * dl_dw, b - lr * dl_db
   
    