import pytorch as torch
'''
def LossMSE(input, weights, bias, target):
    N = np.size(target)
    L = torch.sum(torch.pow(output - target,2))/N
    dL_dw = 2 * torch.mul(input, torch.mul(f_prim(input,weights,bias) , (f(input,weights,bias) - target))) /N
    dL_dB = 2 * torch.mul(f_prim(input,weights,bias) , (f(input,weights,bias) - target)) /N
    return L , dL_dw , dL_dB
'''
#some activation functions
######################################################################

def sigma(x):
    y = torch.tanh(x)
    return y

def dsigma(x):
    y = 1 - torch.pow(sigma(x),2)
    return y
######################################################################

def relu(x):
    a = x.sign().add(1).div(2).long()
    return torch.mul(a,x)

def drelu(x):
    return x.sign().add(1).div(2).long()
######################################################################

#Loss
def loss(v, t):
    y = torch.sum(torch.mul((t-v),(t-v)))
    return y

def dloss(v, t):
    y = torch.mul((t-v),-2)
    return y

#######################################################################
def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    s1 = w1.mv(x0) + b1
    x1 = sigma(s1)
    s2 = w2.mv(x1) + b2
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
                t,
                x, s1, x1, s2, x2,
                dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1

    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)

##########################
#update parameters

def optimizer(w,b,lr,dl_dw,dl_db):
    w = w - lr * dl_dw1
    b = b - lr * dl_db1
    