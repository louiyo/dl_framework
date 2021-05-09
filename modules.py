import pytorch as torch

def LossMSE(input, weights, target, bias):
    N = np.size(target)
    L = torch.sum(torch.pow(output - target,2))/N
    dL_dw = 2 * torch.mul(input, torch.mul(f_prim(input,weights,bias) , (f(input,weights,bias) - target))) /N
    dL_dB = 2 * torch.mul(f_prim(input,weights,bias) , (f(input,weights,bias) - target)) /N
    return L , dL_dw , dL_dB
def sigma(sex):
    return 1/(1 + torch.exp(-sex))

def lossigma(input, weights, target, bias):
    L = - torch.sum(torch.log(sigma(torch.mul(output, target))))
    dL_dw = - torch.sum(target torch.log(sigma(-torch.mul(output, target))))
    dL_db = - torch.sum(input target torch.log(sigma(-torch.mul(output, target))))
    return L , dL_dw , dL_dB