import torch
import math as ma
import matplotlib.pyplot as plt
def generate_dataset(N = 1000):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    inp = torch.empty(N, 2).uniform_(0, 1) 
    #centre du cercle en 0.5
    a = torch.sub(inp, 0.5)
    #équation de cercle
    clas = a.pow(2).sum(1).sub(1 / (2*ma.pi)).sign().div(-1).add(1).div(2).long()
    return inp, clas

def ngenerate_dataset(N = 1000):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    inp = torch.empty(N, 2).uniform_(0, 1) 
    tar = torch.empty(N, 2)
    print(tar)
    #centre du cercle en 0.5
    a = torch.sub(inp, 0.5)
    #équation de cercle
    clas = a.pow(2).sum(1).sub(1 / (2*ma.pi)).sign().div(-1).add(1).div(2).long()
    for i in range(clas.size()[0]):
        if clas[i] == 0:
            tar[i][0] = 1
        if clas[i] == 1:
            tar[i][1] = 1
    return inp, tar
