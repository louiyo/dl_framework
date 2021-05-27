import torch
import math as ma

def generate_dataset(N = 1000):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    inp = torch.empty(N, 2).uniform_(0, 1) 
    #centre du cercle en 0.5
    a = torch.subtract(inp, 0.5)
    #équation de cercle
    clas = a.pow(2).sum(1).sub(1 / (2*ma.pi)).sign().div(-1).add(1).div(2).long()
    return inp, clas
