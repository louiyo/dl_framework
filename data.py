import torch
import math as ma

def generate_dataset(N):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    input = torch.empty(N, 2).uniform_(0, 1) 
    #centre du cercle en 0.5
    input = torch.subtract(input, 0.5)
    #équation de cercle
    clas = input.pow(2).sum(1).sub(1 / (2*ma.pi)).sign().add(1).div(2).long()
    return input, clas



