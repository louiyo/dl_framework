import torch
import math

def generate_dataset(N = 1000):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    inp = torch.empty(N, 2).uniform_(0, 1) 
    #centre du cercle en 0.5
    a = torch.sub(inp, 0.5)
    #équation de cercle
    clas = a.pow(2).sum(1).sub(1 / (2*math.pi)).sign().div(-1).add(1).div(2)
    return inp, clas

def normalize(X, mean, std):
    out = (X.sub(mean)).div(std)
    return out

def augment(N):
    epsilon = 0.005
    coords = torch.Tensor([])
    targets = torch.Tensor([])
    for i in range(N):
        
        r = 1/math.sqrt(2*math.pi)
        random_dev = torch.empty(1).uniform_(-epsilon, epsilon).item()
        
        rand_r = r + random_dev
        rand_angle = torch.empty(1).uniform_(0, 2*math.pi)
        
        x = 0.5 + rand_r*math.cos(rand_angle)
        y = 0.5 + rand_r*math.sin(rand_angle)
        coord = torch.Tensor([x, y]).unsqueeze(1)
    
        if ((coord[0] - 0.5)**2+(coord[1] - 0.5)**2 <= r**2): 
            target = torch.Tensor([1])
        else: 
            target = torch.Tensor([0])
            
            
        coords = torch.cat((coords,coord), axis =1)
        targets = torch.cat((targets, target))
        
        
    return coords.t(), targets

def data_augment(train_input, train_targets, N):
    
    new_points, new_targets = augment(N)
    print(train_input.type())
    print(new_points.type())
    print(train_targets.type())
    print(new_targets.type())

    return torch.cat((train_input, new_points)), torch.cat((train_targets, new_targets))