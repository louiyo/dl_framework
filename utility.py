import torch
import math
from time import time

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

def compute_performances(trials = 10, lossType = "MSE", N_normal = 800, N_aug = 200, 
                        lr = 0.1, epochs = 50, mini_batch_size = 10, verbose = False,
                        plot = True):
    
    his = []
    
    for i in range(trials):
        
        print("Beginning training #", i+1, "\n")        
        model = Network(lr)
        
        train_input, train_target = generate_dataset(N_normal)
        test_input_, test_target = generate_dataset(N_normal)

        mean, std = train_input.mean(), train_input.std()
        train_input, train_target = data_augment(train_input, train_target, N_aug)


        train_input = normalize(train_input, mean, std)
        test_input = normalize(test_input_, mean, std)

        loss = Loss(lossType)
        
        for e in range(epochs):
            
            rand_idx = torch.randperm(train_input.size()[0])
            train_input, train_target = train_input[rand_idx], train_target[rand_idx]

            running_loss = 0
            time0 = time()

            for b in range(0, train_input.size(0), mini_batch_size):

                output = model.forward(train_input.narrow(0, b, mini_batch_size))
                cost, grad = loss.compute(output, train_target.narrow(0, b, mini_batch_size))
                model.backward(grad)   

                running_loss += cost


            if verbose: 
                print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
                print("\nTraining Time =", round(time()-time0, 2), "seconds")

        correct_count, all_count = 0, 0
        for b in range(0, test_input.size(0), mini_batch_size):

            output = model.forward(test_input.narrow(0, b, mini_batch_size))
            targets = test_target.narrow(0, b, mini_batch_size)
            for pred, target in zip(output, targets):
                if((pred >= 0.5 and target == 1) or (pred < 0.5 and target == 0)):
                    #print(pred, target)
                    correct_count += 1
                all_count +=1
        his.append(correct_count/all_count)
        print("Training Accuracy of training #",i+1," = ", correct_count/all_count, "\n\n")
    return torch.mean(torch.Tensor(his))

def data_augment(train_input, train_targets, N):
    
    new_points, new_targets = augment(N)
    print(train_input.type())
    print(new_points.type())
    print(train_targets.type())
    print(new_targets.type())

    return torch.cat((train_input, new_points)), torch.cat((train_targets, new_targets))
