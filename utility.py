import torch
import math
from time import time
from plot import *


def norm(coords):
    return coords[0]**2 + coords[1]**2

def generate_dataset(N = 1000):
    
    R = 1/math.sqrt(2*math.pi)
    
    data = torch.empty(N, 2).uniform_(0, 1) 
    data_centered = data - 0.5
    
    targets = torch.Tensor([1 if norm(coord) <= R**2 else 0 for coord in data_centered])
    
    return data, targets

def normalize(X, mean, std):
    out = (X.sub(mean)).div(std)
    return out


def data_augment(train_input, train_targets, N):
    
    new_points, new_targets = augment(N)
    print(train_input.type())
    print(new_points.type())
    print(train_targets.type())
    print(new_targets.type())

    return torch.cat((train_input, new_points)), torch.cat((train_targets, new_targets))

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






def compute_performances(trials = 10, lossType = "MSE", N_normal = 1000, N_aug = 0, 
                        lr = 0.01, epochs = 200, mini_batch_size = 100, verbose = False, plot = True):
    
    his = []
    
    for i in range(trials):
        
        print("-"*50,"\nBeginning training #", i+1, "\n")        
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

        his.append(print_accuracies(train_input, train_target, test_input, 
                                    test_target, mini_batch_size))
    his = torch.Tensor(his)
    print("Average accuracies:\n", "Training = ", round(his.mean(axis=0)[0].item(), 3),
          ", Test = ", round(his.mean(axis=0)[1].item(), 3))
    if plot: plot(test_input, test_target, mini_batch_size)
    return his

def compute_accuracy(data, targets, mini_batch_size):
    correct_count, all_count = 0, 0
    for b in range(0, data.size(0), mini_batch_size):

        output = model.forward(data.narrow(0, b, mini_batch_size))
        targets_curr = targets.narrow(0, b, mini_batch_size)
        for pred, target in zip(output, targets_curr):
            if((pred >= 0.55 and target == 1) or (pred < 0.55 and target == 0)):
                #print(pred, target)
                correct_count += 1
            all_count +=1
    return correct_count/all_count


def print_accuracies(train_input, train_targets, test_input, test_targets, mini_batch_size):
    
    train_acc = compute_accuracy(train_input, train_targets, mini_batch_size)
    test_acc = compute_accuracy(test_input, test_targets, mini_batch_size)
    
    print("Training Accuracy = " , train_acc)
    print("Test Accuracy = ", test_acc)
    return train_acc, test_acc

