import torch
import torchvision
import math
from torch import nn, optim
from torch.nn import functional as F


class Module:
    def __init__(self):
        pass
    def forward(self, *args):
        return 0
    def backward(self, *args):
        return 0


class ReLU(Module):

    def __init__(self, input_shape):
        pass

    def forward(self, x):
        self.prev_input = x
        a = x.sign().add(1).div(2).long()
        self.out = torch.mul(a,x) 
        return self.out

    def backward(self, grad, *args):
        self.grad = self.prev_input > 0
        return grad * self.grad
    
class Sigmoid(Module):

    def __init__(self, input_shape):
        pass

    def forward(self, x):
        self.prev_input = x
        self.out = 1 / (1 + torch.exp(-x))
        return self.out

    def backward(self, grad, *args):
        self.grad = grad * self.out*(1-self.out)
        return self.grad
    
class Tanh(Module):

    def __init__(self, input_shape):
        pass

    def forward(self, x):
        self.prev_input = x
        self.out = torch.tanh(x).add(1).div(2)
        return self.out

    def backward(self, grad, *args):
        self.grad = grad * (1 - torch.pow(tanh(self.out),2)).div(2)
        return self.grad
    
      
import math
class Linear(Module):
    def __init__(self, n_in, n_out, lr):
        
        stdv = 1. / math.sqrt(n_out)
        epsilon = 0.1
        self.lr = lr

        self.weights = torch.empty(n_in,n_out).uniform_(-stdv, stdv)
        self.biases = torch.empty(n_out).uniform_(-stdv, stdv)
        
        self.weights = torch.empty(n_in,n_out).normal_(0, epsilon)
        self.biases = torch.zeros(n_out)
        
    def forward(self, x):

        self.prev_input = x
        out = torch.matmul(x, self.weights) + self.biases
        self.out = out
        return out
        
    def backward(self, grad):

        x = self.prev_input
        grad_out = torch.mm(grad, self.weights.t())
        
        self.dW = torch.mm(self.prev_input.t(), grad)
        self.db = grad.mean(axis=0)*x.shape[0]

        self.weights -= self.lr * self.dW
        self.biases -= self.lr * self.db
        
        return grad_out
        


class Network(Module):
    
    def __init__(self, lr = 0.01):
        
        self.fc1 = Linear(2, 25, lr)
        self.fc2 = Linear(25, 25, lr)
        self.fc3 = Linear(25, 25, lr)
        self.fc4 = Linear(25, 1, lr)
        self.relu1 = ReLU(25)
        self.relu2 = ReLU(25)
        self.relu3 = ReLU(25)
        self.activ = Sigmoid(1) 
        
        self.layers = [
            self.fc1,
            self.relu1,
            self.fc2,
            self.relu2,
            self.fc3,
            self.relu3,
            self.fc4,
            self.activ]
           
    def forward(self, x):

        x = self.fc1.forward(x)
        x = self.relu1.forward(x)
        x = self.fc2.forward(x)
        x = self.relu2.forward(x)
        x = self.fc3.forward(x)
        x = self.relu3.forward(x)
        x = self.fc4.forward(x)
        out = self.activ.forward(x)
        
        return out
        
    def backward(self, grad):
        
        for layer_idx in range(len(self.layers))[::-1]:

            grad = self.layers[layer_idx].backward(grad)
        
        
    
class Loss(Module):

    def __init__(self, loss_type):

        if loss_type == "BCE":
            self.loss_type = "BCE"
        elif loss_type == "MSE":
            self.loss_type = "MSE"
        else:
            raise NameError("Invalid loss type, try BSE or MSE.")
            

    def compute(self, preds, targets):
        
        return self.cost(preds, targets), self.grad(preds, targets)
            

    def cost(self, preds, targets):

        if(self.loss_type == "MSE"):

            return  1/len(preds) * torch.sum((preds - targets)**2)

        elif(self.loss_type == "BCE"):
            
            targets = targets.unsqueeze(1)
            return torch.sum(preds - preds*targets + torch.log(1 + torch.exp(-torch.abs(preds))))
    
        else: return 0
        

    def grad(self, preds, targets):
        
        if(self.loss_type == "MSE"):

            out = 2 * (preds - targets.unsqueeze(1))

            return out

        elif(self.loss_type == "BCE"):

            dw = ((1/(1+torch.exp(- preds))) - targets.unsqueeze(1))

            return dw


        
