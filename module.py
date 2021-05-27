class Module:
    def __init__(self):
        pass
    def forward(self, *args):
        return 0
    def backward(self, *args):
        return 0


class ReLU(Module):
    def __init__(self, input_shape):
        self.out = torch.empty(input_shape)
    def forward(self, x):
        self.prev_input = x
        a = x.sign().add(1).div(2).long()
        self.out = torch.mul(a, x)
        return self.out
    def backward(self, grad, x, *args):
        self.grad = grad * self.out.sign().add(1).div(2).long()
        return self.grad
    
class Sigmoid(Module):
    def __init__(self, input_shape):
        self.out = torch.empty(input_shape)
    def forward(self, x):
        self.prev_input = x
        self.out = 1 / (1 + torch.exp(-x))
        return self.out
    def backward(self, grad, *args):
        self.grad = grad * self.out*(1-self.out)
        return self.grad
    
class Tanh(Module):
    def __init__(self, input_shape):
        self.out = torch.empty(input_shape)
    def forward(self, x):
        self.prev_input = x
        self.out = torch.tanh(x).add(1).div(2)
        return self.out
    def backward(self, grad, *args):
        self.grad = grad * (1 - torch.pow(tanh(self.out),2)).div(2)
        return self.grad
      
class Linear(Module):
    def __init__(self, n_in, n_out, lr):
        
        epsilon = 0.001
        self.lr = lr
        self.weights = torch.empty(n_in, n_out).normal_(0, epsilon)
        self.biases = torch.empty(n_out)
        
    def forward(self, x):
        self.prev_input = x
        self.out = torch.mm(x, self.weights) + self.biases   
        return self.out
        
    def backward(self, grad, x):
        
        self.grad = torch.mm(grad, self.weights.t())
        
        print("hello", x.shape, grad.shape)
        
        self.dW = torch.matmul(x.t(), grad)
        print(grad, "class linear")
        self.db = grad.mean(axis = 1)
        print(self.db.shape, "check shape")
        assert self.weights.shape == self.dW.shape
        self.weights -= self.lr*self.dW
        self.biases -= self.lr*self.db
        
        
        
        
        
class Sequential(Module):
    def __init__(self, *args):
        self.layers = list(args)
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    """def backward(self, output, loss):
        for layer_idx in range(len(self.layers))[::-1]:
            layer_input = self.layers[layer_idx].prev_input
            self.layers[layer_idx].backward()"""
            
        


class Network(Module):
    
    def __init__(self, lr = 0.01):
        
        self.fc1 = Linear(2, 25, lr)
        self.fc2 = Linear(25, 25, lr)
        self.fc3 = Linear(25, 1, lr)
        self.relu1 = ReLU(25)
        self.relu2 = ReLU(25)
        self.activ = Sigmoid(1)
        
           
    def forward(self, x):
        x = self.relu1.forward(self.fc1.forward(x))
        x = self.relu2.forward(self.fc2.forward(x))
        out = self.activ.forward(self.fc3.forward(x))
        return out
        
    def backward(self, grad):
        for layer_idx in range(len(self.layers))[::-1]:
            layer_input = self.layers[layer_idx-1].prev_input
            self.layers[layer_idx].backward(layer_input, grad)
            grad = self.layers[layer_idx].grad
        
        
    
class Loss(Module):
    def __init__(self, loss_type):
        if loss_type == "BCE":
            self.loss_type = "BCE"
        elif loss_type == "MSE":
            self.loss_type = "MSE"
        else:
            raise NameError("Invalid loss type, try BSE or MSE.")
            
    def compute(self, x, preds, targets):
        return self.cost(preds, targets), self.grad(preds, targets, x)
            
    def cost(self, preds, targets):
        if(self.loss_type == "MSE"):
            return (preds - targets)**2
        elif(self.loss_type == "BCE"):
            return torch.sum(torch.maximum(preds, torch.zeros(len(preds))) - 
                      preds*targets + torch.log(1 + torch.exp(-torch.abs(preds))))
        else: return 0
        
    def grad(self, preds, targets, input_ = 0):
        if(self.loss_type == "MSE"):
            m = len(preds)
            print(preds.shape, targets.shape, input_.shape)
            dw = (2/m) * torch.sum(input_ * (preds - targets)
            db = (2/m) * torch.sum(input_ * (preds - targets))
            return dw, db
        elif(self.loss_type == "BCE"):
            dw = ((1/(1+torch.exp(- value))) - targets)
        
    
