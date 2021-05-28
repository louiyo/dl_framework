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
    def derivative(self, x):
        return torch.Tensor([1 if element >= 0 else 0 for element in x])
    def backward(self, grad, x, *args):
        self.grad = grad * self.out.sign().add(1).div(2).long()
        return self.grad
    
class Sigmoid(Module):
    def __init__(self, input_shape):
        self.out = torch.empty(input_shape)
    def derivative(self, x):
        return x * (1 - x)
    def forward(self, x):
        self.prev_input = x
        self.out = 1 / (1 + torch.exp(-x))
        return self.out
    def backward(self, grad, *args):
        self.grad = grad * self.prev_input*(1-self.prev_input)
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
        
        epsilon = 0.000001
        self.lr = lr
        self.weights = torch.empty(n_in, n_out).normal_(0, epsilon)
        self.biases = torch.empty(n_out)
        
    def forward(self, x):
        #print(x,"AAAAAAAAAAAAAAAAAA")
        self.prev_input = x
        self.out = torch.matmul(x, self.weights) + self.biases  
        return self.out
        
    def backward(self, x, grad):
        grad_out = torch.mm(grad, self.weights.t())
        self.dW = torch.mm(self.prev_input.t(), grad)
        self.db = grad.mean(axis=0)*x.shape[0]
        self.weights -= self.lr * self.dW.mean(axis = 0)
        self.biases -= self.lr*self.db.mean(axis = 0)
        return grad_out
        
        
        
        
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
        x = self.relu1.forward(self.fc1.forward(x))
        #print("fc1", x)
        x = self.relu2.forward(self.fc2.forward(x))
        #print("fc2", x)
        x = self.relu3.forward(self.fc3.forward(x))
        #print(x)
        out = self.activ.forward(self.fc4.forward(x))
        
        print(out)
        return out
        
    def backward(self, grad):
        
        for layer_idx in range(len(self.layers))[::-1]:
            print(layer_idx)
            if layer_idx == 0: print("-"*60)
            layer_input = self.layers[layer_idx].prev_input
            
            grad = self.layers[layer_idx].backward(layer_input, grad)
        
        
    
class Loss(Module):
    def __init__(self, loss_type, model):
        self.model = model
        if loss_type == "BCE":
            self.loss_type = "BCE"
        elif loss_type == "MSE":
            self.loss_type = "MSE"
        else:
            raise NameError("Invalid loss type, try BSE or MSE.")
            
    def compute(self, last_layer, preds, targets):
        return self.cost(preds, targets), self.grad(preds, targets, last_layer)
            
    def cost(self, preds, targets):
        if(self.loss_type == "MSE"):
            return torch.sum((preds - targets))**2
        elif(self.loss_type == "BCE"):
            return torch.sum(torch.maximum(preds, torch.zeros(len(preds))) - 
                      preds*targets + torch.log(1 + torch.exp(-torch.abs(preds))))
        else: return 0
        
    def grad(self, preds, targets, layer = None):
        
        if(self.loss_type == "MSE"):
            assert layer != None
            m = len(preds)
            
            last_layer = self.model.layers[-1]
            x2 = self.model.layers[-2].prev_input
            s3 = self.model.layers[-1].prev_input
            
            dw = (2/m) * torch.sum((preds - targets.unsqueeze(1))*last_layer.derivative(s3)*x2)
            db = (2/m) * torch.sum((preds - targets.unsqueeze(1))*last_layer.derivative(s3))
            #print(dw, db)
            return 2 * (preds - targets.unsqueeze(1))
        elif(self.loss_type == "BCE"):
            dw = ((1/(1+torch.exp(- value))) - targets)
        
