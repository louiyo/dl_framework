def generate_dataset(N):
    input = torch.empty(1000, 2).uniform_(0, 1)
    disc = torch.pow(input[0]-0.5,2)+ torch.pow(input[1]-0.5,2) 
    
    if disc <= 1/ma.sqrt(2*ma.pi):
        target = 1
    else:
        target = 0
    return input, target

