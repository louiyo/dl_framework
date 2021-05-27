import torch
import math as ma
### je fais un brouyons pour voir la structure global
from data import *
from modules import *
#upload les datas
train_input, train_target = generate_dataset(100)
test_input, test_target = generate_dataset(1000)
nb_train_samples = train_input.size(0)
#parametres du network
nb_layers = 3
nb_poids = 4
lr = 0.1/nb_train_samples
#variance des poids a definir
epsilon = 0.0001
error = []
#init 3 hidden layers
w1 = torch.empty(nb_poids, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_poids).normal_(0, epsilon)
w2 = torch.empty(nb_poids, nb_poids).normal_(0, epsilon)
b2 = torch.empty(nb_poids).normal_(0, epsilon)
w3 = torch.empty(1,nb_poids).normal_(0, epsilon)
b3 = torch.empty(1).normal_(0, epsilon)

dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())
dl_dw3 = torch.empty(w3.size())
dl_db3 = torch.empty(b3.size())

nb_epochs = 100
for epoch in range(nb_epochs):
    er = 0.
    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()
    for i in range(train_input.size()[0]):
        #print(i,epoch)
        s1 = forward_pass(train_input[i], w1, b1)
        #print('s1',s1)
        x1 = relu(s1)
        #print('x1',x1)
        s2 = forward_pass(x1, w2, b2)
        #print('s2',s2)
        x2 = relu(s2)
        #print('x2',x2)
        s3 = forward_pass(x2, w3, b3)
        #print('s3',s3)
        x3 = tanh(s3)
        #print('x3',x3)
        if (x3 <= 0.5 and train_target[i] == 1) or (x3 > 0.5 and train_target[i] == 0) :
            er = er + 1.
        backward_pass(w1, b1, w2, b2, w3, b3,
                    train_target[i],
                    train_input[i], s1, x1, s2, x2, s3, x3,
                    dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3)
    error.append(er/float(train_input.size()[0]))
    w1, b1 = optimizer(w1,b1,lr,dl_dw1,dl_db1)
    w2, b2 = optimizer(w2,b2,lr,dl_dw2,dl_db2)
    w3, b3 = optimizer(w3,b3,lr,dl_dw3,dl_db3)
        
print(error)
s1 = forward_pass(torch.transpose(test_input, 0, 1), w1, b1[:,None])
x1 = tanh(s1)
s2 = forward_pass(x1, w2, b2[:,None])
x2 = tanh(s2)
s3 = forward_pass(x2, w3, b3[:,None])
x3 = tanh(s3)
y = x3
print(loss(y,test_target))