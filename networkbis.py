import torch
import math as ma
### je fais un brouyons pour voir la structure global
from data import *
from modules import *
#upload les datas
train_input, train_target = generate_dataset(1000)
print(train_input)
test_input, test_target = generate_dataset(1000)
nb_train_samples = train_input.size(0)
#parametres du network
nb_layers = 3
nb_poids = 25
lr = 1/nb_train_samples
#variance des poids a definir
epsilon = 0.01
error = []
#init 3 hidden layers
w1 = torch.empty(nb_poids, train_input.size(1)).uniform_(-1,-0.5)
b1 = torch.empty(nb_poids).uniform_(0.5, 1)
w2 = torch.empty(nb_poids, nb_poids).uniform_(-1,-0.5)
b2 = torch.empty(nb_poids).uniform_(0.5, 1)
w3 = torch.empty(1,nb_poids).uniform_(-1, -0.5)
b3 = torch.empty(1).uniform_(0.5, 1)
print(w1,b2)
dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())
dl_dw3 = torch.empty(w3.size())
dl_db3 = torch.empty(b3.size())

nb_epochs = 20
for epoch in range(nb_epochs):
    er = 0.
    acc_loss = 0.
    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()
    for i in range(train_input.size()[0]):
        s1 = forward_pass(train_input[i], w1, b1)
        x1 = tanh(s1)
        s2 = forward_pass(x1, w2, b2)
        x2 = tanh(s2)
        s3 = forward_pass(x2, w3, b3)
        x3 = tanh(s3)
        if (x3 <= 0.5 and train_target[i] == 1) or (x3 > 0.5 and train_target[i] == 0) :
            er = er + 1.
        acc_loss = acc_loss + loss(x3, train_target[i])
        backward_pass(w1, b1, w2, b2, w3, b3,
                    train_target[i],
                    train_input[i], s1, x1, s2, x2, s3, x3,
                    dl_dw1, dl_db1, dl_dw2, dl_db2, dl_dw3, dl_db3)
    print('acc',acc_loss)
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
print(loss(y,test_target).sum())