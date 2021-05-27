#!/usr/bin/env python

# Any copyright is dedicated to the Public Domain.
# https://creativecommons.org/publicdomain/zero/1.0/

# Written by Francois Fleuret <francois@fleuret.org>

import math as ma
import torch

###############################################################
def generate_dataset(N = 1000):
    #Fait un tensor N*2 (ensemble de coordonnées (x,y))
    inp = torch.empty(N, 2).uniform_(0, 1) 
    tar = torch.empty(N, 2) 
    #centre du cercle en 0.5
    a = torch.sub(inp, 0.5)
    #équation de cercle
    clas = a.pow(2).sum(1).sub(1 / (2*ma.pi)).sign().div(-1).add(1).div(2).long()
    for i in range(clas.size()[0]):
        if clas[i] == 0:
            tar[i][0] = 1
        if clas[i] == 1:
            tar[i][1] = 1
    return inp, tar

######################################################################

def sigma(x):
    return x.tanh()

def dsigma(x):
    return 4 * (x.exp() + x.mul(-1).exp()).pow(-2)

######################################################################

def loss(v, t):
    return (v - t).pow(2).sum()

def dloss(v, t):
    return 2 * (v - t)

######################################################################

def forward_pass(w1, b1, w2, b2, x):
    x0 = x
    #print('0',x0)
    s1 = w1.mv(x0) + b1
    #print('s1',s1)
    x1 = sigma(s1)
    #print('1',x1)
    s2 = w2.mv(x1) + b2
    #print('s2',s1)
    x2 = sigma(s2)

    return x0, s1, x1, s2, x2

def backward_pass(w1, b1, w2, b2,
                  t,
                  x, s1, x1, s2, x2,
                  dl_dw1, dl_db1, dl_dw2, dl_db2):
    x0 = x
    dl_dx2 = dloss(x2, t)
    dl_ds2 = dsigma(s2) * dl_dx2
    dl_dx1 = w2.t().mv(dl_ds2)
    dl_ds1 = dsigma(s1) * dl_dx1

    dl_dw2.add_(dl_ds2.view(-1, 1).mm(x1.view(1, -1)))
    dl_db2.add_(dl_ds2)
    dl_dw1.add_(dl_ds1.view(-1, 1).mm(x0.view(1, -1)))
    dl_db1.add_(dl_ds1)

######################################################################
train_input, train_target = generate_dataset(100)
test_input, test_target = generate_dataset(100)
#print(train_target)

nb_classes = train_target.size(1)
nb_train_samples = train_input.size(0)
zeta = 0.90

train_target = train_target * zeta
test_target = test_target * zeta


nb_hidden = 50
eta = 0.1/ nb_train_samples
epsilon = 1e-6

w1 = torch.empty(nb_hidden, train_input.size(1)).normal_(0, epsilon)
b1 = torch.empty(nb_hidden).normal_(0, epsilon)
w2 = torch.empty(nb_classes, nb_hidden).normal_(0, epsilon)
b2 = torch.empty(nb_classes).normal_(0, epsilon)

dl_dw1 = torch.empty(w1.size())
dl_db1 = torch.empty(b1.size())
dl_dw2 = torch.empty(w2.size())
dl_db2 = torch.empty(b2.size())

for k in range(200):

    # Back-prop

    acc_loss = 0
    nb_train_errors = 0

    dl_dw1.zero_()
    dl_db1.zero_()
    dl_dw2.zero_()
    dl_db2.zero_()

    for n in range(nb_train_samples):
        x0, s1, x1, s2, x2 = forward_pass(w1, b1, w2, b2, train_input[n])
    
        pred = x2.max(0)[1].item()
        print('x2',x2)
        if train_target[n, pred] < 0.5: nb_train_errors = nb_train_errors + 1
        acc_loss = acc_loss + loss(x2, train_target[n])
        
        backward_pass(w1, b1, w2, b2,
                      train_target[n],
                      x0, s1, x1, s2, x2,
                      dl_dw1, dl_db1, dl_dw2, dl_db2)

    # Gradient step
  
    w1 = w1 - eta * dl_dw1
    b1 = b1 - eta * dl_db1
    w2 = w2 - eta * dl_dw2
    b2 = b2 - eta * dl_db2

    # Test error

    nb_test_errors = 0

    for n in range(test_input.size(0)):
        _, _, _, _, x2 = forward_pass(w1, b1, w2, b2, test_input[n])

        pred = x2.max(0)[1].item()
        if test_target[n, pred] < 0.5: nb_test_errors = nb_test_errors + 1
        
    print('{:d} acc_train_loss {:.02f} acc_train_error {:.02f}% test_error {:.02f}%'
          .format(k,
                  acc_loss,
                  (100 * nb_train_errors) / train_input.size(0),
                  (100 * nb_test_errors) / test_input.size(0)))
