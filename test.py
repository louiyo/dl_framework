from time import time
from module import *
from utility import *



performance = compute_performances(trials = 50, lossType = "MSE", N_normal = 700, N_aug = 300, lr = 0.01, epochs = 100, mini_batch_size = 50, verbose = False, plot = True)

print(performance)

#10 -> 0,958
#20 -> 0,954
#50 ->

#data_aug -> 0,966
#no_data_aug -> 0,9703