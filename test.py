from time import time
from module import *
from utility import *


compute_performances(trials = 10, lossType = "MSE", N_normal = 1000, 
                     N_aug = 0, lr = 0.01, epochs = 300, mini_batch_size = 100, verbose = False)


#10 -> 0,958
#20 -> 0,954
#50 ->

#data_aug -> 0,966
#no_data_aug -> 0,9703
