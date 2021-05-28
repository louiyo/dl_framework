# Plotting
import matplotlib.pyplot as plt
import numpy as np


def plot(test_input, test_target, model):
    correct_labels = []
    incorrect_labels = []
    for b in range(0, test_input.size(0), mini_batch_size):
        
        output = model.forward(test_input.narrow(0, b, mini_batch_size))
        targets = test_target.narrow(0, b, mini_batch_size)
        for pred, target, point in zip(output, targets, test_input_.narrow(0,b,mini_batch_size)):
            if((pred >= 0.5 and target == 1) or (pred < 0.5 and target == 0)):
                correct_labels.append(np.array(point))
            else: 
                #print(pred)
                incorrect_labels.append(np.array(point))
    correct_labels, incorrect_labels = np.array(correct_labels), np.array(incorrect_labels)
    
    plt.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()
    plt.grid()
    plt.scatter(correct_labels[:,0], correct_labels[:,1], 
                label = "Correct labels : "+str(correct_labels.shape[0]), 
                s = 10, marker='o', color='green')
    plt.scatter(incorrect_labels[:,0], incorrect_labels[:,1], 
                label = "Incorrect labels: "+str(incorrect_labels.shape[0]),
                s = 25, marker='o', color='red')
    plt.legend(loc = 'upper right', prop={'size': 15})
    circle = plt.Circle((0.5, 0.5), (1/math.sqrt(2*math.pi)), color='black', fill=False)
    ax.add_patch(circle)
    
    plt.savefig("Labelling.png")
    plt.show()
