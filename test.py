from time import time
model = Network(0.00001)
N = 1000
train_input, train_target = generate_dataset(N)
test_input, test_target = generate_dataset(N)
mini_batch_size = 10
loss = Loss("MSE", model)
epochs = 20


for e in range(epochs):
    running_loss = 0
    time0 = time()
    for b in range(0, train_input.size(0), mini_batch_size):

        output = model.forward(train_input.narrow(0, b, mini_batch_size))
        #print(output)
        cost, grad = loss.compute(model.layers[-1], output, 
                                  train_target.narrow(0, b, mini_batch_size))
        model.backward(grad)        
        
        running_loss += cost

    print("Epoch {} - Training loss: {}".format(e+1, running_loss/len(train_input)))
    print("\nTraining Time =", round(time()-time0, 2), "seconds")
