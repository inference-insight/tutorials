import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

from linear_regressor import LinearRegressor
from nonlinear_regressor import NonLinearRegressor


if __name__ == "__main__":

    # read the data
    file = open('data_train.csv', 'r')
    x_train = []
    y_train = []
    for line in file.readlines():
        l = line.rstrip().split(',')
        x_train.append(float(l[0]))
        y_train.append(float(l[1]))
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)

    file = open('data_test.csv', 'r')
    x_test = []
    y_test = []
    for line in file.readlines():
        l = line.rstrip().split(',')
        x_test.append(float(l[0]))
        y_test.append(float(l[1]))
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    learningRate = 0.01 
    epochs = 5000

    inputDim = 1        # takes variable 'x' 
    outputDim = 1       # takes variable 'y'
    #model = LinearRegressor(inputDim, outputDim)
    model = NonLinearRegressor(inputDim, outputDim)
    ##### For GPU #######
    if torch.cuda.is_available():
        model.cuda()
    model = model.float()

    criterion = torch.nn.MSELoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

    for epoch in range(epochs):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            inputs = Variable(torch.from_numpy(x_train).cuda().float())
            labels = Variable(torch.from_numpy(y_train).cuda().float())
        else:
            inputs = Variable(torch.from_numpy(x_train).float())
            labels = Variable(torch.from_numpy(y_train).float())

        inputs = inputs.unsqueeze(-1)
        labels = labels.unsqueeze(-1)
        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = criterion(outputs, labels)
        print(loss)
        # get gradients w.r.t to parameters
        loss.backward()

        # update parameters
        optimizer.step()

        print('epoch {}, loss {}'.format(epoch, loss.item()))


    with torch.no_grad(): # we don't need gradients in the testing phase
        if torch.cuda.is_available():
            predicted_train = model(Variable(torch.from_numpy(x_train).cuda().float().unsqueeze(-1))).cpu().data.numpy()
            predicted_test = model(Variable(torch.from_numpy(x_test).cuda().float().unsqueeze(-1))).cpu().data.numpy()
        else:
            predicted_train = model(Variable(torch.from_numpy(x_train).float().unsqueeze(-1))).data.numpy()
            predicted_test = model(Variable(torch.from_numpy(x_test).float().unsqueeze(-1))).data.numpy()

        train_error = np.mean(np.power(np.power(y_train - predicted_train, 2), 0.5))
        test_error = np.mean(np.power(np.power(y_test - predicted_test, 2), 0.5))
        print("Train RMSE: ", train_error)
        print("Test RMSE: ", test_error)

    plt.clf()
    plt.plot(x_train, y_train, 'go', label='Train data', alpha=0.5)
    plt.plot(x_test, y_test, 'ro', label='Test data', alpha=0.5)
    plt.plot(x_test, predicted_test, 'bo', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    plt.title("Linear Regression")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()