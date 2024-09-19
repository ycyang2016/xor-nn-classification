import sys
import time
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt 
from nn import NN, FCLayer, ActivationLayer

    
# data generation function - main
def generate_data(type):
    if type == 'linear':
        X, Y = generate_linear(n=100)
    elif type == 'XOR':
        X, Y = generate_XOR_easy()

    return X, Y


# data generation function - linear
def generate_linear(n=100): 
    pts = np.random.uniform(0, 1, (n, 2)) 
    inputs = [] 
    labels = [] 
    for pt in pts: 
        inputs.append([pt[0], pt[1]]) 
        distance = (pt[0]-pt[1])/1.414 
        if pt[0] > pt[1]: 
            labels.append(0)
        else: 
            labels.append(1) 
    return np.array(inputs), np.array(labels).reshape(n, 1) 


# data generation function - XOR
def generate_XOR_easy(): 
    inputs = [] 
    labels = [] 

    for i in range(11): 
        inputs.append([0.1*i, 0.1*i]) 
        labels.append(0) 

        if 0.1*i == 0.5: 
            continue 

        inputs.append([0.1*i, 1-0.1*i]) 
        labels.append(1) 

    return np.array(inputs), np.array(labels).reshape(21, 1) 


def show_result(x, y, pred_y): 
    plt.subplot(1,2,1) 
    plt.title('Ground truth', fontsize=18) 
    for i in range(x.shape[0]): 
        if y[i] == 0: 
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1,2,2) 
    plt.title('Predict result', fontsize=18) 
            
    for i in range(x.shape[0]): 
        if pred_y[i] == 0: 
            plt.plot(x[i][0], x[i][1], 'ro')
        else: 
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.show() 


def run():
    #X, Y = generate_data('linear')
    #X, Y = generate_data('XOR')
    X = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    Y = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    print ('X shape:')
    print (X.shape)
    print ('Y shape:')
    print (Y.shape)
    network = NN()
    network.add(FCLayer(2, 3))
    activation_layer_1 = ActivationLayer()
    activation_layer_1.set_activation_func('sigmoid')
    network.add(activation_layer_1)
    network.add(FCLayer(3, 1))
    activation_layer_2 = ActivationLayer()
    activation_layer_2.set_activation_func('sigmoid')
    network.add(activation_layer_2)

    # train
    network.set_loss_func('mse')
    network.train(X, Y, epochs=100, learning_rate=0.1)

    # test
    predict_Y = network.predict(X)
    print(predict_Y)

    show_result(X, Y, predict_Y)



if __name__ == "__main__":
    start_time = time.time()

    # Start to run
    run()

    end_time = time.time()
    print ('total cost time:{cost_time}'.format(cost_time=end_time-start_time))