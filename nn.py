import numpy as np

# activation function - sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# activation function derivative - sigmoid
def sigmoid_deriv(x):
    sigmoid = 1 / (1 + np.exp(-x))
    return sigmoid * (1 - sigmoid)

# activation function - tahh
def tanh(x):
    return np.tanh(x)

# activation function derivative - tahh
def tanh_deriv(x):
    return 1-np.tanh(x)**2

# loss function - MSE
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

# loss function derivative - MSE
def mse_deriv(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
    

class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, batch_size=1):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5


    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        # update parameters
        self.weights = self.weights - learning_rate * weights_error
        self.bias = self.bias - learning_rate * output_error
        return input_error
    
    def __str__(self):
        return 'FCLayer(Weight: {}, Bias: {})'.format(self.weights.shape, self.bias.shape)
    

class ActivationLayer(Layer):
    def __init__(self):
        self.activation = None
        self.activation_deriv = None

    def set_activation_func(self, func_name):
        if func_name == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv            
        elif func_name == 'tahh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv   

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        # print ('activation back propagation:')
        return self.activation_deriv(self.input) * output_error
    
    def __str__(self):
        return 'ActivationLayer(activation: {}, activation_deriv: {})'.format(self.activation.__name__, self.activation_deriv.__name__)

class NN:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def set_loss_func(self, func_name):
        if func_name == 'mse':
            self.loss = mse
            self.loss_deriv = mse_deriv

    # predict output for given input
    def predict(self, input_data, batch_size=None):
        # sample dimension first
        batch_size = batch_size if batch_size else len(input_data)
        pad_num = len(input_data) % batch_size if batch_size < len(input_data) else batch_size - len(input_data)
        if pad_num:
            pad_x = np.zeros((pad_num, input_data.shape[1]))
            input_data = np.concatenate((input_data, pad_x))
        # run network over all samples
        samples = len(input_data)
        result = []
        for j in range(samples):
            output = input_data[j]
            # forward propagation
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return np.concatenate(result)


    def __str__(self):
        return '\n'.join('layer {} | {}'.format(idx, l.__str__()) for idx, l in enumerate(self.layers))

    # train the network
    def train(self, x_train, y_train, epochs, learning_rate, batch_size=None, verbose=False):
        # sample dimension first
        # batch_size = batch_size if batch_size else len(x_train)
        pad_num = batch_size - len(x_train) % batch_size
        if pad_num:
            pad_x = np.zeros((pad_num, x_train.shape[1]))
            pad_y = np.zeros((pad_num, y_train.shape[1]))
            x_train = np.concatenate((x_train, pad_x))
            y_train = np.concatenate((y_train, pad_y))

        x_train = x_train.reshape(int(len(x_train)/batch_size), batch_size, x_train.shape[1])
        y_train = y_train.reshape(int(len(y_train)/batch_size), batch_size, y_train.shape[1])
        samples = len(x_train)
        # training loop
        for i in range(epochs):
            loss = 0

            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                loss += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_deriv(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            # calculate average error on all samples
            loss /= samples
            if verbose:
                print('epoch %d/%d loss: %f' % (i+1, epochs, loss))
