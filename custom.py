# A custom fully connected (dense) Neural Network Model built in Python/Numpy.
# We define here the Custom_model class and the forward_propagation method.
# For the "Digit Recognizer" app we load pre-trained weights into the model.
# The complete code for training the model can be found in this Kaggle kernel:
# www.kaggle.com/szaitseff/under-the-hood-a-dense-net-w-mnist-dataset .

import numpy as np

# Decide on the model architecture: [pixels, hidden_layers, classes]
# We use a simple fully connected neural network model with 1 hidden layer:
layer_dims = [784, 512, 10]  # the model architecture is adjustable
# But the Custom_model can operate with any number of hidden layers/activation units

# Define a neural network model with fully connected layers
class Custom_model(object):

    def __init__(self, layer_dims):
        """
        The model consists of the input layer (pixels), a number of hidden layers and
        the output layer (categorical classifier). To create an instance of the model, we set
        dimensions of its layers and initialize parameters for the hidden/ output layers.
        Arguments:
            layer_dims -- list containing the input size and each layer size
        Returns:
            parameters -- python dictionary containing parameters "W1", "b1", ..., "WL", "bL":
                Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                bl -- bias vector of shape (layer_dims[l], 1)
        """
        self.layer_dims = layer_dims       # a list with dimensions of all layers
        self.num_layers = len(layer_dims)  # number of layers (with input layer)
        self.parameters = {}        # a dictionary with weights and biases of the model
        # Initializing weights randomly (He initialization) and biases to zeros
        for l in range(1, len(layer_dims)):
            self.parameters[f"W{l}"] = np.random.randn(layer_dims[l],
                                                       layer_dims[l-1])*np.sqrt(2./layer_dims[l-1])
            self.parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

    # define getters and setters for accessing the model class attributes:
    def get_layer_dims(self):
        return self.layer_dims
    def get_num_layers(self):
        return self.num_layers
    def get_params(self, key):
        return self.parameters.get(key)
    def set_params(self, key, value):
        self.parameters[key] = value


    def forward_propagation(self, X, keep_prob):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SOFTMAX computation
        Arguments:
            X -- data, numpy array of shape (input size, number of examples)
            keep_prob - probability of keeping a neuron active during drop-out, scalar
        Returns:
            AL -- last post-activation value
            caches -- list of caches containing:
               every cache of layers w/ReLU activation (there are L-1 of them, indexed from 0 to L-2)
               the cache of the output layer with Softmax activation (there is one, indexed L-1)
        """
        caches = []
        L = self.get_num_layers()-1    # number of layers with weights (hidden + output)
        A = X                          # set input as the first hidden layer activation
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A                # initialize activation of the previous layer
            W, b = self.get_params(f'W{l}'), self.get_params(f'b{l}') # get weights and biases
            Z = W.dot(A_prev) + b     # linear activation for the hidden layers
            A = np.maximum(0,Z)       # ReLU activation for the hidden layers
            if keep_prob == 1:        # if no dropout
                cache = (A_prev, Z)   # useful during backpropagation
            elif keep_prob < 1:       # if dropout is used for regularization
                D = np.random.rand(A.shape[0], A.shape[1])  # initialize matrix D
                D = D < keep_prob   # convert entries of D to 0/1 (using keep_prob as threshold)
                A *= D              # shut down some neurons of A
                A /= keep_prob      # scale the value of neurons that haven't been shutdown
                cache = (A_prev, Z, D)   # useful during backpropagation
            caches.append(cache)
        # Implement LINEAR -> SOFTMAX. Add "cache" to the "caches" list.
        W, b = self.get_params(f'W{L}'), self.get_params(f'b{L}')
        Z = W.dot(A) + b                        # Linear activation of the output layer
        Z -= np.max(Z, axis=0, keepdims=True)   # Normalize Z to make Softmax stable
        AL = np.exp(Z)/np.sum(np.exp(Z),axis=0,
                              keepdims=True) # Softmax activation of the output layer
        cache = (A, Z)                          # useful during backpropagation
        caches.append(cache)
        return AL, caches