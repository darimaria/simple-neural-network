import numpy as np

class Layer(object):
    """a layer of neurons in a neural network"""
    def __init__(self, num_neurons, activationf_name=None):
        self.num_neurons = num_neurons
        self.activationf_name = activationf_name
        self.weights = None
        self.biases = None
        self.activations = None
        self.inputs = None
        
    def activate(self, a, w, b):
        """return the vector output of the layer if a is the input"""
        self.inputs = np.dot(w, a) + b
        self.activations = self.activation_function(self.inputs)
        return self.activations
    
    def backpropagate(self, delta, prev_activations):
        """return the gradient of the cost function with respect to the weights and biases.
        delta is the error of the next layer and weights is the weights ofthe next layer."""
        delta_b = delta
        delta_w = np.dot(delta, prev_activations.transpose())
        
        return (delta_b, delta_w) #tuple representing the change in weight and bias
    
    def activation_function(self, z):
        if self.activationf_name == "sigmoid":
            return self.sigmoid(z)
        elif self.activationf_name == "tanh":
            return self.tanh(z)
        elif self.activationf_name == "relu":
            return self.relu(z)
        elif self.activationf_name == "leaky_relu":
            return self.leaky_relu(z)
        else:
            raise ValueError(
                "Unknown activation function: {}".format(self.activation_function)
            )
            
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    def tanh(self, z):
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def leaky_relu(self, z):
        return np.where(z > 0, z, z * 0.01)
    
    def activation_derivative(self, z):
        """derivative of the activation function"""
        if self.activationf_name == "sigmoid":
            return self.sigmoid(z) * (1 - self.sigmoid(z))
        elif self.activationf_name == "tanh":
            return self.tanh(z) ** 2
        elif self.activationf_name == "relu":
            return np.where(z > 0, 1, 0)
        elif self.activationf_name == "leaky_relu":
            return np.where(z > 0, 1, 0.01)