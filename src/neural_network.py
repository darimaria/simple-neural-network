import random
import numpy as np
import codecs, json
from layer import Layer

class Network(object):
    def __init__(self, layers):
        self.num_layers = len(layers)
        self.layers = layers
        self.initialize_layers()
        
    def initialize_layers(self):
        if self.layers[1].weights is None:
            print("initializing layers...")
            for i in range(1, self.num_layers):
                self.layers[i].weights = np.random.randn(
                    self.layers[i].num_neurons, self.layers[i-1].num_neurons
                ) #each weight matrix is an ixj array where i=num of neurons in current layer and j=num neurons in previous layer
                self.layers[i].biases = np.random.randn(self.layers[i].num_neurons, 1)
            print("done.")
            
    def train_network(self, training_data, test_data=None):
        self.SGD(training_data, 30, 10, 3.0, test_data)
    
    def process(self, x):
        """return the output of the network with x as the input"""
        a = np.argmax(self.feedforward(x))
        return a
    
    def feedforward(self, a):
        for i in range(1, self.num_layers):
            a = self.layers[i].activate(a, self.layers[i].weights, self.layers[i].biases)
        return a
    
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def backpropagate(self, x, y):
        nabla_b = [
            np.zeros(self.layers[i].biases.shape) for i in range(1, self.num_layers)
        ]
        nabla_w = [
            np.zeros(self.layers[i].weights.shape) for i in range(1, self.num_layers)
        ]
        
        #set the activations for the first layer equal to the input data since they will be unchanged
        self.layers[0].activations = x
        
        self.feedforward(x)
        z = self.layers[-1].inputs #z = (a_L-1) = (a_L-2)(w_L-1)+(b_L-1)
        
        #find the output error using the chain rule on the cost function
        # delta = C'(a(x))a'(x)
        delta = self.cost_derivative(self.layers[-1].activations, y) * self.layers[-1].activation_derivative(z)
        
        #backpropagate
        nabla_b[-1] = delta #change in biases
        nabla_w[-1] = np.dot(delta, self.layers[-2].activations.transpose()) #change in weights, delta*activation_l
        
        for l in range(2, self.num_layers):
            # delta = w_l * delta_l+1 * sigmoid'(z_l)
            delta = np.dot(self.layers[-l + 1].weights.transpose(), delta) * self.layers[-l].activation_derivative(self.layers[-l].inputs)
            prev_activations = self.layers[-l - 1].activations
            nabla_b[-l], nabla_w[-l] = self.layers[-l].backpropagate(delta, prev_activations)
        return (nabla_b, nabla_w)
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
            
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for batch in mini_batches:
                self.update_mini_batch(batch, eta)
            if test_data:
                print("Epoch {0}: {1}/{2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))
                
    def update_mini_batch(self, mini_batch, eta):
        for x, y in mini_batch:
            nabla_b, nabla_w = self.backpropagate(x, y)
            
            for i in range(1, self.num_layers):
                self.layers[i].weights -= (eta / len(mini_batch)) * nabla_w[i-1]
                self.layers[i].biases -= (eta / len(mini_batch)) * nabla_b[i-1]
                
    def cost_derivative(self, output_activations, y):
        # returns derivative of the cost function C(x) = 1/2 * (f(x) - y)^2
        return output_activations - y
    
    def error_of_layer(self, weights_of_next, delta_of_next, activation_prime_of_current):
        a = np.dot(weights_of_next.transpose(), delta_of_next)
        b = activation_prime_of_current
        return np.dot(a, b)
    
    def save_model(self, name):
        # save model to json file
        model = []
        for layer in self.layers:
            layer_info = {
                "num_neurons": layer.num_neurons,
                "activationf_name": layer.activationf_name,
                "weights": layer.weights.tolist() if layer.weights is not None else None,
                "biases": layer.biases.tolist() if layer.biases is not None else None
            }
            model.append(layer_info)
        json.dump(model, codecs.open(name, "w", encoding="utf-8"), indent=4)
        
    def load_model(name):
        #load the model from a json file
        model = json.load(codecs.open(name, "r", encoding="utf-8"))
        layers = []
        for layer_info in model:
            new_layer = Layer(layer_info["num_neurons"], layer_info["activationf_name"])
            if layer_info["weights"] is not None:
                new_layer.weights = np.array(layer_info["weights"])
            if layer_info["biases"] is not None:
                new_layer.biases = np.array(layer_info["biases"])
            layers.append(new_layer)
        return Network(layers)