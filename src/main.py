import data_loader
import layer as layer
import neural_network as network

# load the training, validation, and test data
training_data, validation_data, test_data = data_loader.load_data_wrapper()

# #initialize layers
# layer1 = layer.Layer(784, "input")
# layer2 = layer.Layer(30, "sigmoid")
# layer3 = layer.Layer(10, "sigmoid")

# net = network.Network([layer1, layer2, layer3])
# net.train_network(training_data, test_data=test_data)
# net.save_model("mnist_model.json")

model = network.Network.load_model("mnist_model.json")
print(model.process(test_data[0][0]))
print(test_data[0][1])