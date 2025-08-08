import gzip
import pickle
import numpy as np

def load_data():
    with gzip.open("./data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return (training_data, validation_data, test_data)

def vectorized_result(integer):
    v = np.zeros((10, 1))
    v[integer] = 1.0
    return v

def load_data_wrapper():
    tr, va, te = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr[0]]
    training_results = [vectorized_result(y) for y in tr[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va[0]]
    validation_data = list(zip(validation_inputs, va[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te[0]]
    test_data = list(zip(test_inputs, te[1]))
    return (training_data, validation_data, test_data)
