#! /usr/bin/python3

import sys
import pennylane as qml
import autograd.numpy as np


def classify_data(X_train, Y_train, X_test):
    """Develop and train your very own variational quantum classifier.

    Use the provided training data to train your classifier. The code you write
    for this challenge should be completely contained within this function
    between the # QHACK # comment markers. The number of qubits, choice of
    variational ansatz, cost function, and optimization method are all to be
    developed by you in this function.

    Args:
        X_train (np.ndarray): An array of floats of size (250, 3) to be used as training data.
        Y_train (np.ndarray): An array of size (250,) which are the categorical labels
            associated to the training data. The categories are labeled by -1, 0, and 1.
        X_test (np.ndarray): An array of floats of (50, 3) to serve as testing data.

    Returns:
        str: The predicted categories of X_test, converted from a list of ints to a
            comma-separated string.
    """

    # Use this array to make a prediction for the labels of the data in X_test
    predictions = []

    # QHACK #

    def data_prep(data):
        """pads X-train to make it normalisable"""
        data = data / np.sqrt(len(data[0]))
        pad = np.ones(len(data)) - np.sum(data**2, axis=1)
        pad = np.sqrt(np.abs(pad)) # to avoid NaN for pad=0
        data = np.column_stack((data, pad))
        return data

    def ampl_embed4(data, wires):
        """amplitude embeds 4 data/floats in a quantum state
        of data[0] |00> + data[1] |01> + data[2] |10> + data[3] |11>
        """
        # RY(1) for data[0] |00> + data[1,2,3] |01>
        cosangle0 = data[0] / np.sqrt(np.sum(data**2))
        if np.abs(cosangle0) == 1: # otherwise gives error w/ autograd
            return None
        else:
            angle0 = 2 * np.arccos(cosangle0)
            qml.RY(angle0, wires=wires[1])

        # CRY(1,0) for data[0] |00> + data[1] |01> + data[2,3] |11>
        cosangle1 = data[1] / np.sqrt(np.sum(data[1:]**2))
        if cosangle1 == 1: # otherwise gives error w/ autograd
            return None
        elif cosangle1 == -1:
            qml.PauliZ(wires=wires[1])
            return None
        else:
            angle1 = 2 * np.arccos(cosangle1)
            qml.CRY(angle1, wires=(wires[1], wires[0]))

        # CRY(0,1) for data[0] |00> + data[1] |01> + data[2] |10> \pm data[3] |11>
        sinangle2 = -data[2] / np.sqrt(np.sum(data[2:]**2))
        if sinangle2 == 1:
            qml.CZ(wires=[wires[0], wires[1]])
            qml.CNOT(wires=(wires[0], wires[1]))
            return None
        elif sinangle2 == -1:
            qml.CNOT(wires=(wires[0], wires[1]))
            return None
        else:  # otherwise gives error w/ autograd
            angle2 = 2 * np.arcsin(sinangle2)
            qml.CRY(angle2, wires=(wires[0], wires[1]))

        # CZ(0,1) for data[0] |00> + data[1] |01> + data[2] |10> + data[3] |11>
        if data[3] < 0:
            qml.CZ(wires=[wires[0], wires[1]])

        return None

    dev = qml.device("default.qubit", wires=2)
    @qml.qnode(dev)
    def classifier(weights, data):
        """classifier circuit"""
        wires = range(2)
        # read in data
        ampl_embed4(data, wires)
        # apply classifying layers
        for weight in weights:
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=weight)
            qml.broadcast(qml.CNOT, wires, pattern="ring")
        # return prediction
        return qml.expval(qml.PauliZ(0))

    def cost(weights, data, labels):
        """cost function for classifier training"""
        predictions = [classifier(weights, dat) for dat in data]
        msq_error = np.sum((labels - predictions)**2) / len(labels)
        return msq_error

    # prepare data
    X_train = data_prep(X_train)
    X_test = data_prep(X_test)

    # classifier/training parameters
    opt = qml.GradientDescentOptimizer(1)
    num_layers = 2 # number of classifier layers
    batch_size = 10 # number of data to train on per iteration
    num_iter = 20 # number of training iterations

    # classifier training
    weights = np.random.uniform(size=(num_layers, 2, 3)) # random starting point
    for _ in range(num_iter):
        # Update the weights by one optimizer step
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = X_train[batch_index]
        Y_batch = Y_train[batch_index]
        weights = opt.step(lambda w: cost(w, X_batch, Y_batch), weights)

        # # print progress
        # curr_cost = cost(weights, X_train[:50], Y_train[:50])
        # if _ % 2 == 0:
        #     print('Iteration = {:},  Cost = {:.8f}'.format(_, curr_cost))

    # apply classifier on test data
    predictions = [int(np.round(classifier(weights, x))) for x in X_test]

    # QHACK #

    return array_to_concatenated_string(predictions)


def array_to_concatenated_string(array):
    """DO NOT MODIFY THIS FUNCTION.

    Turns an array of integers into a concatenated string of integers
    separated by commas. (Inverse of concatenated_string_to_array).
    """
    return ",".join(str(x) for x in array)


def concatenated_string_to_array(string):
    """DO NOT MODIFY THIS FUNCTION.

    Turns a concatenated string of integers separated by commas into
    an array of integers. (Inverse of array_to_concatenated_string).
    """
    return np.array([int(x) for x in string.split(",")])


def parse_input(giant_string):
    """DO NOT MODIFY THIS FUNCTION.

    Parse the input data into 3 arrays: the training data, training labels,
    and testing data.

    Dimensions of the input data are:
      - X_train: (250, 3)
      - Y_train: (250,)
      - X_test:  (50, 3)
    """
    X_train_part, Y_train_part, X_test_part = giant_string.split("XXX")

    X_train_row_strings = X_train_part.split("S")
    X_train_rows = [[float(x) for x in row.split(",")] for row in X_train_row_strings]
    X_train = np.array(X_train_rows)

    Y_train = concatenated_string_to_array(Y_train_part)

    X_test_row_strings = X_test_part.split("S")
    X_test_rows = [[float(x) for x in row.split(",")] for row in X_test_row_strings]
    X_test = np.array(X_test_rows)

    return X_train, Y_train, X_test


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    X_train, Y_train, X_test = parse_input(sys.stdin.read())
    output_string = classify_data(X_train, Y_train, X_test)
    print(f"{output_string}")