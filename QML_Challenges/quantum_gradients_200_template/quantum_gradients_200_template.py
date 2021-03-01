#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #

    shift = 1

    # find unmixed derivatives
    # 1st derivative: [f(r + 2s) - f(r - 2s)] / [2 sin(2s)]
    # 2nd derivative: [-2f(r) + f(r + 2s) + f(r - 2s)] / [4 sin^2(s)]
    fr = circuit(weights)
    for i in range(len(weights)):
        # f(r + 2s)
        weights[i] += 2 * shift
        tmp = circuit(weights)
        gradient[i] = tmp
        hessian[i,i] = tmp
        # f(r - 2s)
        weights[i] -= 4 * shift
        tmp = circuit(weights)
        gradient[i] -= tmp
        hessian[i,i] += tmp
        # f(r)
        hessian[i,i] -= 2 * fr
        # reset weights
        weights[i] += 2 * shift

    # find mixed derivatives
    # 2nd derivative: [f(r+x+y) - f(r-x+y) - f(r+x-y) + f(r-x-y)] / [4 sin^2(s)]
    for i in range(len(weights)):
        for j in range(i+1, len(weights)):
            # f(r + x + y)
            weights[i] += shift
            weights[j] += shift
            hessian[i,j] = circuit(weights)
            # f(r - x + y)
            weights[i] -= 2 * shift
            hessian[i,j] -= circuit(weights)
            # f(r - x - y)
            weights[j] -= 2 * shift
            hessian[i,j] += circuit(weights)
            # f(r + x - y)
            weights[i] += 2 * shift
            hessian[i,j] -= circuit(weights)
            # symmetric matrix
            hessian[j,i] = hessian[i,j]
            # reset weights
            weights[i] -= shift
            weights[j] += shift

    # include denominator
    gradient = gradient / (2 * np.sin(2*shift))
    hessian = hessian / (4 * np.sin(shift)**2)

    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )