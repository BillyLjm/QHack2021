#! /usr/bin/python3

import sys
import pennylane as qml
import autograd.numpy as np


def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    ## Subspace-Search VQE
    ## This uses SS-VQE, and it does output relatively accurate minimum energies
    ## However, it seems to be too slow/inaccurate for the hackathon

    def variational_ansatz(params, wires, basis_state=None):
        """variational ansatz circuit"""
        # initialise given basis state
        qml.BasisState(basis_state, wires=wires)
        # variationally tweak the state
        for param in params:
            qml.broadcast(qml.Rot, wires, pattern="single", parameters=param)
            qml.broadcast(qml.CNOT, wires, pattern="ring")

    def cost_ssvqe(num_eigen, *args, **kwargs):
        """cost function for SS-VQE with num_eigen orthogonal states"""
        out = []
        for i in range(num_eigen):
            # get orthogonal state |...00>, |...01>, |...10>, etc
            basis_state = format(i, "0%db" % num_qubits)
            basis_state = np.array(list(basis_state), dtype=int)
            # find cost of orthogonal state
            costfn = qml.ExpvalCost(lambda *args, **kwargs: \
                variational_ansatz(*args,**kwargs,basis_state=basis_state), H, dev)
            out.append(costfn(*args, **kwargs))
        return np.array(out)

    def cost_fn(*args, **kwargs):
        """sums up SS-VQE to train classifier"""
        return np.sum(cost_ssvqe(num_eigen, *args, **kwargs))

    # fixed variables
    num_layers = 3 # number of layers in variational ansatz
    num_iter = 500 # number of VQE iterations
    num_eigen = 3 # number of eigen-energies to output
    num_qubits = len(H.wires) # number of qubits
    threshold = 1e-5 # threshold to stop

    # initialise qnode & optimiser
    dev = qml.device("default.qubit", wires=num_qubits)
    opt = qml.GradientDescentOptimizer(0.1)

    # VQE
    params = np.random.uniform(size=(num_layers, num_qubits, 3))
    for _ in range(num_iter):
        params, prev_energy = opt.step_and_cost(cost_fn, params)
        energy = cost_fn(params)
        if np.abs(energy - prev_energy) < threshold:
            break

        # # print progress
        # if _ % 20 == 0:
        #     print('Iteration = {:},  Energy = {:.8f} Ha'.format(_, energy))

    energies = np.sort(cost_ssvqe(num_eigen, params))

    # QHACK #

    return ",".join([str(E) for E in energies])


def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
