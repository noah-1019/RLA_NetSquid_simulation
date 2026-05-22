"""
numpy_simulation.py
=======================

Simulate the quantum network using numpy, this is a simple less robust version of what will eventually be implemented in NetSquid. This is used for testing the protocol extraction and evaluation code without having to run the full NetSquid simulations, which can be time consuming. The goal is to have this code produce the same results as the NetSquid simulations, but much faster. This will allow us to quickly test the protocol extraction and evaluation code before running the full NetSquid simulations.
"""
import numpy as np
import helper_functions.utils as utils

def simulate_bit_flip(thetas: list[float], protocol:list[list[int]],trials: int = 1000) -> list[tuple[int,float]]:
    """
    Simulate the bit flip noise model using numpy. This is a simple simulation that applies the bit flip noise to the protocol and returns the estimated thetas.

    Parameters:
    -------------
    thetas: list[float]
        The true thetas of the noise model.
    protocol: list[list[int]]
        The protocol to be simulated, represented as a list of lists of integers.
    trials: int
        The number of trials to be simulated.

    Returns:
    -------------
    list[float]
        The proportion of successful states (0) measured for each qubit, which can be used to estimate the thetas. Each element in the list corresponds to a qubit and is a tuple of (number of qubits used, amount of successful states measured).
    """

    rng = np.random.default_rng()

    states=[]
    for i in range(3):
        path_list=utils.nodes_to_paths(protocol[i][:])
        counts=utils._counts_123(path_list)
        states.append(utils._rho_from_counts_and_params(counts[0], counts[1], counts[2], 
                                                            thetas[0], thetas[1], thetas[2]))

    
    p0_list = [states[i][0,0] for i in range(3)]  # Projector onto |0><0| for each state

    iteration_measurements = [0,0,0] # Store measurements for each iteration (group of trials)
    for _ in range(trials):
        for i in range(3): # Loop over the three states
            p0 = p0_list[i]
            # Clip numerics to [0,1] for near-pure / low-theta edge cases
            p0 = min(1.0, max(0.0, p0))
            outcome = 1 if rng.random() < p0 else 0 # increment the sucess counter.

            iteration_measurements[i]+= outcome  # Increment the count of successful states (0) measured


    # Convert counts to tuples of (number of qubits used, amount of successful states measured)
    results = [iteration_measurements[i]/trials for i in range(3)]
    return results

def simulate_depolarizing_noise(thetas: list[float], protocol:list[list[int]],trials: int = 1000) -> list[tuple[int,float]]:
    """
    Simulate the depolarizing noise model using numpy. This is a simple simulation that applies the depolarizing noise to the protocol and returns the estimated thetas.

    Uses the depolarizing noise formula directly:
    p(0) = 0.5 * (1 + Q), where Q = product_j (1 - 4/3 * theta_j)^(N_j)

    Parameters:
    -------------
    thetas: list[float]
        The true thetas of the noise model.
    protocol: list[list[int]]
        The protocol to be simulated, represented as a list of lists of integers.
    trials: int
        The number of trials to be simulated.

    Returns:
    -------------
    list[tuple[int,float]]
        A list of tuples, where each tuple contains the number of qubits used and the amount of successful states (0) measured.
    """

    rng = np.random.default_rng()
    
    # Convert nodes to paths and extract probability functions
    path_lists = []
    probability_funcs = []
    
    for i in range(3):
        path_list = utils.nodes_to_paths(protocol[i][:])
        path_lists.append(path_list)
        p0, _ = utils.make_equation_D(path_list)
        # Gets the function for p(0) for this qubit, which depends on the path taken and thetas. We only need p(0) since p(1) = 1 - p(0).
        probability_funcs.append(p0)

    # Initialize measurement counts
    num_rho0 = [0, 0, 0]
    
    for _ in range(trials):
        measurements = []
        for i in range(3):
            # Compute probability of measuring |0> directly using depolarizing formula
            p0 = probability_funcs[i](thetas[0], thetas[1], thetas[2])
            # Clip to [0,1] for numerical safety
            p0 = min(1.0, max(0.0, p0))
            # Sample outcome
            outcome = 0 if rng.random() < p0 else 1
            measurements.append(outcome)
        
        # Count |0> measurements
        for i in range(3):
            if measurements[i] == 0:
                num_rho0[i] += 1
    
    # Convert counts to tuples of (number of qubits used, amount of successful states measured)
    results = [num_rho0[i]/trials for i in range(3)]
    return results
    
