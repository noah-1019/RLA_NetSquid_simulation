"""
Trials per iteration experiment: Run multiple trials of the RLA estimation process for different numbers of iterations, while keeping the total qubit budget constant. This will allow us to see how the number of iterations (and thus the number of trials per iteration) affects the convergence and accuracy of the estimation process. We can plot the average guess vs actual theta over iterations for each trial to visualize the convergence behavior.
"""
import numpy as np
import helper_functions.RLA_functions as rla
import helper_functions.numpy_simulation as sim
import helper_functions.numpy_protocols as pmp

import matplotlib.pyplot as plt
import csv
from datetime import datetime
from pathlib import Path


# Simulation Parameters
#actual_thetas = rla.generate_thetas() # Random thetas for testing, can be changed to specific values if desired

actual_thetas = [0.15,0.15,0.15]
qubit_budget = 10_000
pmp_rate = 100

# Estimator Parameters for multiple trials
initial_guess_thetas = [0.2, 0.2, 0.2]
iterations = [10,100] # Number of iterations for each trial
trials_per_iteration = [qubit_budget//x for x in iterations] # Number of trials per iteration, calculated based on the total qubit budget and number of iterations

# Load in the RLA models
models = rla.get_committee_models("project/Models/obj_committee_1")

# Create an output directory for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path("data/numpy_simulations") / f"{timestamp}_qubits_{qubit_budget}"
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize arrays to store values
GUESSES_matrix = []
PROTOCOLS_matrix = []
PROPORTIONS_matrix = []
TRIALS_matrix = []

average_guesses_matrix = []

# compute pre made protocols results
MULTICAST_GUESSES=[initial_guess_thetas]


BF_GUESSES=[initial_guess_thetas]


n2_GUESSES=[initial_guess_thetas]

n6_GUESSES=[initial_guess_thetas]

pmp_qubit_axis = []

for i in range(10,qubit_budget,pmp_rate):
    guess_m = pmp.estimate_multicast_D(trials = i,
                                     theta_0=actual_thetas[0],
                                     theta_1=actual_thetas[1],
                                     theta_2=actual_thetas[2],
                                     guess=initial_guess_thetas)
    MULTICAST_GUESSES.append(np.mean(guess_m, axis=0))
   

    guess_bf = pmp.estimate_back_and_forth_z_D(trials = i,
                                                theta_0=actual_thetas[0],
                                                theta_1=actual_thetas[1],
                                                theta_2=actual_thetas[2],
                                                guess=initial_guess_thetas)
    BF_GUESSES.append(np.mean(guess_bf, axis=0))


    guess_n2 = pmp.estimate_repeated_unicast_D(theta_0=actual_thetas[0],
                                                theta_1=actual_thetas[1],
                                                theta_2=actual_thetas[2],
                                                n=2,
                                                trials=i,
                                                guess=initial_guess_thetas)
    n2_GUESSES.append(np.mean(guess_n2, axis=0))


    guess_n6 = pmp.estimate_repeated_unicast_D(theta_0=actual_thetas[0],
                                                theta_1=actual_thetas[1],
                                                theta_2=actual_thetas[2],
                                                n=6,
                                                trials=i,
                                                guess=initial_guess_thetas)
    n6_GUESSES.append(np.mean(guess_n6, axis=0))


    pmp_qubit_axis.append(i)



for trial in range(len(iterations)):
    print(f"Starting trial with {iterations[trial]} iterations and {trials_per_iteration[trial]} trials per iteration.")
    GUESSES = [initial_guess_thetas]
    PROTOCOLS = []
    PROPORTIONS = []
    TRIALS = []
    for i in range(iterations[trial]):
        if i % 10 == 0:
            print(f"Iteration {i}/{iterations[trial]}")
        guess=GUESSES[-1]
        # print(f"Iteration {i+1}/{iterations}")
        # print(f"Current guess: {guess}")

        # Get optimal protocol for current guess
        _,protocol = rla.committee_predict_D(models=models, thetas=guess, is_OD=True) # Using obj immplementation

        # print(f"Optimal protocol: {protocol}")
        # print("\n")
        # Run simulation using the optimal protocol and actual thetas to get measurement proportions
        results = sim.simulate_depolarizing_noise(thetas=actual_thetas, protocol=protocol, trials=trials_per_iteration[trial])

        # Update arrays
        PROTOCOLS.append(protocol)
        PROPORTIONS.append(results)
        TRIALS.append(trials_per_iteration[trial])
        

        # Plug into estimation function
        new_guess = rla.estimator_D(all_protocols=PROTOCOLS, 
                                        all_measurement_proportions=PROPORTIONS, 
                                        current_guess=guess, 
                                        trials_per_iteration=TRIALS)
        
        GUESSES.append(new_guess)

    # Aggregate results for this trial
    GUESSES_matrix.append(GUESSES)
    PROTOCOLS_matrix.append(PROTOCOLS)
    PROPORTIONS_matrix.append(PROPORTIONS)
    TRIALS_matrix.append(TRIALS)
        

    avg_guess = np.mean(GUESSES, axis=1)

    average_guesses_matrix.append(avg_guess)


# Make dot plot of average guess vs actual theta over iterations
qubit_axis = [list(np.linspace(0, qubit_budget, iterations[i]+1)) for i in range(len(iterations))]

# Save one CSV per experiment using the iteration count in the filename
for trial in range(len(iterations)):
    csv_path = output_dir / f"iterations_{iterations[trial]}.csv"
    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["qubits", "average_guess"])
        writer.writerows(zip(qubit_axis[trial], average_guesses_matrix[trial]))





markers = ['o', 's', '^']  # Different markers for each trial   
plt.figure(figsize=(12,6))
for i in range(len(iterations)):
    plt.plot(qubit_axis[i], average_guesses_matrix[i], label=f"Average Guess (Iterations={iterations[i]})",marker=markers[i%len(markers)],linestyle='')

# Plot the pre made protocol results
plt.plot(pmp_qubit_axis, MULTICAST_GUESSES[1:], label="Multicast Protocol", marker='x', linestyle='--')
plt.plot(pmp_qubit_axis, BF_GUESSES[1:], label="Back and Forth Z Protocol", marker='x', linestyle='--')
plt.plot(pmp_qubit_axis, n2_GUESSES[1:], label="Repeated Unicast n=2 Protocol", marker='x', linestyle='--')
plt.plot(pmp_qubit_axis, n6_GUESSES[1:], label="Repeated Unicast n=6 Protocol", marker='x', linestyle='--')

plt.axhline(y=actual_thetas[0], color='r', linestyle='--', label="Actual Theta")
plt.xlabel("Number of qubits")
plt.ylabel("Theta Value")
plt.title(f"Average Guess vs Actual Theta Over Iterations \n (Qubit Budget={qubit_budget})")
plt.legend()
plt.show()


    