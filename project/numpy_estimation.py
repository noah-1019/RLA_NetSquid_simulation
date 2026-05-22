
import numpy as np
import helper_functions.RLA_functions as rla
import helper_functions.numpy_simulation as sim

# Simulation Parameters
#actual_thetas = rla.generate_thetas() # Random thetas for testing, can be changed to specific values if desired

actual_thetas = [0.35,0.001,0.35]

# Estimator Parameters
initial_guess_thetas = [0.2, 0.2, 0.2]
iterations = 100
trials_per_iteration = 100

# Load in the RLA models
models = rla.get_committee_models("project/Models/obj_committee_1")

# Initialize arrays to store values
GUESSES = [initial_guess_thetas]
PROTOCOLS = []
PROPORTIONS = []


for i in range(iterations):
    guess=GUESSES[-1]
    print(f"Iteration {i+1}/{iterations}")
    print(f"Current guess: {guess}")
    # Get optimal protocol for current guess
    _,protocol = rla.committee_predict_D(models=models, thetas=guess, is_OD=True) # Using obj immplementation

    print(f"Optimal protocol: {protocol}")
    print("\n")
    # Run simulation using the optimal protocol and actual thetas to get measurement proportions
    results = sim.simulate_depolarizing_noise(thetas=actual_thetas, protocol=protocol, trials=trials_per_iteration)

    # Update arrays
    PROTOCOLS.append(protocol)
    PROPORTIONS.append(results)

    # Plug into estimation function
    new_guess = rla.estimator_D(all_protocols=[protocol], 
                                    all_measurement_proportions=[results], 
                                    current_guess=guess, 
                                    trials_per_iteration=[trials_per_iteration])
    
    GUESSES.append(new_guess)
    
final_guess = GUESSES[-1]

print("\n")
print("-"*50)
print("Final guess:", final_guess)
print("Actual thetas:", actual_thetas)

print("Mean Square Error:", rla.mse(final_guess, actual_thetas))

    