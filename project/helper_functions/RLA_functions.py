import itertools
import numpy as np
import warnings
from pathlib import Path

from stable_baselines3 import PPO
from scipy.optimize import minimize

# Custom module import
import helper_functions.utils as u
import helper_functions.agent_enviornments as envs

# Suppress specific warnings about precision in QFIM calculations
warnings.filterwarnings(
    "ignore",
    message=".*precision lowered by casting to float32.*",
    category=UserWarning
)


# ----------------------------------------------------------
## General utility functions for RLA protocol processing
# ----------------------------------------------------------
def get_committee_models(models_dir: str) -> list:
    """Load all PPO models from a directory matching a glob pattern.

    Args:
        models_dir (str | pathlib.Path): Path to the directory containing model files.

    Returns:
        list: A list of loaded `stable_baselines3.PPO` models. Files are loaded in
        sorted order by filename. If no models are found an empty list is returned.
    """

    p = Path(models_dir)
    if not p.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    # Gets all model folders
    model_files = [str(p) for p in Path(models_dir).glob("*.zip")]
    # The above line iterates over all entries in the directory, checks if they are zip files, 
    # and collects their names into a list.

    if len(model_files) == 0:
        warnings.warn(f"No model folders found in {models_dir}")
        return []

    models = []
    for mf in model_files:
        try:
            # Loads the PPO model
            models.append(PPO.load(str(mf)))
        except Exception as e:
            warnings.warn(f"Failed to load model {mf}: {e}")

    # Returns the list of loaded models.
    return models

def protocol_to_qfim_BF(protocol: list[list[int]],thetas: list[float],h: float = 1e-5) -> np.ndarray:
    """Convert a protocol to its corresponding QFIM for the bit flip noise model.

    Parameters:
    -------------
    protocol: list[list[int]]
        The protocol to be converted, represented as a list of lists of integers. 
    thetas: list[float]
        The true thetas of the noise model.
    h: float
        The step size for numerical differentiation when calculating the QFIM. Default is 1e-5.

    Returns:
    -------------
    np.ndarray
        The QFIM corresponding to the input protocol and noise parameters, calculated using numerical differentiation.
    
    """
    return u.optimized_nodes_to_qfim(protocol,thetas,h) 

def protocol_to_qfim_D(protocol: list[list[int]],thetas: list[float],h: float = 1e-5) -> np.ndarray:
    """Convert a protocol to its corresponding QFIM for the depolarizing noise model.

    Parameters:
    -------------
    protocol: list[list[int]]
        The protocol to be converted, represented as a list of lists of integers. 
    thetas: list[float]
        The true thetas of the noise model.

    Returns:
    -------------
    np.ndarray
        The QFIM corresponding to the input protocol and noise parameters calculated using the found qfim formula.
    """
    return u.nodes_to_qfim_D(protocol,thetas)

def qfim_to_qcrb(qfim: np.ndarray,infinity: float = 1e12) -> float:
    """Convert a QFIM to its corresponding QCRB.

    Parameters:
    -------------
    qfim: np.ndarray
        The Quantum Fisher Information Matrix (QFIM) to be converted.

    Returns:
    -------------
    float
        The Quantum Cramér-Rao Bound (QCRB) corresponding to the input QFIM, calculated using numerical methods.
    
    """
    return u.calculate_QCRB_numerical(qfim,infinity)

def mse(estimate: list[float], actual: list[float]) -> float:
    """Calculate the Mean Squared Error (MSE) between an estimate and the actual values.

    Parameters:
    -------------
    estimate: list[float]
        The estimated values for the noise parameters.
    actual: list[float]
        The true values of the noise parameters.

    Returns:
    -------------
    float
        The Mean Squared Error (MSE) between the estimate and the actual values, calculated as the average of the squared differences.
    
    """
    return sum([(estimate[i] - actual[i])**2 for i in range(len(estimate))])/len(estimate)

def generate_thetas(min_val: float = 0.001, max_val: float = 0.4) -> list[float]:
    return [float(x) for x in np.random.uniform(min_val, max_val, size=3)]

# ----------------------------------------------
## RLA Prediction functions for bit flip noise
# ----------------------------------------------

def get_qcrb_direct(thetas: list[float], model):
    """ 
    Given a list of thetas, run the model in the QubitMoverEnv1_Estimator environment and
    abstract the QCRB and move (state) from the final observation.
    """

    env = envs.QubitMoverEnv1_Estimator()
    obs, info = env.reset()
    done = False

    # Override thetas in the environment
    env.thetas = thetas

    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True

    qfim = obs[-10:-1].reshape(3,3)  # Extract QFIM from observation    
    qcrb = u.calculate_QCRB_numerical(qfim) # Calculate the QCRB 

    protocol = obs[:30].reshape(3,10)  # Extract protocol (state) from observation

    return qcrb, protocol

def committee_predict_BF(models: list, thetas: list[float]):
    """
    Search a local theta stencil and return the protocol with the lowest QCRB.

    The function evaluates each model on a 3x3x3 hypercube around the input thetas,
    clamps the perturbed values to a valid range, and keeps the protocol with the best
    numerical QCRB.

    Args:
        models (list): Trained models used to propose candidate protocols.
        thetas (list[float]): Three input theta values.

    Returns:
        tuple[float, list]: The minimum QCRB value and the best protocol found.
    """

    min_qcrb = np.inf
    best_protocol = None

    perturbed_thetas_list = []
    perturbation_magnitude = 0.01  # Small perturbation

    # Create stencil of perturbed thetas around the input thetas
    

    for deltas in itertools.product([-perturbation_magnitude, 0, perturbation_magnitude], repeat=3):
        perturbed_thetas = [thetas[i] + deltas[i] for i in range(3)]
        perturbed_thetas_list.append(perturbed_thetas)


    #print("Evaluating perturbed thetas:", perturbed_thetas_list)

    # Make sure each perturbed theta is within valid bounds (e.g., non-negative)
    perturbed_thetas_list = [
        [max(0.001, theta) for theta in perturbed_thetas]
        for perturbed_thetas in perturbed_thetas_list
    ]
    perturbed_thetas_list=[
        [min(0.499, theta) for theta in perturbed_thetas]
        for perturbed_thetas in perturbed_thetas_list
    ]


        
    for perturbed_thetas in perturbed_thetas_list:
        for model in models:
            temp_qcrb,protocol = get_qcrb_direct(perturbed_thetas,model)

            # Ensure the protocol is valid
            for i,node_list in enumerate(protocol):
                path_list=u.nodes_to_paths(node_list)
                removed_0_path = [path for path in path_list if path != 0]
                # If the length of the remaining path is odd, repeat the last element to make it even
                if len(removed_0_path) % 2 != 0:
                    new_node_list = node_list.copy()
                    # Replace last 0 element with the last non 0 element
                    for j in range(len(new_node_list)-1, -1, -1):
                        if new_node_list[j] != 0:
                            new_node_list[-1] = new_node_list[j]
                            break
                    protocol[i] = new_node_list
                
                
            # Calculate QCRB for the obtained protocol
            qfim = u.optimized_nodes_to_qfim(protocol, thetas)  # Convert protocol to QFIM
            qcrb = u.calculate_QCRB_numerical(qfim) # Calculate the QCRB 

            if qcrb < min_qcrb:
                #print("New minimum QCRB found:", qcrb, "with thetas:", perturbed_thetas)
                min_qcrb = qcrb
                best_protocol = protocol  # Extract protocol (state) from observation
    
    # Remove all 0's from the best protocol
    formatted_protocols = []
    for path_list in best_protocol:
        temp_protocol = [int(p) for p in path_list if p != 0]
        # Ensure even-length paths after removing zeros
        if len(temp_protocol) % 2 != 0:
            temp_protocol.append(temp_protocol[-1])
        formatted_protocols.append(temp_protocol)

    return min_qcrb, formatted_protocols

# ----------------------------------------------
## RLA Prediction functions for depolarizing noise
# ----------------------------------------------
def get_qcrb_D(thetas:list[float], model):
    """ 
    Given a list of thetas, run the model in the QubitMoverEnv1_Estimator environment with depolarizing noise and
    abstract the QCRB and move (state) from the final observation.

    Parameters
    ----------
    thetas : list[float]
        A list of three depolarizing noise parameters corresponding to the channels in the environment.
    model : stable_baselines3.PPO
        A trained PPO model that can interact with the QubitMoverEnv1_D_Estimator environment to propose protocols based on the input thetas.

    Returns
    -------
    float :
        (qcrb) The calculated Quantum Cramér-Rao Bound (QCRB) for the protocol proposed by the model under the given noise parameters.
    list[list[int]] :
        (protocol) The protocol (as a list of lists of integers) extracted from the final observation of the environment after running the model.
    """

    env = envs.QubitMoverEnv1_D_Estimator()
    obs, info = env.reset()
    done = False

    # Override thetas in the environment
    env.thetas = thetas

    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, truncated, info = env.step(action)

        if truncated:
            done = True

   
    

    protocol = obs[:30].reshape(3,10)  # Extract protocol (state) from observation
    qfim = u.nodes_to_qfim_D(protocol, thetas)  # Convert protocol and thetas to QFIM for depolarizing noise
    qcrb = u.calculate_QCRB_numerical(qfim) # Calculate the QCRB 

    return qcrb, protocol

def get_qcrb_OD(thetas: list[float], model):
    """
    Computest the qcrb using the objective function approach. Given a list of thetas, run the model in the QubitMoverEnv1_OD_Estimator environment and abstract the QCRB and move (state) from the final observation.

    Parameters
    ----------
    thetas : list[float]
        A list of three depolarizing noise parameters corresponding to the channels in the environment.
    model : stable_baselines3.PPO
        A trained PPO model that can interact with the QubitMoverEnv1_OD_Estimator environment to propose protocols based on the input thetas.

    Returns
    -------
    float :
        (qcrb) The calculated Quantum Cramér-Rao Bound (QCRB) for the protocol proposed by the model under the given noise parameters.
    np.ndarray :
        (A) The protocol matrix proposed by the model, reconstructed from the LU representation output by the model and adjusted to satisfy parity constraints. This matrix is used to evaluate the QCRB for the proposed protocol in the objective function approach.
    
    """

    env = envs.QubitMoverEnv1_OD_Estimator()
    obs, info = env.reset()
    done = False

    # Override thetas in the environment
    env.thetas = thetas

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        thetas = obs
        # Add 1 to first 3 entries to convert to valid form.
        valid_action = action.copy()
        valid_action[0] += 1
        valid_action[1] += 1
        valid_action[2] += 1


    A = u.LU_to_A(valid_action)
    # Check to see if any rows have odd sum
    qcrb = u.A_to_qcrb(A, thetas)

    return qcrb,A


# Depolarizing noise prediction function
# --------------------------------
def committee_predict_D(models: list, thetas: list[float], is_OD: bool = False):
    """
    Given a list of trained models and input thetas, returns the best action that the models create.

        This function uses a hypercube stencil of perturbed thetas around the input thetas to evaluate 
        the models' predictions. It can also use a full 3x3x3 stencil. Supports both OD and standard 
        implementations.

    Parameters:
    -----------
    models: list
        List of trained models to evaluate.
    thetas: list[float]
        List of parameter values to evaluate the models on.
    is_OD : bool
        If True, uses an enviornment suited for the objective function approach. If False, uses the standard estimator environment.

    Returns:
    --------
    min_qcrb: float
        The minimum QCRB found across all models and perturbed thetas.
    best_protocol: list[list[int]]
        The protocol (move) corresponding to the minimum QCRB found. This will be a list
        of node lists. A node list is a list of the desired nodes to traverse for each qubit.
    """

    min_qcrb = np.inf
    best_protocol = None

    perturbed_thetas_list = []
    perturbation_magnitude = 0.01  # Small perturbation

    # Create a stencil of perturbations. This helps the agent find a better protocol by evaluating the models on a range of nearby theta values, rather than just the input thetas. This can help mitigate issues with local minima and improve generalization.
    for deltas in itertools.product([-perturbation_magnitude, 0, perturbation_magnitude], repeat=3):
        perturbed_thetas = [thetas[i] + deltas[i] for i in range(3)]
        perturbed_thetas_list.append(perturbed_thetas)

    # Make sure each perturbed theta is within valid bounds (e.g., non-negative)
    perturbed_thetas_list = [
        [max(0.001, theta) for theta in perturbed_thetas]
        for perturbed_thetas in perturbed_thetas_list
    ]
    perturbed_thetas_list=[
        [min(0.499, theta) for theta in perturbed_thetas]
        for perturbed_thetas in perturbed_thetas_list
    ]

    for perturbed_thetas in perturbed_thetas_list:
        for model in models:
            if not is_OD: # Normal Depolarizing noise approach
                try:
                    _,protocol = get_qcrb_D(perturbed_thetas,model)
                except:
                    raise ValueError("It looks like you used a standard depolarizing noise enviornment with a model trained on the objective function approach. Please set is_OD to True to use the correct environment and protocol extraction method.")
            
            else:
                    _,A = get_qcrb_OD(perturbed_thetas,model)
                    protocol = u.A_to_node_list(A)
                
                

            # Ensure the protocol is valid
            for i,node_list in enumerate(protocol):
                path_list=u.nodes_to_paths(node_list)
                removed_0_path = [path for path in path_list if path != 0]

                # If the length of the remaining path is odd, repeat the last element to make it even
                if len(removed_0_path) % 2 != 0:
                    new_node_list = node_list.copy()
                    # Replace last 0 element with the last non 0 element
                    # Find last 0 element
                    last_zero_index = np.where(new_node_list == 0)[0][-1]  
                    for j in range(len(new_node_list)-1, -1, -1):
                        if new_node_list[j] != 0:
                            new_node_list[last_zero_index] = new_node_list[j]
                            break
                    protocol[i] = new_node_list
                
                
            # Calculate QCRB for the obtained protocol
            qfim = u.nodes_to_qfim_D(protocol, thetas)  # Convert protocol to QFIM
            qcrb = u.calculate_QCRB_numerical(qfim) # Calculate the QCRB 

            if qcrb < min_qcrb:
                #print("New minimum QCRB found:", qcrb, "with thetas:", perturbed_thetas)
                min_qcrb = qcrb
                best_protocol = protocol  # Extract protocol (state) from observation
    
    # Remove all 0's from the best protocol
    formatted_protocols = []
    for node_list in best_protocol:
        temp_protocol = [int(n) for n in node_list if n != 0]
        # Ensure even-length paths after removing zeros
        if len(temp_protocol) % 2 != 0:
            temp_protocol.append(temp_protocol[-1])
        formatted_protocols.append(temp_protocol)

    return min_qcrb, formatted_protocols


# ----------------------------------------------
## Estimation Functions
# ----------------------------------------------

# Bit Flip noise model estimation function
# -----------------------------------------------
def estimator_BF(all_protocols: list[list[list[int]]],
                all_measurement_proportions: list[list[float]],
                current_guess: list[float],
                trials_per_iteration: list[int],
                tolerance:float = 1e-6
                        )->list[float]:
    """Estimate noise parameters using maximum likelihood estimation for bit-flip noise.

    Uses the negative log-likelihood of measurement outcomes to optimize the noise
    parameters for a bit-flip error model. Performs gradient-based minimization with
    bounds constraints.

    Parameters
    ----------
    all_protocols : list[list[list[int]]]
        List of protocols, where each protocol is a list of node lists (one per qubit).
        Each node list contains integers representing the quantum operations to apply.
    all_measurement_proportions : list[list[float]]
        Measurement outcome proportions for each protocol. For each protocol, a list
        of proportions (one per qubit) indicating the fraction of |0> measurements.
    current_guess : list[float]
        Initial estimate for the three noise parameters [p1, p2, p3].
    num_tials_per : list[int]
        Number of trials (shots) for each protocol.
    tolerance : float, optional
        Convergence tolerance for the optimization routine. Default is 1e-6.

    Returns
    -------
    ndarray
        Optimized noise parameters [p1, p2, p3] that minimize the negative
        log-likelihood. Values are constrained to [0.001, 0.4].

    Notes
    -----
    - The bit-flip model assumes measurement probabilities depend on the parity of
      noise applications across the three channels.
    - Parameters are constrained to [0.001, 0.4] to maintain numerical stability.
    - Uses the L-BFGS-B method with analytical gradients for optimization.
    """

    num_tials_per = trials_per_iteration
    guess = current_guess # renaming for clarity
    measurement_proportions = all_measurement_proportions # renaming for clarity
    node_lists = all_protocols # renaming for clarity

    # Convert protocols to NLL
    # ------------------------------
    residual_specs = []  # List to hold (function, proportion, number of trials)
                                # [p_i, k_i, n_i]


    # Iterate through each protocol
    for i,trial in enumerate(node_lists): # Each protocol that is iterated over
        proportions = measurement_proportions[i]
        n_i = num_tials_per[i]
        for j,qubit_moves in enumerate(trial): # Each qubit move in the protocol

            temp_paths = u.nodes_to_paths(qubit_moves)
            filtered_path = [k for k in temp_paths if k != 0]  # Remove identity gates
            k_i = n_i * proportions[j] # Number of successful trials
            p_i, _ = u.make_equation(filtered_path)
            

            # Append the function, target value, and weight to residual_specs
            residual_specs.append((p_i, k_i, n_i,filtered_path))
            

    # Define the NLL Function
    # ------------------------------
    # This estimator will choose an x1, x2, x3 that minimizes the NLL


    def NLL(x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        res = []
        total = 0.0
        gradient_totals = [0.0,0.0,0.0]
        for p_j, k_j, n_j, filtered_path in residual_specs:
                # Evaluate probability
                p = p_j(x1, x2, x3)

                # Convert potential numpy scalar to float
                p = float(np.squeeze(p))


                grad = u.grad_p(filtered_path,[x1,x2,x3])
                w = (n_j*p - k_j) / (p*(1.0 - p))   # scalar weight

                for idx in range(3):
                    gradient_totals[idx] += w * grad[idx]
                

                
                # Accumulate NEGATIVE log-likelihood contribution
                total -= k_j * np.log(p) + (n_j - k_j) * np.log(1.0 - p)
        
        return total, np.array(gradient_totals)

    # Initial guess
    # Find the root
    sol = minimize(NLL, guess,bounds=([(0.001,0.4),(0.001,0.4),(0.001,0.4)]),tol=tolerance,jac=True,method='L-BFGS-B')

    return(sol.x)


# Depolarizing noise model estimation function
# -----------------------------------------------
def estimator_D(all_protocols: list[list[list[int]]],
                all_measurement_proportions: list[list[float]],
                current_guess: list[float],
                trials_per_iteration: list[int],
                tolerance:float = 1e-6
                        )->list[float]:
    """Estimate noise parameters using maximum likelihood estimation for depolarizing noise.

    Uses the negative log-likelihood of measurement outcomes to optimize the noise
    parameters for a depolarizing error model. Performs gradient-based minimization
    with bounds constraints.

    Parameters
    ----------
    all_protocols : list[list[list[int]]]
        List of protocols, where each protocol is a list of node lists (one per qubit).
        Each node list contains integers representing the quantum operations to apply.
    all_measurement_proportions : list[list[float]]
        Measurement outcome proportions for each protocol. For each protocol, a list
        of proportions (one per qubit) indicating the fraction of |0> measurements.
    current_guess : list[float]
        Initial estimate for the three noise parameters [theta1, theta2, theta3].
    num_tials_per : list[int] or int
        Number of trials (shots) for each protocol. Either a list with one entry
        per protocol, or a single integer applied to all protocols.
    tolerance : float, optional
        Convergence tolerance for the optimization routine. Default is 1e-6.

    Returns
    -------
    ndarray
        Optimized noise parameters [theta1, theta2, theta3] that minimize the
        negative log-likelihood. Values are constrained to [0.001, 0.4].

    Notes
    -----
    - The depolarizing model assumes measurement probabilities depend on the product
      Q = prod_j (1 - 4/3 * theta_j)^(N_j) across the three channels.
    - Parameters are constrained to [0.001, 0.4] to maintain numerical stability.
    - Uses the L-BFGS-B method with analytical gradients for optimization.
    - Automatically broadcasts num_tials_per if provided as a scalar integer.
    """

    num_tials_per = trials_per_iteration
    guess = current_guess # renaming for clarity
    # clip guess to valid range
    guess = [max(0.001, theta) for theta in guess]
    guess = [min(0.4, theta) for theta in guess]

    measurement_proportions = all_measurement_proportions # renaming for clarity
    node_lists = all_protocols # renaming for clarity

    # If the number of trials per protocol is given as a single integer, convert it to a list of the same integer for each protocol
    if not isinstance(num_tials_per, list) and isinstance(num_tials_per, int):
        num_tials_per = [num_tials_per] * len(node_lists)
        
    # Convert protocols to NLL
    # ------------------------------
    residual_specs = []  # List to hold (function, proportion, number of trials)
                                # [p_i, k_i, n_i,filtered_path]
    # Iterate through each protocol
    for i,trial in enumerate(node_lists): # Each protocol that is iterated over
        proportions = measurement_proportions[i]
        n_i = num_tials_per[i]
        for j,qubit_moves in enumerate(trial): # Each qubit move in the protocol

            temp_paths = u.nodes_to_paths(qubit_moves)
            filtered_path = [k for k in temp_paths if k != 0]  # Remove identity gates
            k_i = n_i * proportions[j] # Number of successful trials
            p_i, _ = u.make_equation_D(filtered_path)

            # Append the function, target value, and weight to residual_specs
            residual_specs.append((p_i, k_i, n_i, filtered_path))
            

    # Define the NLL Function
    # ------------------------------
    # This estimator will choose an x1, x2, x3 that minimizes the NLL



    def NLL(x: np.ndarray) -> np.ndarray:
        x1, x2, x3 = x
        res = []
        total = 0.0
        gradient_totals = [0.0,0.0,0.0]
        for p_j, k_j, n_j, filtered_path in residual_specs:
                # Evaluate probability
                p = p_j(x1, x2, x3)

                # Convert potential numpy scalar to float
                p = float(np.squeeze(p))


                grad = u.grad_p_D(filtered_path,[x1,x2,x3])
                w = (n_j*p - k_j) / (p*(1.0 - p)+1e-10)   # scalar weight

                for idx in range(3):
                    gradient_totals[idx] += w * grad[idx]
                

                
                # Accumulate NEGATIVE log-likelihood contribution
                total -= k_j * np.log(p) + (n_j - k_j) * np.log(1.0 - p)
        
        return total, np.array(gradient_totals)

    # Initial guess
    # Find the root
    sol = minimize(NLL, guess,bounds=([(0.001,0.4),(0.001,0.4),(0.001,0.4)]),tol=tolerance,jac=True,method='L-BFGS-B')

    return(sol.x)



    