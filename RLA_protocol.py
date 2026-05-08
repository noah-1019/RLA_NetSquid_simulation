import itertools
import math
import numpy as np
import warnings
from pathlib import Path

from stable_baselines3 import PPO
from scipy.optimize import root, least_squares, minimize

# Custom module import
import agent_enviornments as envs

# Suppress specific warnings about precision in QFIM calculations
warnings.filterwarnings(
    "ignore",
    message=".*precision lowered by casting to float32.*",
    category=UserWarning
)


# -------------------------------------
## Intitalize constants
# -------------------------------------
_X = np.array([[0, 1],
               [1, 0]], dtype=np.complex128)

_R0 = np.array([[1, 0],
                [0, 0]], dtype=np.complex128)

_RXR = _X @ _R0 @ _X  # == [[0,0],[0,1]]


# ----------------------------------------------------------
## General utility functions for RLA protocol processing
# ----------------------------------------------------------
def nodes_to_paths(node_list,debug: bool=False):
    """
    Convert a node-list action into the path representation used by the protocol.

    The first node is kept once, later nodes are duplicated, zero entries are removed,
    and the final trailing element is dropped to match the expected path format.

    Args:
        node_list (list[int]): Sequence of node identifiers for a protocol action.
        debug (bool): Unused debug flag retained for compatibility.

    Returns:
        list[int] | int: The path list after conversion, or -1 if the move set is
        invalid.

    Notes:
        A single-node list is considered invalid, and an all-zero move set is also
        rejected.
    """
    path_list=[]
    path_list.append(node_list[0])

    if len(node_list)==1: # Ensures that single node lists are handled correctly
        print("ERROR: Single node list detected")
        return -1

    for i in range(1,len(node_list)):
        path_list.append(node_list[i])
        path_list.append(node_list[i])

    filtered_list = [x for x in path_list if x != 0]
    if len(filtered_list) == 0:
        return -1 # Invalid move if all elements are zero
    
    filtered_list.pop()
    return filtered_list

def _counts_123(path_list) -> tuple[int, int, int]:
    """Count how many 1, 2, and 3 values appear in a path list.

    Zero entries are ignored because they represent no-op positions.

    Args:
        path_list (list[int]): Path values containing entries in {0, 1, 2, 3}.

    Returns:
        tuple[int, int, int]: A three-tuple ``(n1, n2, n3)`` containing the counts
        of 1s, 2s, and 3s in the list.
    """
    # path_list has values in {0,1,2,3}; ignore zeros (no-op).
    n1 = path_list.count(1)
    n2 = path_list.count(2)
    n3 = path_list.count(3)
    return n1, n2, n3

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

def is_singular_qfim(F, tol=1e-12):
    # Check if eigenvalues are below tolerance
    eigenvalues = np.linalg.eigvalsh(F)
    if np.any(eigenvalues < tol):
        return True

    # Check if the matrix is positive definite using Cholesky decomposition
    try:
        np.linalg.cholesky(F)
        return False  # Positive definite
    except np.linalg.LinAlgError:
        # Check eigenvalues for near-singular
        return True

def calculate_QCRB_numerical(F: np.ndarray, infinity = 1e10) -> float:
    """
    Calculate the Quantum Cramér-Rao Bound (QCRB) from the Quantum Fisher Information Matrix (QFIM).

    """
    # Symmetrize the QFIM to mitigate numerical asymmetry
    F = 0.5 * (F + F.T)

    if is_singular_qfim(F):
        return infinity
    
    # Calculate the inverse of the QFIM
    F_inv = np.linalg.inv(F)


    qcrb = np.trace(F_inv)
    # If the matrix has large magnitude and is negative return infinity
    if np.abs(qcrb) > infinity and qcrb < 0:
        return infinity
    
    if qcrb < 0:
        return np.abs(qcrb)
    
    # Return the transpose of the inverse as the QCRB
    return np.trace(F_inv)

# ----------------------------------------------
## RLA Prediction functions for bit flip noise
# ----------------------------------------------

def _rho_from_counts_and_params(n1, n2, n3, p1, p2, p3, out_dtype=np.complex128):
    """Build the 2x2 density matrix for a given count tuple and noise parameters.

    Args:
        n1 (int): Number of 1 entries in the path.
        n2 (int): Number of 2 entries in the path.
        n3 (int): Number of 3 entries in the path.
        p1 (float): Noise parameter associated with count 1.
        p2 (float): Noise parameter associated with count 2.
        p3 (float): Noise parameter associated with count 3.
        out_dtype (numpy.dtype): Output dtype for the returned matrix.

    Returns:
        numpy.ndarray: A 2x2 complex density matrix with shape ``(2, 2)`` and dtype
        ``out_dtype``.
    """
    s = (1.0 - 2.0*p1)**n1 * (1.0 - 2.0*p2)**n2 * (1.0 - 2.0*p3)**n3
    a = 0.5*(1.0 + s)
    b = 0.5*(1.0 - s)
    # 2x2 array (Hermitian, PSD, trace 1)
    return (a * _R0 + b * _RXR).astype(out_dtype, copy=False)

def _rho7_from_counts_arrays(counts: tuple[int,int,int], params: list[float], h: float):
    """Construct the central and finite-difference perturbed density matrices.

    The return value preserves the original nested layout expected by the QFIM code.

    Args:
        counts (tuple[int, int, int]): The ``(n1, n2, n3)`` count tuple.
        params (list[float]): Three noise parameters ``[p1, p2, p3]``.
        h (float): Finite-difference step size.

    Returns:
        list[list[numpy.ndarray]]: Nested density matrices in the form
        ``[[rho], [rho1low, rho1high], [rho2low, rho2high], [rho3low, rho3high]]``,
        where each matrix has shape ``(2, 2)``.
    """
    n1, n2, n3 = counts
    p1, p2, p3 = map(float, params)
    rho      = _rho_from_counts_and_params(n1, n2, n3, p1,     p2,     p3)
    rho1low  = _rho_from_counts_and_params(n1, n2, n3, p1 - h, p2,     p3)
    rho1high = _rho_from_counts_and_params(n1, n2, n3, p1 + h, p2,     p3)
    rho2low  = _rho_from_counts_and_params(n1, n2, n3, p1,     p2 - h, p3)
    rho2high = _rho_from_counts_and_params(n1, n2, n3, p1,     p2 + h, p3)
    rho3low  = _rho_from_counts_and_params(n1, n2, n3, p1,     p2,     p3 - h)
    rho3high = _rho_from_counts_and_params(n1, n2, n3, p1,     p2,     p3 + h)
    # IMPORTANT: match original nesting exactly
    return [[rho], [rho1low, rho1high], [rho2low, rho2high], [rho3low, rho3high]]

def calculate_QFIM_direct_singular(rho_lists: list[np.ndarray],h=1e-5,lam_tol=1e-10):
    """
    Calculate QFIM for a single density matrix using direct numerical methods.
    
    This function computes the QFIM using pre-computed perturbed density matrices
    and numerical differentiation. It's designed for maximum efficiency in RL
    environments where the same perturbations are used repeatedly.
    
    Args:
        rho_lists (list[np.ndarray]): Pre-computed density matrices in format:
            [
                [rho_base],                    # Base density matrix at current parameters
                [rho1_low, rho1_high],        # Perturbed matrices for parameter 1
                [rho2_low, rho2_high],        # Perturbed matrices for parameter 2  
                [rho3_low, rho3_high]         # Perturbed matrices for parameter 3
            ]
        h (float, optional): Step size used for perturbations. Defaults to 1e-5.
        lam_tol (float, optional): Tolerance for eigenvalue filtering. Defaults to 1e-10.
    Returns:
        np.ndarray: 3x3 QFIM matrix for this single density matrix
    """

    # Unpack
    rho  = rho_lists[0][0]
    rho1low, rho1high = rho_lists[1]
    rho2low, rho2high = rho_lists[2]
    rho3low, rho3high = rho_lists[3]

    # Base eigendecomposition (1 call)
    evals, evecs = np.linalg.eigh(rho)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Derivative density matrices
    dp1 = (rho1high - rho1low) / (2*h)
    dp2 = (rho2high - rho2low) / (2*h)
    dp3 = (rho3high - rho3low) / (2*h)

    # Project derivatives onto eigenvectors (vectorized over i)
    # der_p[i] = v_i^H (dρ_p) v_i
    # einsum pattern: 'ik,kl,il->i' uses conj(evecs) * dρ * evecs
    der1 = np.real(np.einsum('ik,kl,il->i', evecs.conj(), dp1, evecs, optimize=True))
    der2 = np.real(np.einsum('ik,kl,il->i', evecs.conj(), dp2, evecs, optimize=True))
    der3 = np.real(np.einsum('ik,kl,il->i', evecs.conj(), dp3, evecs, optimize=True))

    # Keep only valid eigenvalues (avoid divide-by-zero)
    mask = evals > lam_tol
    evals_valid = evals[mask]
    D = np.empty((3, evals_valid.size), dtype=evals.dtype)
    D[0] = der1[mask]
    D[1] = der2[mask]
    D[2] = der3[mask]

    # Vectorized QFIM: F = D @ diag(1/λ) @ D^T
    F = D @ (D / evals_valid).T
    return F

def calculate_QFIM_direct(rho_list:list[np.ndarray],h=1e-5):
    """
    Calculate the total QFIM by summing contributions from multiple density matrices.
    
    This function computes the QFIM for a quantum protocol involving multiple qubits
    by summing the Fisher information contributions from each qubit's density matrix.
    
    Mathematical Formula:
        F_total = Σ_i F_i
        
    Where F_i is the QFIM contribution from the i-th density matrix.
    
    Args:
        rho_list (list[np.ndarray]): List of pre-computed density matrix sets.
                                    Each element should be in the format expected 
                                    by calculate_QFIM_direct_singular().
        h (float, optional): Step size for numerical differentiation. Defaults to 1e-5.
    
    Returns:
        np.ndarray: 3×3 total QFIM matrix summing all contributions
    
    Example:
        >>> # For a 3-qubit protocol
        >>> rho_list = [rho_qubit1, rho_qubit2, rho_qubit3]
        >>> F_total = calculate_QFIM_direct(rho_list)
        >>> print(f"Total QFIM shape: {F_total.shape}")  # (3, 3)
    
    Notes:
        - Used for multi-qubit quantum protocols
        - Each density matrix contributes independently to Fisher information
        - Optimized for RL environments with repeated calculations
    """
    F=np.zeros((3,3))# Initialize QFIM matrix, 3 parameters so 3 x 3 matrix
    


    for rho in rho_list:
        F+=calculate_QFIM_direct_singular(rho,h)

    return F

def optimized_nodes_to_qfim(node_lists: list[list[int]],
                        thetas: list[float],
                        h: float = 1e-5,
                        debug: bool = False):
    """
    Compute the QFIM for a protocol using closed-form density matrices.

    This version avoids QuTiP work inside the inner loop by deriving each qubit's
    density matrices directly from the path counts. Returns -1 for invalid move sets.

    Args:
        node_lists (list[list[int]]): Three node lists, one per qubit/path.
        thetas (list[float]): Three noise parameters ``[p1, p2, p3]``.
        h (float): Finite-difference step size used for the QFIM calculation.
        debug (bool): If True, prints a message when an invalid path is detected.

    Returns:
        numpy.ndarray | int: The QFIM matrix as a NumPy array, typically with shape
        ``(3, 3)`` or ``(n, n)`` depending on the downstream QFIM implementation, or
        -1 if the protocol is invalid.
    """
    # 1) Build path_lists via your existing fast validator
    path_lists = []
    for node_list in node_lists:
        path_list = nodes_to_paths(node_list, debug=debug)
        if path_list == -1:
            if debug:
                print("Invalid path detected as determined by nodes_to_paths")
            return -1
        path_lists.append(path_list)

    # 2) Precompute counts once per qubit
    counts_list = [_counts_123(path_list) for path_list in path_lists]  # [(n1,n2,n3), ...] x3

    # 3) Build the 7 rhos for each of the 3 qubits (fast, many cache hits)
    rho_1 = _rho7_from_counts_arrays(counts_list[0], thetas, h)  # arrays with correct nesting
    rho_2 = _rho7_from_counts_arrays(counts_list[1], thetas, h)
    rho_3 = _rho7_from_counts_arrays(counts_list[2], thetas, h)


    # 4) Compute QFIM
    F = calculate_QFIM_direct([rho_1, rho_2, rho_3], h)

    
    return F

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
    qcrb = calculate_QCRB_numerical(qfim) # Calculate the QCRB 

    protocol = obs[:30].reshape(3,10)  # Extract protocol (state) from observation

    return qcrb, protocol

def committee_predict(models: list, thetas: list[float]):
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
                path_list=nodes_to_paths(node_list)
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
            qfim = optimized_nodes_to_qfim(protocol, thetas)  # Convert protocol to QFIM
            qcrb = calculate_QCRB_numerical(qfim) # Calculate the QCRB 

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
def compute_QFIM_D(theta: list[float],N: np.ndarray) -> np.ndarray:
    """
    Compute the fisher informaiton matrix.

    Parameters
    ----------
    theta : list
        List of depolarizing noise parameter for each channel.
    N : np.ndarray
        The number of times each channel is traversed. Each row represents a trial, each column represents a channel.
    """

    theta = np.asarray(theta)
    m = len(theta) # Number of edges
    trials = N.shape[0] # Number of trials

    
    q = 1 - 4/3 * theta
    log_q = np.log(q)

    F = np.zeros((m, m))
    for trial in range(trials):
        # Compute product in logspace to minimize numerical instability
        logQ = np.dot(N[trial,:], log_q)
        Q2 = np.exp(2 * logQ)
        
        # Guard against division by zero when Q2 approaches 1
        # If (1 - Q2) is very small, the QFIM is singular/ill-conditioned
        denominator = 1 - Q2
        if abs(denominator) < 1e-10:
            # Skip this trial's contribution when Q2 ≈ 1
            continue

        for i in range(m):
            for j in range(i,m):
                F[i,j] +=  Q2/denominator  * N[trial,i]*N[trial,j] / (q[i] * q[j])
                if i!=j:
                    F[j,i]=F[i,j]

    return F

def nodes_to_qfim_D(node_lists: list[list[int]], theta: list[float],debug=False) -> np.ndarray:
    """
    Convert a protocol (nodes) and noise parameters (theta) into a QFIM for the depolarizing noise model.
    """

    path_lists = []
    for node_list in node_lists:
        path_list = nodes_to_paths(node_list, debug=debug)
        if path_list == -1:
            if debug:
                print("Invalid path detected as determined by nodes_to_paths")
            return -1
        path_lists.append(path_list)

    # 2) Precompute counts once per qubit
    counts_list = [_counts_123(path_list) for path_list in path_lists]  # [(n1,n2,n3), ...] x3

    # Get into numpy array format
    N = np.array(counts_list)  # shape (trials, channels)

    # 3) Compute QFIM using the counts and theta
    QFIM = compute_QFIM_D(theta, N)

    # Ensure valid QFIM

    if np.any(np.isnan(QFIM)) or np.any(np.isinf(QFIM)):
        if debug:
            print("QFIM contains NaN or Inf values, returning -1")
        return -1

    return QFIM

def get_qcrb_D(thetas:list[float], model):
    """ 
    Given a list of thetas, run the model in the QubitMoverEnv1_Estimator environment with depolarizing noise and
    abstract the QCRB and move (state) from the final observation.
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

    qfim = obs[-10:-1].reshape(3,3)  # Extract QFIM from observation    
    qcrb = calculate_QCRB_numerical(qfim) # Calculate the QCRB 

    protocol = obs[:30].reshape(3,10)  # Extract protocol (state) from observation

    return qcrb, protocol

def committee_predict_D(models: list, thetas: list[float]):
    """
    Given a list of trained models and input thetas, returns the best action that the models create.

        This function uses a hypercube stencil of perturbed thetas around the input thetas to evaluate 
        the models' predictions. It can also use a full 3x3x3 stencil.

    Parameters:
    -----------
    models: list
        List of trained models to evaluate.
    thetas: list[float]
        List of parameter values to evaluate the models on.

    Returns:
    --------
    min_qcrb: float
        The minimum QCRB found across all models and perturbed thetas.
    best_protocol: list[list[int]]
        The protocol (move) corresponding to the minimum QCRB found. This will be a list
        of node lists. A node list is a list of the desired nodes to traverse for each qubit, with 0 representing no-op positions.
    """

    min_qcrb = np.inf
    best_protocol = None

    perturbed_thetas_list = []
    perturbation_magnitude = 0.01  # Small perturbation

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
            temp_qcrb,protocol = get_qcrb_D(perturbed_thetas,model)

            # Ensure the protocol is valid
            for i,node_list in enumerate(protocol):
                path_list=nodes_to_paths(node_list)
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
            qfim = nodes_to_qfim_D(protocol, thetas)  # Convert protocol to QFIM
            qcrb = calculate_QCRB_numerical(qfim) # Calculate the QCRB 

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

# Bit Flip noise model estimation functions
# -----------------------------------------------
def make_equation(path_list):
    """
    Builds a python equation for the probability of measuring |0> as well 
     as one for |1>  given a list of paths.
    """
    I= np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    def f0(theta1, theta2, theta3):

        vals = [theta1, theta2, theta3]
        output = np.array([[1, 0], [0, 0]]) # Start in |0><0|
        # Iterate through each operation and multiply the corresponding matrices
        for i in path_list:
            output = I@output@I*(1-vals[i-1])+X@output@X*(vals[i-1])
        return float(output[0, 0])
    
    def f1(theta1, theta2, theta3):
        vals = [theta1, theta2, theta3]
        output = np.array([[1, 0], [0, 0]]) # Start in |0><0|
        # Iterate through each operation and multiply the corresponding matrices
        for i in path_list:
            output = I@output@I*(1-vals[i-1])+X@output@X*(vals[i-1])
        return float(output[1, 1])
    
    return f0, f1

# Numerically stable computation functions
def log1m2theta(theta: float) -> float:
    """Stable log(1 - 2*theta). Assumes 0 < theta < 0.5."""
    # When theta is small, 2*theta is small and log1p(-2*theta) is accurate.
    return math.log1p(-2.0 * theta)

def parity_counts(single_path_list: list[int]) -> tuple[int,int,int]:
    N1 = single_path_list.count(1)
    N2 = single_path_list.count(2)
    N3 = single_path_list.count(3)
    return N1, N2, N3

def even_parity_logp_and_grads(single_path_list: list[int],
                               thetas: list[float]) -> tuple[float, float, float, float, float]:
    """
    Returns:
        logp, p, dp/dθ1, dp/dθ2, dp/dθ3
    for the probability that (X1 + X2 + X3) is even, where Xj ~ Bin(Nj, θj).

    Uses log-space throughout for stability.
    """
    theta1, theta2, theta3 = thetas
    # Safety: keep strictly inside (0, 0.5) to avoid log singularities
    eps = 1e-15
    for t in (theta1, theta2, theta3):
        if not (eps < t < 0.5 - eps):
            raise ValueError("Theta out of safe range (0, 0.5).")

    N1, N2, N3 = parity_counts(single_path_list)

    # S = sum_j N_j * log(1 - 2*theta_j)
    L1 = N1 * log1m2theta(theta1)
    L2 = N2 * log1m2theta(theta2)
    L3 = N3 * log1m2theta(theta3)
    S  = L1 + L2 + L3

    # p = 0.5 * (1 + exp(S))
    # log p = log(0.5) + log(1 + exp(S)) = -log(2) + log1p(exp(S))
    logp = -math.log(2.0) + math.log1p(math.exp(S))
    p    = math.exp(logp)  # safe: logp is moderate (<= log(1))

    # Derivatives: dp/dθj = -Nj * exp(S) / (1 - 2*θj)
    # Compute exp(S) from log-space
    expS = math.exp(S)
    denom1 = (1.0 - 2.0 * theta1)
    denom2 = (1.0 - 2.0 * theta2)
    denom3 = (1.0 - 2.0 * theta3)

    dp_dtheta1 = - N1 * expS / denom1 if N1 > 0 else 0.0
    dp_dtheta2 = - N2 * expS / denom2 if N2 > 0 else 0.0
    dp_dtheta3 = - N3 * expS / denom3 if N3 > 0 else 0.0

    return logp, p, dp_dtheta1, dp_dtheta2, dp_dtheta3

# Gradient Function
def grad_p(single_path_list: list[int], 
            thetas: list[float]) -> list[float]:
    logp, p, dp1, dp2, dp3 = even_parity_logp_and_grads(single_path_list, thetas)
    return [dp1, dp2, dp3]

# Estimators
def estimator(all_protocols: list[list[list[int]]],
                all_measurement_proportions: list[list[float]],
                current_guess: list[float],
                num_tials_per: list[int],
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

            temp_paths = nodes_to_paths(qubit_moves)
            filtered_path = [k for k in temp_paths if k != 0]  # Remove identity gates
            k_i = n_i * proportions[j] # Number of successful trials
            p_i, _ = make_equation(filtered_path)
            

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


                grad = grad_p(filtered_path,[x1,x2,x3])
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


# Depolarizing noise model estimation functions
# -----------------------------------------------
def make_equation_D(path_list):
    """
    Builds a python equation for the probability of measuring |0> as well 
     as one for |1>  given a list of paths.
    """
    # From the paper we know that the probabilities are given by:
    # p(0) = 0.5 * (1+Q) Where Q = product_j (1 - 4/3 * theta_j)^(N_j)

    def f0(theta1, theta2, theta3):
        vals = [theta1, theta2, theta3]
        q = 1 - 4/3 * np.array(vals)
        Q = 1.0
        for i in path_list:
            Q *= q[i-1]
        return 0.5 * (1 + Q)
    
    def f1(theta1, theta2, theta3):
        vals = [theta1, theta2, theta3]
        q = 1 - 4/3 * np.array(vals)
        Q = 1.0
        for i in path_list:
            Q *= q[i-1]
        return 0.5 * (1 - Q)

    # Count the number of times each channel is traversed
    return f0, f1

# Gradient Function
def grad_p_D(single_path_list: list[int], 
            thetas: list[float]) -> list[float]:
    

    theta1, theta2, theta3 = thetas
    # Get the number of times each channel is traversed
    n1,n2,n3 = _counts_123(single_path_list)

    # Compute Q
    q = 1 - 4/3 * np.array(thetas)
    Q = 1.0
    for i in single_path_list:
        Q *= q[i-1]

    prefactor = -0.5 * Q


    dp1 = prefactor * (n1 / (1.0 - (4.0 / 3.0) * theta1))
    dp2 = prefactor * (n2 / (1.0 - (4.0 / 3.0) * theta2))
    dp3 = prefactor * (n3 / (1.0 - (4.0 / 3.0) * theta3))

    return [dp1, dp2, dp3]

def estimator_D(all_protocols: list[list[list[int]]],
                all_measurement_proportions: list[list[float]],
                current_guess: list[float],
                num_tials_per: list[int],
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


    guess = current_guess # renaming for clarity
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

            temp_paths = nodes_to_paths(qubit_moves)
            filtered_path = [k for k in temp_paths if k != 0]  # Remove identity gates
            k_i = n_i * proportions[j] # Number of successful trials
            p_i, _ = make_equation_D(filtered_path)

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


                grad = grad_p_D(filtered_path,[x1,x2,x3])
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





if __name__ == "__main__":
    # Example usage
    models_dir_B = "Models/bit_flip_committee_1"
    models_dir_D = "Models/depo_committee_1"

    models_D = get_committee_models(models_dir_D)
    models_B = get_committee_models(models_dir_B)

    print("\nTesting committee prediction for depolarizing noise model...")
    test_thetas = [0.05, 0.1, 0.15]
    min_qcrb_D, best_protocol_D = committee_predict_D(models_D, test_thetas)
    print("Depolarizing Noise Model:")
    print("Minimum QCRB:", min_qcrb_D)
    print("Best Protocol:", best_protocol_D)

    print("\nTesting committee prediction for bit flip noise model...")
    min_qcrb_B, best_protocol_B = committee_predict(models_B, test_thetas)
    print("Bit Flip Noise Model:")
    print("Minimum QCRB:", min_qcrb_B)
    print("Best Protocol:", best_protocol_B)
    print("\n")

    