# RLA Protocol Module

This module provides functions for working with Randomized Local Alignment (RLA) quantum protocols, including committee-based prediction and noise parameter estimation for both bit-flip and depolarizing noise models.

## Overview

### Node Lists

A **node list** is a sequence of integers representing nodes the specific qubit visits in a protocol. It serves as the input representation for the protocol actions.

- **Format**: `[n1, n2, ..., nk]` where each element is an integer
- **Valid Values**: Integers typically in the range [0, 1, 2, 3]
  - `0`: No-op (identity operation)
  - `1, 2, 3`: Different quantum channels or operations
- **Example**: `[1, 0, 2, 2, 3]` means apply channel 1, then no-op, then channel 2 twice, then channel 3

### Path Lists

Internally, node lists are converted to **path lists** using the `nodes_to_paths()` function:
- The first node is kept once
- Later nodes are duplicated
- Zero entries are removed
- The final trailing element is dropped

This transformation encodes the protocol in a canonical form for calculation.

---

## Committee Prediction Functions

Committee-based prediction uses multiple trained RL models to find the optimal protocol for given noise parameters.

### For Bit-Flip Noise: `committee_predict()`

```python
from RLA_protocol import get_committee_models, committee_predict

# Load trained models from a directory
models_dir = "Models/bit_flip_committee_1"
models = get_committee_models(models_dir)

# Predict the best protocol
thetas = [0.05, 0.1, 0.15]  # Noise parameters [p1, p2, p3]
min_qcrb, best_protocol = committee_predict(models, thetas)

print(f"Minimum QCRB: {min_qcrb}")
print(f"Best Protocol: {best_protocol}")
```

**Parameters:**
- `models`: List of trained `stable_baselines3.PPO` models
- `thetas`: List of three noise parameters [p1, p2, p3]

**Returns:**
- `min_qcrb`: Float, the minimum Quantum Cramér-Rao Bound achieved
- `best_protocol`: List of three node lists (one per qubit), guaranteed to have even length after removing zeros

**How It Works:**
1. Creates a 3×3×3 hypercube of perturbed thetas around the input values
2. Evaluates each model on each perturbation
3. Validates the returned protocols (ensures even-length paths)
4. Computes QCRB for each protocol
5. Returns the protocol with the lowest QCRB

### For Depolarizing Noise: `committee_predict_D()`

```python
from RLA_protocol import get_committee_models, committee_predict_D

# Load trained models from a directory
models_dir = "Models/depo_committee_1"
models = get_committee_models(models_dir)

# Predict the best protocol
thetas = [0.05, 0.1, 0.15]  # Noise parameters [theta1, theta2, theta3]
min_qcrb, best_protocol = committee_predict_D(models, thetas)

print(f"Minimum QCRB: {min_qcrb}")
print(f"Best Protocol: {best_protocol}")
```

**Parameters:**
- `models`: List of trained `stable_baselines3.PPO` models
- `thetas`: List of three noise parameters [theta1, theta2, theta3]

**Returns:**
- `min_qcrb`: Float, the minimum Quantum Cramér-Rao Bound achieved
- `best_protocol`: List of three node lists (one per qubit), guaranteed to have even length after removing zeros

**Differences from Bit-Flip:**
- Uses the depolarizing noise model: $q_j = 1 - \frac{4}{3}\theta_j$
- Measurement probabilities follow: $p(0) = 0.5(1 + Q)$ where $Q = \prod_j q_j^{N_j}$

---

## Estimator Functions

Estimator functions use measurement data to infer the noise parameters via maximum likelihood estimation (MLE).

### For Bit-Flip Noise: `estimator()`

```python
from RLA_protocol import estimator

# Protocol data
all_protocols = [
    [[1, 2, 3], [1, 1, 2], [3, 2, 1]],  # Protocol 1: 3 qubits
    [[2, 1, 3], [1, 2, 2], [3, 1, 2]],  # Protocol 2: 3 qubits
]

# Measurement proportions: fraction of |0> measurements
all_measurement_proportions = [
    [0.6, 0.55, 0.65],  # Protocol 1 results
    [0.58, 0.62, 0.60],  # Protocol 2 results
]

# Number of trials per protocol
num_trials_per = [1000, 1000]  # 1000 shots each

# Initial guess
guess = [0.1, 0.15, 0.12]

# Estimate noise parameters
estimated_params = estimator(
    all_protocols,
    all_measurement_proportions,
    guess,
    num_trials_per,
    tolerance=1e-6
)

print(f"Estimated parameters: {estimated_params}")  # [p1_est, p2_est, p3_est]
```

**Parameters:**
- `all_protocols`: List of protocols, where each protocol is a list of 3 node lists (one per qubit)
- `all_measurement_proportions`: List of measurement outcomes, where each entry is a list of 3 proportions (0 to 1) for each qubit
- `current_guess`: Initial estimate for [p1, p2, p3]
- `num_tials_per`: Number of trials (shots) for each protocol
- `tolerance`: Convergence tolerance (default: 1e-6)

**Returns:**
- `ndarray`: Optimized parameters [p1, p2, p3] constrained to [0.001, 0.4]

**How It Works:**
1. Builds the negative log-likelihood (NLL) function from the protocols and measurement data
2. Uses analytical gradients for efficient optimization
3. Applies L-BFGS-B optimization with parameter bounds
4. Returns the parameters that minimize the NLL

### For Depolarizing Noise: `estimator_D()`

```python
from RLA_protocol import estimator_D

# Protocol data
all_protocols = [
    [[1, 2, 3], [1, 1, 2], [3, 2, 1]],  # Protocol 1: 3 qubits
    [[2, 1, 3], [1, 2, 2], [3, 1, 2]],  # Protocol 2: 3 qubits
]

# Measurement proportions
all_measurement_proportions = [
    [0.6, 0.55, 0.65],  # Protocol 1 results
    [0.58, 0.62, 0.60],  # Protocol 2 results
]

# Number of trials (can be a single int applied to all protocols)
num_trials = 1000

# Initial guess
guess = [0.1, 0.15, 0.12]

# Estimate noise parameters
estimated_params = estimator_D(
    all_protocols,
    all_measurement_proportions,
    guess,
    num_trials,
    tolerance=1e-6
)

print(f"Estimated parameters: {estimated_params}")  # [theta1_est, theta2_est, theta3_est]
```

**Parameters:**
- `all_protocols`: List of protocols, where each protocol is a list of 3 node lists
- `all_measurement_proportions`: List of measurement outcomes (proportions of |0> measurements)
- `current_guess`: Initial estimate for [theta1, theta2, theta3]
- `num_tials_per`: Number of trials per protocol (int or list)
- `tolerance`: Convergence tolerance (default: 1e-6)

**Returns:**
- `ndarray`: Optimized parameters [theta1, theta2, theta3] constrained to [0.001, 0.4]

**Differences from Bit-Flip:**
- Uses depolarizing noise model
- Automatically broadcasts `num_tials_per` if provided as a scalar integer
- Measurement model: $p(0) = 0.5(1 + \prod_j (1 - \frac{4}{3}\theta_j)^{N_j})$

---

## Model Loading: `get_committee_models()`

Load pre-trained PPO models from a directory:

```python
from RLA_protocol import get_committee_models

# Load all .zip files from the directory
models = get_committee_models("Models/bit_flip_committee_1")

print(f"Loaded {len(models)} models")
```

**Parameters:**
- `models_dir`: Path to directory containing `.zip` model files

**Returns:**
- List of loaded `stable_baselines3.PPO` models sorted by filename

**Behavior:**
- Raises `FileNotFoundError` if directory doesn't exist
- Warns if no `.zip` files are found
- Warns if individual models fail to load (but continues loading others)

---

## Complete Example Workflow

```python
from RLA_protocol import (
    get_committee_models,
    committee_predict_D,
    estimator_D
)

# Step 1: Load trained models
models = get_committee_models("Models/depo_committee_1")

# Step 2: Use committee to predict best protocol for given thetas
thetas_true = [0.08, 0.12, 0.09]
min_qcrb, predicted_protocol = committee_predict_D(models, thetas_true)
print(f"Best protocol found with QCRB: {min_qcrb}")

# Step 3: Simulate measurement data with this protocol
# (In practice, this would come from actual quantum measurements)
# all_protocols, all_measurement_proportions = run_quantum_experiment(...)

# Step 4: Estimate noise parameters from measurements
guess = [0.1, 0.1, 0.1]
estimated_thetas = estimator_D(
    all_protocols,
    all_measurement_proportions,
    guess,
    num_trials_per=1000
)

print(f"True parameters:      {thetas_true}")
print(f"Estimated parameters: {estimated_thetas}")
print(f"Error: {abs(estimated_thetas - np.array(thetas_true))}")
```

---

## Parameter Ranges and Constraints

- **Bit-flip parameters [p1, p2, p3]**: Probability of bit-flip on each channel
  - Valid range: (0, 0.5), typically constrained to [0.001, 0.4]
  
- **Depolarizing parameters [theta1, theta2, theta3]**: Depolarizing rate on each channel
  - Valid range: (0, 0.5), typically constrained to [0.001, 0.4]

Both models constrain parameters to [0.001, 0.4] to maintain numerical stability.

---

## Notes

- All functions expect **three qubits** (or three noise channels)
- Protocols are returned without leading/trailing zeros and with guaranteed even length after zero removal
- The QFIM (Quantum Fisher Information Matrix) is computed with finite-difference derivatives using step size h=1e-5
- Gradient-based optimization uses L-BFGS-B for robustness with bounds constraints
