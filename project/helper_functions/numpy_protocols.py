import numpy as np
from itertools import product
from scipy.optimize import minimize
from scipy.special import expit, logsumexp  # sigmoid and stable log-sum-exp

import helper_functions.utils as utils
import helper_functions.numpy_simulation as sim
import helper_functions.RLA_functions as rla
#############################
# Multicast Protocol
#############################
def multicast_qcrb(theta_0, theta_1, theta_2):
    # Denominator
    denom = ((1 - 2 * theta_0)**2 * (1 - 2 * theta_1)**2 * (1 - 2 * theta_2)**2)

    # Numerator components
    term1 = -(1 - 2 * theta_2)**2 * (-1 + theta_2) * theta_2
    term2 = 8 * theta_1**3 * (1 - 3 * theta_2 + 3 * theta_2**2)
    term3 = -4 * theta_1**4 * (1 - 3 * theta_2 + 3 * theta_2**2)
    term4 = theta_1**2 * (-5 + 17 * theta_2 - 29 * theta_2**2 + 24 * theta_2**3 - 12 * theta_2**4)
    term5 = theta_1 * (1 - 5 * theta_2 + 17 * theta_2**2 - 24 * theta_2**3 + 12 * theta_2**4)
    term6 = 8 * theta_0**3 * (
        (1 - 3 * theta_2 + 3 * theta_2**2) +
        theta_1 * (-3 + 4 * theta_2 - 4 * theta_2**2) +
        theta_1**2 * (3 - 4 * theta_2 + 4 * theta_2**2)
    )
    term7 = -4 * theta_0**4 * (
        (1 - 3 * theta_2 + 3 * theta_2**2) +
        theta_1 * (-3 + 4 * theta_2 - 4 * theta_2**2) +
        theta_1**2 * (3 - 4 * theta_2 + 4 * theta_2**2)
    )
    term8 = theta_0**2 * (
        -5 + 17 * theta_2 - 29 * theta_2**2 + 24 * theta_2**3 - 12 * theta_2**4 +
        8 * theta_1**3 * (3 - 4 * theta_2 + 4 * theta_2**2) -
        4 * theta_1**4 * (3 - 4 * theta_2 + 4 * theta_2**2) +
        theta_1**2 * (-29 + 32 * theta_2 - 48 * theta_2**2 + 32 * theta_2**3 - 16 * theta_2**4) +
        theta_1 * (17 - 16 * theta_2 + 32 * theta_2**2 - 32 * theta_2**3 + 16 * theta_2**4)
    )
    term9 = theta_0 * (
        1 - 5 * theta_2 + 17 * theta_2**2 - 24 * theta_2**3 + 12 * theta_2**4 -
        8 * theta_1**3 * (3 - 4 * theta_2 + 4 * theta_2**2) +
        4 * theta_1**4 * (3 - 4 * theta_2 + 4 * theta_2**2) -
        theta_1 * (5 + 16 * theta_2**2 - 32 * theta_2**3 + 16 * theta_2**4) +
        theta_1**2 * (17 - 16 * theta_2 + 32 * theta_2**2 - 32 * theta_2**3 + 16 * theta_2**4)
    )

    # Final result
    result = (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9) / denom
    return result*(2/3) # The 2/3 factor is because we are calculating the average qcrb for 3 qubits

def simulate_multicast_protocol(theta_0, theta_1, theta_2, trials=1000):
    """
    Simulate the multicast protocol by generating random measurement outcomes based on the probabilities defined by the theta parameters.
    """
        
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    for _ in range(trials):
        x0 = np.random.rand() < theta_0
        x1 = np.random.rand() < theta_1
        x2 = np.random.rand() < theta_2
        y1 = int(x0 ^ x1)  # XOR
        y2 = int(x0 ^ x2)
        # Save to Dictionary
        outcome = f"{y1}{y2}"
        counts[outcome] += 1
    return counts

def estimate_multicast(trials,theta_0, theta_1, theta_2,guess= [0.2,0.2,0.2]):
    """
    Estimate the theta parameters from the simulated multicast protocol outcomes.
    """

    counts = simulate_multicast_protocol(theta_0, theta_1, theta_2, trials)

    # Calculate probabilities
    p00 = counts["00"]
    p01 = counts["01"]
    p10 = counts["10"] 
    p11 = counts["11"] 


    data = [p00, p01, p10, p11]
    # Estimate theta parameters
    # Estimate the theta paramaters using the ML estimator for the multicast protocol, which is derived from the probabilities of the outcomes. 

    def probs(t0, t1, t2):
        t0h, t1h, t2h = 1-t0, 1-t1, 1-t2

        p00 = t0*t1*t2 + t0h*t1h*t2h
        p01 = t0*t1*t2h + t0h*t1h*t2
        p10 = t0*t1h*t2 + t0h*t1*t2h
        p11 = t0*t1h*t2h + t0h*t1*t2

        return np.array([p00, p01, p10, p11])

    def neg_log_likelihood(theta, counts):
        t0, t1, t2 = theta
        p = probs(t0, t1, t2)

        # Soft clipping to avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)

        return -np.sum(counts * np.log(p))

    def estimate_thetas(data):
        # Initial guess (uniform-ish)
        x0 = guess

        bounds = [(0.001, 0.45)]*3

        result = minimize(neg_log_likelihood,
                        x0,
                        args=(np.array(data)),
                        bounds=bounds)

        return result.x, result.success

    
    estimation, ok = estimate_thetas(data)

    return estimation

# Depolarizing noise versions

def multicast_D_qcrb(theta_0, theta_1, theta_2, eps=1e-12, infinity=1e12):
    q_1 = 1-4/3*theta_0
    q_2 = 1-4/3*theta_1
    q_3 = 1-4/3*theta_2

    # Taken Straight from Mathematica, see notebook for details. Note that this is the average qcrb for the 3 qubits, which is what we are interested in.

    QCRB = (-9*(q_1**4*(-1 + 2*q_3**2 + q_2**2*(2 + q_3**2)) + (q_2**2 + \
            q_3**2)*(-q_3**2 + q_2**2*(-1 + 2*q_3**2)) + q_1**2*(2*q_3**2*(-1 + \
            q_3**2) + q_2**4*(2 + q_3**2) + q_2**2*(-2 - 6*q_3**2 + \
            q_3**4))))/(64*q_1**2*q_2**2*q_3**2)

    return QCRB # The 2/3 factor is because we are calculating the average qcrb for 3 qubits

def simulate_multicast_protocol_D(theta_1, theta_2, theta_3, trials=1000):
    """
    Simulate the multicast protocol by generating random measurement outcomes based on the probabilities defined by the theta parameters.
    """
        
    p00 = (
        (-3 + 2*theta_2)*(-3 + 2*theta_3)
        + theta_1*(-6 + 4*theta_2 + 4*theta_3)
    ) / 9.0

    p01 = (
        4*theta_1*theta_2
        + 6*theta_3
        - 4*(theta_1 + theta_2)*theta_3
    ) / 9.0

    p10 = (
        theta_2*(6 - 4*theta_1 - 4*theta_3)
        + 4*theta_1*theta_3
    ) / 9.0

    p11 = (
        theta_1*(6 - 4*theta_2 - 4*theta_3)
        + 4*theta_2*theta_3
    ) / 9.0

    # Simulate outcomes based on probabilities
    outcomes = {'00': 0, '01': 0, '10': 0, '11': 0}
    for _ in range(trials):
        outcome = np.random.choice(['00', '01', '10', '11'], p=[p00, p01, p10, p11])
        outcomes[outcome] += 1

    return outcomes

def estimate_multicast_D(trials,theta_0, theta_1, theta_2,guess= [0.2,0.2,0.2]):
    """
    Estimate the theta parameters from the simulated multicast protocol outcomes.
    """

    counts = simulate_multicast_protocol_D(theta_0, theta_1, theta_2, trials)

    # Calculate probabilities
    p00 = counts["00"]
    p01 = counts["01"]
    p10 = counts["10"] 
    p11 = counts["11"] 


    data = [p00, p01, p10, p11]
    # Estimate theta parameters
    # Estimate the theta paramaters using the ML estimator for the multicast protocol, which is derived from the probabilities of the outcomes. 

    
    def probs(t0, t1, t2):
        p00 = (
            (-3 + 2*t1)*(-3 + 2*t2)
            + t0*(-6 + 4*t1 + 4*t2)
        ) / 9.0

        p01 = (
            4*t0*t1
            + 6*t2
            - 4*(t0 + t1)*t2
        ) / 9.0

        p10 = (
            t1*(6 - 4*t0 - 4*t2)
            + 4*t0*t2
        ) / 9.0

        p11 = (
            t0*(6 - 4*t1 - 4*t2)
            + 4*t1*t2
        ) / 9.0

        return np.array([p00, p01, p10, p11])


    def neg_log_likelihood(theta, counts):
        t0, t1, t2 = theta
        p = probs(t0, t1, t2)

        # Soft clipping to avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)

        return -np.sum(counts * np.log(p))

    def estimate_thetas(data):
        # Initial guess (uniform-ish)
        x0 = guess

        bounds = [(0.001, 0.45)]*3

        result = minimize(neg_log_likelihood,
                        x0,
                        args=(np.array(data)),
                        bounds=bounds)

        return result.x, result.success

    
    estimation, ok = estimate_thetas(data)

    return estimation


#############################
# Back and Forth Z Protocol
#############################

def back_and_forth_z(theta_0, theta_1, theta_2,infinity=1e12):
    theta_0_hat = 1 - theta_0
    theta_1_hat = 1 - theta_1
    theta_2_hat = 1 - theta_2

    # Eigenvalues
    e_val000 = 0.5 * (theta_0*theta_1*theta_2       + theta_0_hat*theta_1_hat*theta_2_hat)
    e_val001 = 0.5 * (theta_0*theta_1*theta_2_hat   + theta_0_hat*theta_1_hat*theta_2)
    e_val010 = 0.5 * (theta_0*theta_1_hat*theta_2   + theta_0_hat*theta_1*theta_2_hat)
    e_val011 = 0.5 * (theta_0*theta_1_hat*theta_2_hat + theta_0_hat*theta_1*theta_2)
    e_val100 = 0.5 * (theta_0_hat*theta_1*theta_2   + theta_0*theta_1_hat*theta_2_hat)
    e_val101 = 0.5 * (theta_0_hat*theta_1*theta_2_hat + theta_0*theta_1_hat*theta_2)
    e_val110 = 0.5 * (theta_0_hat*theta_1_hat*theta_2 + theta_0*theta_1*theta_2_hat)
    e_val111 = 0.5 * (theta_0_hat*theta_1_hat*theta_2_hat + theta_0*theta_1*theta_2)

    e_val_list = [e_val000, e_val001, e_val010, e_val011, e_val100, e_val101, e_val110, e_val111]

    # d/d theta_0
    d0 = [
        0.5 * (theta_1*theta_2 - theta_1_hat*theta_2_hat),   # 000
        0.5 * (theta_1*theta_2_hat - theta_1_hat*theta_2),   # 001
        0.5 * (theta_1_hat*theta_2 - theta_1*theta_2_hat),   # 010
        0.5 * (theta_1_hat*theta_2_hat - theta_1*theta_2),   # 011
        0.5 * (theta_1_hat*theta_2_hat - theta_1*theta_2),   # 100
        0.5 * (theta_1_hat*theta_2 - theta_1*theta_2_hat),   # 101
        0.5 * (theta_1*theta_2_hat - theta_1_hat*theta_2),   # 110
        0.5 * (theta_1*theta_2 - theta_1_hat*theta_2_hat),   # 111
    ]

    # d/d theta_1
    d1 = [
        0.5 * (theta_0*theta_2 - theta_0_hat*theta_2_hat),   # 000
        0.5 * (theta_0*theta_2_hat - theta_0_hat*theta_2),   # 001
        0.5 * (theta_0_hat*theta_2_hat - theta_0*theta_2),   # 010
        0.5 * (theta_0_hat*theta_2 - theta_0*theta_2_hat),   # 011
        0.5 * (theta_0_hat*theta_2 - theta_0*theta_2_hat),   # 100
        0.5 * (theta_0_hat*theta_2_hat - theta_0*theta_2),   # 101
        0.5 * (theta_0*theta_2_hat - theta_0_hat*theta_2),   # 110
        0.5 * (theta_0*theta_2 - theta_0_hat*theta_2_hat),   # 111
    ]

    # d/d theta_2
    d2 = [
        0.5 * (theta_0*theta_1 - theta_0_hat*theta_1_hat),   # 000
        0.5 * (theta_0_hat*theta_1_hat - theta_0*theta_1),   # 001
        0.5 * (theta_0*theta_1_hat - theta_0_hat*theta_1),   # 010
        0.5 * (theta_0_hat*theta_1 - theta_0*theta_1_hat),   # 011
        0.5 * (theta_0_hat*theta_1 - theta_0*theta_1_hat),   # 100
        0.5 * (theta_0*theta_1_hat - theta_0_hat*theta_1),   # 101
        0.5 * (theta_0_hat*theta_1_hat - theta_0*theta_1),   # 110
        0.5 * (theta_0*theta_1 - theta_0_hat*theta_1_hat),   # 111
    ]
    d= [d0, d1, d2]

    # Compute the QFIM
    qfim = np.zeros((3, 3))
    # Compute upper triangle of the QFIM
    # Then flip to get the lower triangle (since QFIM is symmetric)
    for i in range(3):
        for j in range(i, 3):
            qfim[i, j] = sum([d[i][k] * d[j][k] / e_val_list[k] for k in range(8)])
            if i != j:
                qfim[j, i] = qfim[i, j]

    # Compute qcrb using qm1 function
    qcrb = utils.calculate_QCRB_numerical(qfim, infinity=infinity)
    return qcrb

def simulate_back_and_forth_z(theta_0, theta_1, theta_2, trials=1000,guess=[0.2,0.2,0.2]):
    theta_0_hat = 1 - theta_0
    theta_1_hat = 1 - theta_1
    theta_2_hat = 1 - theta_2

    # Eigenvalues
    p_val000 = 0.5 * (theta_0*theta_1*theta_2       + theta_0_hat*theta_1_hat*theta_2_hat)
    p_val001 = 0.5 * (theta_0*theta_1*theta_2_hat   + theta_0_hat*theta_1_hat*theta_2)
    p_val010 = 0.5 * (theta_0*theta_1_hat*theta_2   + theta_0_hat*theta_1*theta_2_hat)
    p_val011 = 0.5 * (theta_0*theta_1_hat*theta_2_hat + theta_0_hat*theta_1*theta_2)
    p_val100 = 0.5 * (theta_0_hat*theta_1*theta_2   + theta_0*theta_1_hat*theta_2_hat)
    p_val101 = 0.5 * (theta_0_hat*theta_1*theta_2_hat + theta_0*theta_1_hat*theta_2)
    p_val110 = 0.5 * (theta_0_hat*theta_1_hat*theta_2 + theta_0*theta_1*theta_2_hat)
    p_val111 = 0.5 * (theta_0_hat*theta_1_hat*theta_2_hat + theta_0*theta_1*theta_2)

    p_val_list = [p_val000, p_val001, p_val010, p_val011, p_val100, p_val101, p_val110, p_val111] 

    # Simulate the network by sampling from the p_vals
    counts = {"000": 0, "001": 0, "010": 0, "011": 0, "100": 0, "101": 0, "110": 0, "111": 0}
    for _ in range(trials):
        outcome = np.random.choice(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            p=p_val_list
        )
        counts[outcome] += 1
    return counts

def estimate_back_and_forth_z(trials, theta_0, theta_1, theta_2, guess=[0.2, 0.2, 0.2]):
    counts = simulate_back_and_forth_z(trials=trials, theta_0=theta_0, theta_1=theta_1, theta_2=theta_2)

    # ----- Helpers -----
    def all_bit_patterns(n_bits=3):
        """(8,3) array of 0/1 bit patterns for n_bits=3."""
        return np.array(list(product([0, 1], repeat=n_bits)), dtype=int)

    def logA_logB(theta, Y):
        """
        logA[y] = sum_i [ y_i*log(theta_i)     + (1-y_i)*log(1-theta_i) ]
        logB[y] = sum_i [ (1-y_i)*log(theta_i) + y_i*log(1-theta_i) ] = logA[~y]
        """
        logt  = np.log(theta)
        log1t = np.log(1.0 - theta)
        logA = Y @ logt + (1 - Y) @ log1t
        logB = (1 - Y) @ logt + Y @ log1t
        return logA, logB


    # ----- Objective (neg log-likelihood) and gradient in logit space -----
    def neg_ll_and_grad_phi(phi, counts, Y):
        """
        counts: shape (8,), Y: shape (8,3), phi: logits (3,)
        theta = sigmoid(phi) in (0,1)
        p(y) = 0.5 * (A_y + B_y)
        """
        theta = expit(phi)  # (3,)

        # logA/logB for all 8 patterns
        logA, logB = logA_logB(theta, Y)            # (8,), (8,)
        s = logsumexp(np.vstack([logA, logB]), axis=0)  # log(A+B), shape (8,)
        logp = np.log(0.5) + s

        # Negative log-likelihood
        nll = -np.sum(counts * logp)

        # Mixture weights (responsibilities) for A vs B
        wA = np.exp(logA - s)       # (8,)
        wB = 1.0 - wA               # (8,)

        # Gradient wrt theta
        inv_t  = 1.0 / theta
        inv_1t = 1.0 / (1.0 - theta)
        grad_theta = np.zeros(3)

        for k in range(3):
            yk = Y[:, k].astype(float)
            # d/dθ logA =  yk/θ - (1-yk)/(1-θ)
            dlogA = yk * inv_t[k] - (1 - yk) * inv_1t[k]
            # d/dθ logB = (1-yk)/θ -  yk/(1-θ)
            dlogB = (1 - yk) * inv_t[k] - yk * inv_1t[k]
            dlogp = wA * dlogA + wB * dlogB
            grad_theta[k] = -np.sum(counts * dlogp)

        # Chain rule to logits: dθ/dφ = θ(1-θ)
        grad_phi = grad_theta * theta * (1.0 - theta)
        return nll, grad_phi


    def fit_theta_mle(counts_dict, seed=None):
        """
        Minimal MLE for:
        p(y) = 0.5 * [ Π_i θ_i^{y_i} (1-θ_i)^{1-y_i} + Π_i θ_i^{1-y_i} (1-θ_i)^{y_i} ]
        counts_dict keys: "000","001",...,"111"
        Returns: thetahat (3,), success (bool)
        """
        # Fix pattern order "000".."111"
        patterns = ["{0:03b}".format(i) for i in range(8)]
        counts = np.array([counts_dict[p] for p in patterns], dtype=float)
        if counts.sum() <= 0:
            raise ValueError("Total count must be > 0")

        Y = all_bit_patterns(3)  # (8,3)

        # Single simple start in logit space (phi=0 => theta=0.5)
        phi0 = guess

        res = minimize(
            fun=lambda ph: neg_ll_and_grad_phi(ph, counts, Y),
            x0=phi0,
            method="L-BFGS-B",
            jac=True,  # we provide gradient
            options=dict(maxiter=5000, ftol=1e-12, gtol=1e-10)
        )

        thetahat = expit(res.x)

        # Optional: break θ ↔ (1-θ) symmetry (pick canonical representative)
        if thetahat.sum() < 1.5:
            thetahat = 1.0 - thetahat

        return thetahat, bool(res.success)
    
    estimate,_ = fit_theta_mle(counts, seed=42)
    # Format estimate
    for i,e in enumerate(estimate):
        if e > 0.5:
            estimate[i] = 1-e


    return estimate


# Depolarizing noise versions

def back_and_forth_z_D(theta_0, theta_1, theta_2, infinity=1e12):
    q_1 = 1-4/3*theta_0
    q_2 = 1-4/3*theta_1
    q_3 = 1-4/3*theta_2

    # Straight from Mathematica, see notebook for details. Note that this is the average qcrb for the 3 qubits, which is what we are interested in.
    QCRB = (-9*(q_1**4*(-1 + 2*q_3**2 + q_2**2*(2 + q_3**2)) + (q_2**2 + \
            q_3**2)*(-q_3**2 + q_2**2*(-1 + 2*q_3**2)) + q_1**2*(2*q_3**2*(-1 + \
            q_3**2) + q_2**4*(2 + q_3**2) + q_2**2*(-2 - 6*q_3**2 + \
            q_3**4))))/(64*q_1**2*q_2**2*q_3**2)

    return QCRB

def simulate_back_and_forth_z_D(theta_0, theta_1, theta_2, trials=1000,guess=[0.2,0.2,0.2]):
    
    q_1 = 1-4/3*theta_0
    q_2 = 1-4/3*theta_1
    q_3 = 1-4/3*theta_2

    p000 = (1 + q_2*q_3 + q_1*(q_2 + q_3)) / 8.0
    p001 = (1 + q_1*q_2 - (q_1 + q_2)*q_3) / 8.0
    p010 = (1 - q_2*q_3 + q_1*(-q_2 + q_3)) / 8.0
    p011 = (1 + q_2*q_3 - q_1*(q_2 + q_3)) / 8.0

    p100 = p011
    p101 = p010
    p110 = p001
    p111 = p000


    p_val_list = [p000, p001, p010, p011, p100, p101, p110, p111] 

    # Simulate the network by sampling from the p_vals
    counts = {"000": 0, "001": 0, "010": 0, "011": 0, "100": 0, "101": 0, "110": 0, "111": 0}
    for _ in range(trials):
        outcome = np.random.choice(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            p=p_val_list
        )
        counts[outcome] += 1
    return counts

def estimate_back_and_forth_z_D(trials, theta_0, theta_1, theta_2, guess=[0.2, 0.2, 0.2]):
    """
    Estimate the theta parameters from the simulated back-and-forth z protocol outcomes.
    """

    counts = simulate_back_and_forth_z_D(theta_0, theta_1, theta_2, trials)

    # Calculate probabilities
    p000 = counts["000"]
    p001 = counts["001"]
    p010 = counts["010"] 
    p011 = counts["011"] 
    p100 = counts["100"]
    p101 = counts["101"]
    p110 = counts["110"]
    p111 = counts["111"]



    data = [p000, p001, p010, p011, p100, p101, p110, p111]
    # Estimate theta parameters
    # Estimate the theta paramaters using the ML estimator for the back-and-forth z protocol, which is derived from the probabilities of the outcomes. 

    
    def probs(t0, t1, t2):
        q1 = 1-4/3*t0
        q2 = 1-4/3*t1
        q3 = 1-4/3*t2

        # Probabilities as found in mathematica notebook.
        p000 = (1 + q2*q3 + q1*(q2 + q3)) / 8.0
        p001 = (1 + q1*q2 - (q1 + q2)*q3) / 8.0
        p010 = (1 - q2*q3 + q1*(-q2 + q3)) / 8.0
        p011 = (1 + q2*q3 - q1*(q2 + q3)) / 8.0

        p100 = p011
        p101 = p010
        p110 = p001
        p111 = p000

        return np.array([p000, p001, p010, p011, p100, p101, p110, p111])



    def neg_log_likelihood(theta, counts):
        t0, t1, t2 = theta
        p = probs(t0, t1, t2)

        # Soft clipping to avoid log(0)
        eps = 1e-12
        p = np.clip(p, eps, 1-eps)

        return -np.sum(counts * np.log(p))

    def estimate_thetas(data):
        # Initial guess (uniform-ish)
        x0 = guess

        bounds = [(0.001, 0.45)]*3

        result = minimize(neg_log_likelihood,
                        x0,
                        args=(np.array(data)),
                        bounds=bounds)

        return result.x, result.success

    
    estimation, ok = estimate_thetas(data)

    return estimation
    
#############################
# Repeated unicast protocol
#############################

def repeated_unicast(theta_0, theta_1, theta_2, n=2, eps=1e-12, infinity=1e12):
    """
    Calculate the QFIM (diagonal here) and QCRB for repeated unicast.
    n: number of visits; number_gates = 2*n must be even.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]

    qfim = utils.optimized_nodes_to_qfim(protocol, [theta_0, theta_1, theta_2])

    qcrb = utils.calculate_QCRB_numerical(qfim, infinity=infinity)
    return qcrb

def simulate_repeated_unicast(theta_0, theta_1, theta_2, n=2, trials=1000):
    """
    Simulate the repeated unicast protocol by generating random measurement outcomes based on the probabilities defined by the theta parameters.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]
    
    proportions = sim.simulate_bit_flip(thetas=[theta_0, theta_1, theta_2], protocol=protocol, iterations=trials)
    return proportions
    
def estimate_repeated_unicast(theta_0, theta_1, theta_2, n=2, trials=1000,guess=[0.2,0.2,0.2]):
    """
    Estimate the theta parameters from the simulated repeated unicast protocol outcomes.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]
    proportions = simulate_repeated_unicast(theta_0, theta_1, theta_2, n=n, trials=trials)
    estimate = rla.estimator_BF(all_protocols = [protocol],
                                      all_measurement_proportions=[proportions],
                                      trials_per_iteration=[trials],
                                      current_guess=guess)
    
    return estimate    

def repeated_unicast_D(theta_0, theta_1, theta_2, n=2, eps=1e-12, infinity=1e12):
    """
    Calculate the D-matrix for repeated unicast.
    n: number of visits; number_gates = 2*n must be even.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    theta = [theta_0, theta_1, theta_2]
    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]

    qfim = utils.nodes_to_qfim_D(protocol,theta)
    QCRB = utils.calculate_QCRB_numerical(qfim, infinity=infinity)

    return QCRB

def simulate_repeated_unicast_D(theta_0, theta_1, theta_2, n=2, trials=1000):
    """
    Simulate the repeated unicast protocol with depolarizing noise by generating random measurement outcomes based on the probabilities defined by the theta parameters.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]
    
    proportions = sim.simulate_depolarizing_noise(thetas=[theta_0, theta_1, theta_2], protocol=protocol, trials=trials)
    return proportions

def estimate_repeated_unicast_D(theta_0, theta_1, theta_2, n=2, trials=1000,guess=[0.2,0.2,0.2]):
    """
    Estimate the theta parameters from the simulated repeated unicast protocol with depolarizing noise outcomes.
    """
    # Simulate network for n visits we must first translate n into a protocol.
    # for n = 1 the protocol is [1,1] so it should be n+1 entries long.

    protocol1 = [1]*(n+1)
    protocol2 = [2]*(n+1)
    protocol3 = [3]*(n+1)

    protocol = [protocol1, protocol2, protocol3]
    proportions = simulate_repeated_unicast_D(theta_0, theta_1, theta_2, n=n, trials=trials)
    estimate = rla.estimator_D(all_protocols = [protocol],
                                      all_measurement_proportions=[proportions],
                                      trials_per_iteration=[trials],
                                      current_guess=guess)
    
    return estimate

