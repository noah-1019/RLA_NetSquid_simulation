import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Custome module import 
import RLA_protocol as qm1


# =========================================================================
# Bit Flip Estimator
# =========================================================================

class QubitMoverEnv1_Estimator(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 h: float = 1e-5,
                 compute_qfim_each_step: bool = True,   # keep same behavior by default
                 compute_reward: bool = False            # keep same behavior by default
                 ):
        
        super().__init__()
        self.n_qubits = 3
        self.max_steps = 10

        # Store config
        self.h = h
        self.compute_qfim_each_step = compute_qfim_each_step

        # States & parameters
        self.thetas = np.random.uniform(0.05, 0.5, 3).astype(np.float32)

        # Use a compact dtype for states (values in 0..3)
        self.state = np.zeros((self.n_qubits, self.max_steps), dtype=np.uint8)
        self.current_step = 0
        self.done = False

        # QFIM as a persistent buffer to allow in-place updates
        self.qfim = np.zeros((3, 3), dtype=np.float32)

        # --- Observation space definition ---
        # state (3*10), thetas(3), qfim(9), move(1)
        self._obs_state_len = self.n_qubits * self.max_steps
        self._obs_thetas_len = 3
        self._obs_qfim_len = 9
        self._obs_move_len = 1
        self._obs_len = self._obs_state_len + self._obs_thetas_len + self._obs_qfim_len + self._obs_move_len

        low_states = np.zeros(self._obs_state_len, dtype=np.float32)
        high_states = np.full(self._obs_state_len, 3, dtype=np.float32)
        low_thetas = np.zeros(3, dtype=np.float32)
        high_thetas = np.ones(3, dtype=np.float32)
        low_qfim = np.zeros(9, dtype=np.float32)
        high_qfim = np.full(9, 1e6, dtype=np.float32)
        low_move = np.array([0.0], dtype=np.float32)
        high_move = np.array([float(self.max_steps)], dtype=np.float32)

        low = np.concatenate([low_states, low_thetas, low_qfim, low_move]).astype(np.float32)
        high = np.concatenate([high_states, high_thetas, high_qfim, high_move]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: 3 qubits, each in {0,1,2,3}
        self.action_space = spaces.MultiDiscrete([4] * self.n_qubits)

        # Pre-allocate an observation buffer (float32 as required)
        self._obs = np.zeros(self._obs_len, dtype=np.float32)
        self._idx_state = slice(0, self._obs_state_len)
        self._idx_thetas = slice(self._obs_state_len, self._obs_state_len + 3)
        self._idx_qfim = slice(self._obs_state_len + 3, self._obs_state_len + 3 + 9)
        self._idx_move = slice(self._obs_state_len + 3 + 9, self._obs_len)

        self.reset()

    def _write_obs(self):
        # Cast state to float32 on assignment (30 elements -> very cheap)
        self._obs[self._idx_state] = self.state.reshape(-1)
        self._obs[self._idx_thetas] = self.thetas
        # Use ravel() (view) since we keep self.qfim buffer persistent and update in-place
        self._obs[self._idx_qfim] = self.qfim.ravel()
        self._obs[self._idx_move] = float(self.current_step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.thetas = np.random.uniform(0.05, 0.5, 3).astype(np.float32)
        self.state.fill(0)
        self.current_step = 0
        self.done = False
        self.qfim.fill(0.0)
        self._write_obs()
        return self._obs, {}

    @staticmethod
    def _all_measured_col(col: np.ndarray) -> bool:
        # col: shape (n_qubits,), uint8
        # True if all zeros
        return np.all(col == 0)

    #@profile
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")

        # Vectorized write: set the column for this step
        # action expected shape (3,)
        self.state[:, self.current_step] = action

        # Advance step
        self.current_step += 1

        # Check termination: all measured at this step column or max steps reached
        all_measured = self._all_measured_col(self.state[:, self.current_step - 1])
        
        terminated = bool(all_measured)  # natural end (task condition)
        truncated = bool(self.current_step >= self.max_steps and not terminated)


        # Default reward (you can keep a small move penalty)
        reward_val = 0.0

        # Update QFIM only if configured to do so
        if self.compute_qfim_each_step:
            # Prefer passing ndarray directly; if qm1 expects lists, do .tolist() once
            # node_lists = self.state.tolist()   # slower; avoid if qm1 can handle numpy
            node_lists = self.state  # ndarray (3,10)

            F = qm1.optimized_nodes_to_qfim(node_lists, self.thetas, h=self.h, debug=False)
            if isinstance(F, int) and F == -1:
                # Invalid move/path
                self.qfim.fill(0.0)
                # For inference, we typically do not penalize; keep as 0 or a small penalty if needed
                reward_val = 0.0
            else:
                np.copyto(self.qfim, np.asarray(F, dtype=np.float32))


        # Update observation buffer
        self._write_obs()
        # For SB3 safety, returning a copy is unnecessary in most cases; keep as is for speed
        return self._obs, float(reward_val), terminated, truncated, {}

    def render(self): pass
    def close(self): pass

# =========================================================================
# Depolarizing Estimator
# =========================================================================

class QubitMoverEnv1_D_Estimator(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self,
                 h: float = 1e-5,
                 compute_qfim_each_step: bool = True,   # keep same behavior by default
                 compute_reward: bool = False            # keep same behavior by default
                 ):
        
        super().__init__()
        self.n_qubits = 3
        self.max_steps = 10

        # Store config
        self.h = h
        self.compute_qfim_each_step = compute_qfim_each_step

        # States & parameters
        self.thetas = np.random.uniform(0.05, 0.5, 3).astype(np.float32)

        # Use a compact dtype for states (values in 0..3)
        self.state = np.zeros((self.n_qubits, self.max_steps), dtype=np.uint8)
        self.current_step = 0
        self.done = False

        # QFIM as a persistent buffer to allow in-place updates
        self.qfim = np.zeros((3, 3), dtype=np.float32)

        # --- Observation space definition ---
        # state (3*10), thetas(3), qfim(9), move(1)
        self._obs_state_len = self.n_qubits * self.max_steps
        self._obs_thetas_len = 3
        self._obs_qfim_len = 9
        self._obs_move_len = 1
        self._obs_len = self._obs_state_len + self._obs_thetas_len + self._obs_qfim_len + self._obs_move_len

        low_states = np.zeros(self._obs_state_len, dtype=np.float32)
        high_states = np.full(self._obs_state_len, 3, dtype=np.float32)
        low_thetas = np.zeros(3, dtype=np.float32)
        high_thetas = np.ones(3, dtype=np.float32)
        low_qfim = np.zeros(9, dtype=np.float32)
        high_qfim = np.full(9, 1e6, dtype=np.float32)
        low_move = np.array([0.0], dtype=np.float32)
        high_move = np.array([float(self.max_steps)], dtype=np.float32)

        low = np.concatenate([low_states, low_thetas, low_qfim, low_move]).astype(np.float32)
        high = np.concatenate([high_states, high_thetas, high_qfim, high_move]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: 3 qubits, each in {0,1,2,3}
        self.action_space = spaces.MultiDiscrete([4] * self.n_qubits)

        # Pre-allocate an observation buffer (float32 as required)
        self._obs = np.zeros(self._obs_len, dtype=np.float32)
        self._idx_state = slice(0, self._obs_state_len)
        self._idx_thetas = slice(self._obs_state_len, self._obs_state_len + 3)
        self._idx_qfim = slice(self._obs_state_len + 3, self._obs_state_len + 3 + 9)
        self._idx_move = slice(self._obs_state_len + 3 + 9, self._obs_len)

        self.reset()

    def _write_obs(self):
        # Cast state to float32 on assignment (30 elements -> very cheap)
        self._obs[self._idx_state] = self.state.reshape(-1)
        self._obs[self._idx_thetas] = self.thetas
        # Use ravel() (view) since we keep self.qfim buffer persistent and update in-place
        self._obs[self._idx_qfim] = self.qfim.ravel()
        self._obs[self._idx_move] = float(self.current_step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.thetas = np.random.uniform(0.05, 0.5, 3).astype(np.float32)
        self.state.fill(0)
        self.current_step = 0
        self.done = False
        self.qfim.fill(0.0)
        self._write_obs()
        return self._obs, {}

    @staticmethod
    def _all_measured_col(col: np.ndarray) -> bool:
        # col: shape (n_qubits,), uint8
        # True if all zeros
        return np.all(col == 0)

    #@profile
    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is done")

        # Vectorized write: set the column for this step
        # action expected shape (3,)
        self.state[:, self.current_step] = action

        # Advance step
        self.current_step += 1

        # Check termination: all measured at this step column or max steps reached
        all_measured = self._all_measured_col(self.state[:, self.current_step - 1])
        
        terminated = bool(all_measured)  # natural end (task condition)
        truncated = bool(self.current_step >= self.max_steps and not terminated)


        # Default reward (you can keep a small move penalty)
        reward_val = 0.0

        # Update QFIM only if configured to do so
        if self.compute_qfim_each_step:
            # Prefer passing ndarray directly; if qm1 expects lists, do .tolist() once
            # node_lists = self.state.tolist()   # slower; avoid if qm1 can handle numpy
            node_lists = self.state  # ndarray (3,10)

            F = qm1.nodes_to_qfim_D(node_lists, self.thetas, debug=False)
            if isinstance(F, int) and F == -1:
                # Invalid move/path
                self.qfim.fill(0.0)
                # For inference, we typically do not penalize; keep as 0 or a small penalty if needed
                reward_val = 0.0
            else:
                np.copyto(self.qfim, np.asarray(F, dtype=np.float32))


        # Update observation buffer
        self._write_obs()
        # For SB3 safety, returning a copy is unnecessary in most cases; keep as is for speed
        return self._obs, float(reward_val), terminated, truncated, {}

    def render(self): pass
    def close(self): pass