
import numpy as np
from typing import List, Optional, Union

class HopfieldNetwork:
    """
    A Hopfield Network for associative memory implementing Hebbian learning.
    """
    def __init__(self, n_neurons: int):
        """
        Initialize the Hopfield Network.
        
        Args:
            n_neurons (int): Total number of neurons in the network.
                             If representing a grid, this is L*L.
        """
        self.n_neurons = n_neurons
        # Weight matrix W of shape (N, N) initialized to zeros
        self.W = np.zeros((n_neurons, n_neurons))
        # Current state s of shape (N,)
        self.state = np.ones(n_neurons)

    def train(self, patterns: List[np.ndarray]) -> 'HopfieldNetwork':
        """
        Train the network using Hebbian Learning rule.
        W_ij = (1/N) * sum_mu (xi_i^mu * xi_j^mu)
        
        Args:
            patterns: List of interaction patterns (1D or 2D arrays).
                      Must contain values +1 and -1.
            
        Returns:
            self: The trained network instance.
        """
        if not patterns:
            raise ValueError("No patterns provided for training.")

        # Flatten patterns to 1D vectors of size N
        X_list = []
        for p in patterns:
            p_flat = p.flatten()
            if not np.all(np.isin(p_flat, [-1, 1])):
                raise ValueError("Patterns must correspond to spins +1 and -1.")
            if p_flat.size != self.n_neurons:
                raise ValueError(f"Pattern size {p_flat.size} does not match network size {self.n_neurons}")
            X_list.append(p_flat)
        
        # Stack patterns into matrix X of shape (P, N)
        X = np.array(X_list)
        
        # Hebbian Rule: W = (1/N) * X.T @ X
        self.W = (X.T @ X) / self.n_neurons
        
        # Ensure W remains symmetric (numerical stability)
        self.W = (self.W + self.W.T) / 2
        
        # Constraint: No self-connections (W_ii = 0)
        np.fill_diagonal(self.W, 0.0)
        
        return self

    def update(self, state: np.ndarray, mode: str = 'async') -> np.ndarray:
        """
        Update the network state.

        Args:
            state (np.ndarray): Current state configuration (1D array).
                                If mode='async', this is modified in-place.
            mode (str): 'sync' for synchronous update, 'async' for asynchronous (random neuron).
        
        Returns:
            np.ndarray: The updated state.
        """
        # Ensure state is valid
        if state.ndim != 1 or state.size != self.n_neurons:
             raise ValueError(f"State must be 1D array of size {self.n_neurons}")

        if mode == 'sync':
            # Synchronous: s(t+1) = sign(W @ s(t))
            h = self.W @ state
            state_new = np.sign(h)
            # Handle 0 case -> 1
            state_new[state_new == 0] = 1 
            # Update in-place to be consistent with async
            state[:] = state_new[:]
            
        elif mode == 'async':
            # Asynchronous: Update random neurons one by one
            # Pick random neuron index i
            idx = np.random.randint(0, self.n_neurons)
            # Compute h_i = sum(W[i,j] * s[j])
            h_i = self.W[idx] @ state
            # Update: s[i] = sign(h_i) (use +1 if h_i = 0)
            state[idx] = 1 if h_i >= 0 else -1
            
        else:
            raise ValueError(f"Unknown update mode: {mode}")
            
        return state

    def recall(self, pattern: np.ndarray, max_steps: int = 100, mode: str = 'async') -> np.ndarray:
        """
        Retrieve a stored memory from a possibly corrupted input cue.
        
        Args:
            pattern (np.ndarray): The input cue pattern.
            max_steps (int): Maximum number of update steps (or epochs for sync).
            mode (str): Update mode ('sync' or 'async').
            
        Returns:
            np.ndarray: The converged state (flattened).
        """
        # Set initial state
        if pattern.size != self.n_neurons:
            raise ValueError(f"Input pattern size {pattern.size} does not match network size {self.n_neurons}")
        self.state = pattern.flatten().copy()
        
        # Run updates until convergence or max_steps
        prev_state = self.state.copy()
        
        # If async, we treat max_steps as number of full N-size sweeps equivalent
        total_updates = max_steps if mode == 'sync' else max_steps * self.n_neurons
        
        if mode == 'sync':
            for _ in range(max_steps):
                self.state = self.update(self.state, mode='sync')
                if np.array_equal(self.state, prev_state):
                    break
                prev_state = self.state.copy()
        else:
            # Check convergence periodically
            check_interval = self.n_neurons
            for step in range(total_updates):
                self.state = self.update(self.state, mode='async')
                
                # Check convergence every 'N' updates
                if (step + 1) % check_interval == 0:
                     if np.array_equal(self.state, prev_state):
                         break
                     prev_state = self.state.copy()

        return self.state

    def compute_energy(self, state: np.ndarray) -> float:
        """
        Calculate the energy of a specific state configuration.
        
        This corresponds to the Ising Hamiltonian with learned weights:
        E = -0.5 * sum_{i,j} W_{ij} s_i s_j
        
        The factor of 0.5 accounts for the double counting of pairs (i,j) and (j,i).
        Energy minimization in this landscape corresponds to memory retrieval.
        
        Args:
            state (np.ndarray): The state configuration to evaluate (1D array).
            
        Returns:
            float: The scalar energy value.
        """
        if state.ndim != 1 or state.size != self.n_neurons:
             raise ValueError(f"State must be 1D array of size {self.n_neurons}")
             
        # E = -0.5 * s.T @ W @ s
        return -0.5 * np.dot(state, np.dot(self.W, state))

    def energy(self) -> float:
        """
        Calculate the energy of the current state (self.state).
        Convenience wrapper for compute_energy.
        """
        return self.compute_energy(self.state)
