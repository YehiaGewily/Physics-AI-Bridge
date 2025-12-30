"""
Core data structures for the Ising Model simulation.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SimulationParams:
    """
    Configuration parameters for the Ising Model.
    
    Args:
        L (int): Linear lattice size (LxL grid).
        T (float): Temperature in units of J/k_B.
        J (float): Coupling constant (default 1.0).
        B (float): External magnetic field (default 0.0).
    """
    L: int
    T: float
    J: float = 1.0
    B: float = 0.0

class IsingLattice:
    """
    Represents the 2D Ising Model Grid and its physical properties.
    
    Attributes:
        params (SimulationParams): System parameters (L, T, J, B).
        grid (np.ndarray): 2D array of spins (+1 or -1).
        size (int): Linear dimension L.
    """
    
    def __init__(self, L: int, T: float, J: float = 1.0, B: float = 0.0, 
                 initial_state: Optional[str] = 'random'):
        """
        Initialize the Ising Lattice.

        Args:
            L (int): Lattice size.
            T (float): Temperature.
            J (float): Interaction strength.
            B (float): External field.
            initial_state (str): 'random', 'cold' (all up), or 'hot' (random).
        """
        self.params = SimulationParams(L, T, J, B)
        self.size = L
        
        # Initialize Spins
        if initial_state == 'cold':
            self.grid = np.ones((L, L), dtype=np.int8)
        else:
            self.grid = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

        # Pre-compute neighbor indices could be done here if avoiding roll,
        # but numpy.roll is fast enough for MCMC updates if vectorized properly.
        # However, for Metropolis, we used checkerboard. 
        # We'll store masks here.
        x, y = np.indices((L, L))
        self.mask_even = (x + y) % 2 == 0
        self.mask_odd = (x + y) % 2 == 1

    @property
    def magnetization(self) -> float:
        """Total magnetization M = sum(s_i)."""
        return float(np.sum(self.grid))

    def energy(self) -> float:
        """
        Calculate total energy of the configuration: H = -J sum <ij> s_i s_j - B sum s_i.
        
        Returns:
            float: Total energy.
        """
        # Periodic boundaries: neighbors are rolled
        # Shift Right, Left, Up, Down
        neighbors = (
            np.roll(self.grid, 1, axis=0) + 
            np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + 
            np.roll(self.grid, -1, axis=1)
        )
        # Interaction term (each pair counted twice, so divide by 2)
        interaction = -self.params.J * np.sum(self.grid * neighbors) / 2.0
        
        # External field term
        magnetic = -self.params.B * np.sum(self.grid)
        
        return interaction + magnetic

    def reset(self, mode: str = 'random'):
        """Reset the lattice configuration."""
        if mode == 'cold':
            self.grid = np.ones((self.size, self.size), dtype=np.int8)
        elif mode == 'hot' or mode == 'random':
            self.grid = np.random.choice([-1, 1], size=(self.size, self.size)).astype(np.int8)
