"""
Simulation engines for the Ising Model (Monte Carlo algorithms).
"""

import numpy as np
from .core import IsingLattice

class MetropolisEngine:
    """
    Implements the Metropolis-Hastings algorithm for evolving the Ising Lattice.
    Uses vectorized checkerboard decomposition for efficiency.
    """
    
    def __init__(self, lattice: IsingLattice):
        """
        Args:
            lattice (IsingLattice): The model to simulate.
        """
        self.lattice = lattice

    def step(self, steps: int = 1):
        """
        Perform N Monte Carlo sweeps (updates per site).
        A sweep consists of attempting to flip each spin once (on average).
        
        Args:
            steps (int): Number of sweeps to perform.
        """
        L = self.lattice.size
        # Pre-compute exponential table for performance?
        # Since J, B, T might change, we calculate dE on the fly or vectorially.
        # Vectorized allows computing all Boltzmann factors at once.
        
        for _ in range(steps):
            self._update_subgrid(self.lattice.mask_even)
            self._update_subgrid(self.lattice.mask_odd)

    def _update_subgrid(self, mask):
        """
        Update one checkerboard subgrid (even or odd).
        
        Args:
            mask (np.ndarray): Boolean mask for the subgrid.
        """
        grid = self.lattice.grid
        J = self.lattice.params.J
        B = self.lattice.params.B
        T = self.lattice.params.T
        beta = 1.0 / T
        
        # Calculate sum of neighbors for masked sites
        # Note: neighbors of 'even' sites are 'odd' sites, which are fixed during this step.
        # So we can calculate neighbor sum in parallel.
        
        # Periodic shifts
        neighbors = (
            np.roll(grid, 1, axis=0) + 
            np.roll(grid, -1, axis=0) + 
            np.roll(grid, 1, axis=1) + 
            np.roll(grid, -1, axis=1)
        )
        
        # Energy change if spin i flows: dE = 2 * s_i * (J * sum_neighbors + B)
        # We compute this for ALL 'mask' sites.
        
        total_field = J * neighbors + B
        dE = 2 * grid * total_field
        
        # Metropolis criterion:
        # Flip if dE <= 0 OR rand < exp(-beta * dE)
        
        # Select dE for the subgrid
        dE_sub = dE[mask]
        
        # Boltzmann factors
        # Optimization: only compute exp for dE > 0
        w = np.exp(-beta * dE_sub)
        
        # Random numbers
        r = np.random.random(dE_sub.shape)
        
        # Flip condition
        should_flip = (dE_sub <= 0) | (r < w)
        
        # Update grid
        # We need to assign back into the masked locations.
        # Doing `grid[mask][should_flip] *= -1` doesn't work because `grid[mask]` creates a copy.
        # We must use boolean indexing carefully.
        
        # Create a full-size flip mask
        flip_mask = np.zeros_like(grid, dtype=bool)
        flip_mask[mask] = should_flip
        
        grid[flip_mask] *= -1
