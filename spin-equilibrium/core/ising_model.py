import numpy as np

class IsingSimulation:
    def __init__(self, size: int, J: float = 1.0, temperature: float = 2.269, B: float = 0.0):
        """
        Initialize the Ising Model.
        
        Args:
            size (int): The linear size of the grid (LxL).
            J (float): The interaction strength (default 1.0 for ferromagnetism).
            temperature (float): The temperature of the system.
            B (float): External Magnetic Field strength.
        """
        self.size = size
        self.J = J
        self.temperature = temperature
        self.B = B
        
        # Initialize grid with random spins (-1 or 1)
        self.grid = np.random.choice([-1, 1], size=(size, size)).astype(np.int8)
        
        # Pre-compute checkerboard masks
        x, y = np.indices((size, size))
        self.mask_even = (x + y) % 2 == 0
        self.mask_odd = (x + y) % 2 == 1
        
        # Pre-compute exponential lookup table for efficiency if needed, 
        # but for continuous T it's better to compute on fly or cache per T update.
        # We will compute on fly for flexibility.

    @property
    def magnetization(self) -> float:
        """Calculate the total magnetization."""
        return np.sum(self.grid)

    def energy(self) -> float:
        """
        Calculate the total energy of the system.
        H = -J * sum(sigma_i * sigma_j) - B * sum(sigma_i)
        """
        # Shift grid to get neighbors
        neighbors = (
            np.roll(self.grid, 1, axis=0) + 
            np.roll(self.grid, -1, axis=0) + 
            np.roll(self.grid, 1, axis=1) + 
            np.roll(self.grid, -1, axis=1)
        )
        # Factor of 1/2 because each pair is counted twice
        interaction_energy = -self.J * np.sum(self.grid * neighbors) / 2.0
        
        # External Field term
        field_energy = -self.B * np.sum(self.grid)
        
        return interaction_energy + field_energy

    def metropolis_step(self, steps_per_sweep: int = 1):
        """
        Perform Metropolis-Hastings update using Checkerboard Vectorization.
        
        Args:
            steps_per_sweep (int): Number of full lattice sweeps.
        """
        for _ in range(steps_per_sweep):
            # We alternate between even and odd sublattices
            for mask in [self.mask_even, self.mask_odd]:
                # Calculate neighbors sum
                neighbors = (
                    np.roll(self.grid, 1, axis=0) + 
                    np.roll(self.grid, -1, axis=0) + 
                    np.roll(self.grid, 1, axis=1) + 
                    np.roll(self.grid, -1, axis=1)
                )
                
                # Calculate energy change dE for flipping spins
                # dE = 2 * s_i * (J * sum_neighbors + B)
                # We only care about sites in the current mask
                dE = 2 * self.grid * (self.J * neighbors + self.B)
                
                # Metropolis criterion:
                # Flip if dE < 0 OR random < exp(-dE/T)
                
                # Generate random numbers for the whole grid (vectorized)
                # Optimization: We could generate only for masked, but indexing is costly.
                # Masking the operation is usually faster in NumPy for dense arrays.
                random_vals = np.random.random(self.grid.shape)
                
                # Calculate acceptance probability
                # We use specific logic to avoid exp overflow/warnings although dE usually well behaved
                # exp(-dE/T)
                
                # Logic:
                # 1. dE <= 0: Always flip. (exp(-dE/T) >= 1)
                # 2. dE > 0: Flip with prob exp(-dE/T)
                
                # Boolean mask for flips
                # Case 1: dE <= 0
                flip_condition = (dE <= 0)
                
                # Case 2: dE > 0 and random < exp(...)
                # We only compute exp where dE > 0 to save time? 
                # Actually vectorized exp is fast.
                
                # Combined condition:
                # (dE <= 0) | (random_vals < np.exp(-dE / self.temperature))
                # AND applied only to current checkerboard mask
                
                should_flip = (dE <= 0) | (random_vals < np.exp(-dE / self.temperature))
                
                # Apply mask
                final_flip_mask = should_flip & mask
                
                # Apply flips: s -> -s
                self.grid[final_flip_mask] *= -1

    def compute_spatial_correlation(self, max_r: int) -> np.ndarray:
        """
        Compute the spatial correlation function C(r) = <s_i * s_{i+r}> - <s_i>^2
        Averaged over all directions and lattice sites.
        
        Args:
            max_r (int): Maximum distance to compute correlation for.
            
        Returns:
            np.ndarray: Array of correlation values for r = 0 to max_r.
        """
        N = self.size * self.size
        avg_m = np.mean(self.grid)
        squared_avg_m = avg_m ** 2
        
        correlations = []
        
        # Naive implementation using np.roll for each distance
        # For r=0, correlation is 1 (s_i^2 = 1) - <M>^2
        # But standard definition C(r) usually just <s_0 s_r> or <s_0 s_r> - <m>^2
        # We will return <s_i s_{i+r}> - <m>^2
        
        for r in range(max_r + 1):
            if r == 0:
                correlations.append(1.0 - squared_avg_m)
                continue
                
            # We average correlations in x and y directions for distance r
            # Horizontal neighbors at distance r
            corr_x = np.mean(self.grid * np.roll(self.grid, r, axis=1))
            
            # Vertical neighbors at distance r
            corr_y = np.mean(self.grid * np.roll(self.grid, r, axis=0))
            
            # Average both directions
            c_r = (corr_x + corr_y) / 2.0 - squared_avg_m
            correlations.append(c_r)
            
        return np.array(correlations)

    def set_temperature(self, t: float):
        self.temperature = t

