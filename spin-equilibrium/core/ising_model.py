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
        self.grid = np.random.choice([-1, 1], size=(size, size))
        
        # Statistics storage (Ring buffer style or full history)
        # Storing last 1000 steps for rolling variance
        self.energy_history = []
        self.magnetization_history = []
        self.history_limit = 1000

    
    def energy(self) -> float:
        """
        Calculate the total energy of the system.
        H = -J * sum(sigma_i * sigma_j)
        """
        # Shift grid to get neighbors
        neighbors = (
            np.roll(self.grid, 1, axis=0) + 
            np.roll(self.grid, -1, axis=0) + 
            np.roll(self.grid, 1, axis=1) + 
            np.roll(self.grid, -1, axis=1)
        )
        total_energy = -self.J * np.sum(self.grid * neighbors) / 2.0
        
        # Add External Field term: -B * sum(sigma)
        total_energy -= self.B * np.sum(self.grid)
        return total_energy

    def metropolis_step(self, steps_per_sweep: int = 1):
        """
        Perform Metropolis-Hastings update using Checkerboard Vectorization.
        Allows for parallel updates of non-interacting spins.
        
        Args:
            steps_per_sweep (int): Number of checkerboard sweeps (updates all pixels).
        """
        for _ in range(steps_per_sweep):
            # Checkerboard update: 0 = even sites, 1 = odd sites
            for parity in [0, 1]:
                # Calculate neighbors sum for whole grid
                # (We calculate for all, but only use half. Optimization possible but complex in pure numpy without slicing)
                # Slicing approach:
                # To fully optimize, we would slice grid into chessboards.
                # Here we use masking which is still much faster than loops.
                
                # Roll neighbors
                # Note: This is re-calculated for each parity step.
                neighbors = (
                    np.roll(self.grid, 1, axis=0) + 
                    np.roll(self.grid, -1, axis=0) + 
                    np.roll(self.grid, 1, axis=1) + 
                    np.roll(self.grid, -1, axis=1)
                )
                
                # Calculate dE for all sites
                # dE = 2 * s * (J * sum_neighbors + B)
                dE = 2 * self.grid * (self.J * neighbors + self.B)
                
                # Create mask for current parity
                # (i + j) % 2 == parity
                # Efficient mask generation
                x, y = np.indices(self.grid.shape)
                mask = (x + y) % 2 == parity
                
                # Select only relevant dE
                # We want to flip if dE < 0 OR rand < exp(-dE/T)
                
                # Vectorized decision
                # Condition 1: dE < 0 (Always flip)
                # Condition 2: Random < Exp (Probabilistic flip)
                
                # Calculate transition probs for only the valid mask
                # To avoid overflow/warning in exp, we can compute only where necessary,
                # but calculating all and masking is easier for code clarity.
                
                # Random probabilities
                random_probs = np.random.rand(*self.grid.shape)
                
                # Flip condition
                # dE < 0  OR  random < exp(...)
                # Note: if dE < 0, then exp(-dE/T) > 1, so random < exp is always true.
                # So we just need: random < exp(-dE/T)
                
                flip_mask = (random_probs < np.exp(-dE / self.temperature)) & mask
                
                # Apply flips
                self.grid[flip_mask] *= -1
                
        # Update statistics after full sweep(s)
        current_E = self.energy()
        current_M = np.sum(self.grid) # Magnetization
        
        self.energy_history.append(current_E)
        self.magnetization_history.append(current_M)
        
        if len(self.energy_history) > self.history_limit:
            self.energy_history.pop(0)
            self.magnetization_history.pop(0)

    def get_statistics(self):
        """
        Returns stats dictionary including Variance of E and M.
        """
        if len(self.energy_history) < 2:
            current_E = self.energy() if not self.energy_history else self.energy_history[-1]
            current_M = np.sum(self.grid) if not self.magnetization_history else self.magnetization_history[-1]
            return {
                "mean_E": current_E, 
                "mean_M": current_M, 
                "var_E": 0, 
                "var_M": 0, 
                "Cv": 0, 
                "Chi": 0
            }
            
        var_E = np.var(self.energy_history)
        var_M = np.var(self.magnetization_history)
        
        # Specific Heat Cv = Var(E) / (k * T^2)  (Assume k=1)
        Cv = var_E / (self.temperature ** 2)
        
        # Susceptibility Chi = Var(M) / (k * T)
        Chi = var_M / self.temperature
        
        return {
            "mean_E": np.mean(self.energy_history),
            "mean_M": np.mean(self.magnetization_history),
            "var_E": var_E,
            "var_M": var_M,
            "Cv": Cv,
            "Chi": Chi
        }

    def set_temperature(self, t: float):
        self.temperature = t
