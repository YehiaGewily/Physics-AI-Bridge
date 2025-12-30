import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

def test_ising():
    print("Testing Ising Simulation...")
    L = 32
    model = IsingSimulation(size=L, temperature=2.269)
    
    initial_E = model.energy()
    print(f"Initial Energy: {initial_E}")
    
    # Run a few steps
    model.metropolis_step(steps_per_sweep=10)
    
    final_E = model.energy()
    print(f"Final Energy: {final_E}")
    
    assert model.grid.shape == (L, L)
    print("Grid shape OK")
    
    corr = model.compute_spatial_correlation(max_r=5)
    print(f"Spatial Correlations (r=0 to 5): {corr}")
    assert len(corr) == 6
    
    print("Test Passed!")

if __name__ == "__main__":
    test_ising()
