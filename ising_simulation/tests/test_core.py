import pytest
import numpy as np
from ising.core import IsingLattice

def test_lattice_initialization():
    L = 10
    sim = IsingLattice(L, T=2.0, initial_state='cold')
    assert sim.grid.shape == (L, L)
    assert np.all(sim.grid == 1)
    assert sim.magnetization == L*L

def test_magnetization():
    sim = IsingLattice(4, T=1.0)
    sim.grid = np.ones((4,4))
    assert sim.magnetization == 16.0
    
    sim.grid[0,0] = -1
    assert sim.magnetization == 14.0

def test_energy_ferromagnetic():
    # L=4, all spins up. J=1, B=0
    # Each site has 4 neighbors. Total links = 2N.
    # Energy = -J * (2N links) * 1 = -2N
    L = 4
    sim = IsingLattice(L, T=1.0, J=1.0)
    sim.grid = np.ones((L,L))
    
    expected_E = -1.0 * (2 * L*L)
    assert sim.energy() == expected_E

def test_energy_antiferromagnetic():
    # Checkerboard pattern for J=-1 (favored) or J=1 (unfavored)
    L = 4
    sim = IsingLattice(L, T=1.0)
    
    # Create checkerboard
    x, y = np.indices((L, L))
    sim.grid = np.ones((L,L))
    sim.grid[(x+y)%2 == 1] = -1
    
    # Each link connects +1 and -1. Product is -1.
    # Energy = -J * Sum(-1) = -J * (2N * -1) = +2N * J. (If J=1)
    
    expected_E = 1.0 * (2 * L * L)
    assert sim.energy() == expected_E
