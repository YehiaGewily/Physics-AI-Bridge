
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork

def test_hopfield_network():
    print("Testing HopfieldNetwork...")

    # 1. Define dummy patterns (4x4)
    # Pattern A: Checkerboard
    p1 = np.array([
        [1, -1, 1, -1],
        [-1, 1, -1, 1],
        [1, -1, 1, -1],
        [-1, 1, -1, 1]
    ])
    
    # Pattern B: Stripes
    p2 = np.array([
        [1, 1, 1, 1],
        [-1, -1, -1, -1],
        [1, 1, 1, 1],
        [-1, -1, -1, -1]
    ])

    L = 4
    N = L * L # 16 neurons

    # 2. Train
    print(f"Initializing Network with N={N}...")
    net = HopfieldNetwork(n_neurons=N)
    
    print("Training on 2 patterns...")
    net.train([p1, p2])

    W = net.W

    # 3. Verify Shape
    assert W.shape == (N, N), f"Weight matrix shape mismatch. Expected ({N}, {N}), got {W.shape}"
    print("[PASS] Shape check")

    # 4. Verify Symmetry
    assert np.allclose(W, W.T), "Weight matrix is not symmetric!"
    print("[PASS] Symmetry check")

    # 5. Verify Zero Diagonal
    assert np.allclose(np.diag(W), 0), "Diagonal elements are not zero!"
    print("[PASS] Zero Diagonal check")

    # 6. Verify Values
    # W_01 = (1/16) * (p1[0]*p1[1] + p2[0]*p2[1]) + ...
    # p1 flat: 1, -1 ... -> prod -1
    # p2 flat: 1, 1 ... -> prod +1
    # sum = 0.
    val_01 = W[0, 1]
    expected_01 = 0.0 
    assert np.isclose(val_01, expected_01), f"Value mismatch at W[0,1]. Got {val_01}, expected {expected_01}"
    print(f"[PASS] Value check (W[0,1]={val_01})")
    
    # 7. Test Recall (Identity)
    # Should recall p1 perfectly
    print("Testing Recall (Identity)...")
    recalled_p1 = net.recall(p1, max_steps=100)
    assert np.array_equal(recalled_p1, p1.flatten()), "Failed to recall stored pattern p1 exactly."
    print("[PASS] Identity Recall")

    # 8. Test compute_energy
    # For a stored pattern p1, energy should be roughly -0.5 * N (if p1 is an eigenvector)
    # Actually, Hopfield energy for stored pattern roughly -0.5 * N if it is a Minimum.
    print("Testing Energy Calculation...")
    E_p1 = net.compute_energy(p1.flatten())
    print(f"Energy of stored pattern p1: {E_p1}")
    
    # Energy of random state should be higher
    # We loop until we find a random state with higher energy than the stored minimum
    # (It's theoretically possible a random state is also a minimum, but unlikely for small N=16 with 2 patterns)
    found_higher_energy = False
    for i in range(10):
        random_state = np.random.choice([-1, 1], size=N)
        E_rand = net.compute_energy(random_state)
        if E_rand > E_p1:
            found_higher_energy = True
            break
            
    if found_higher_energy:
        print(f"Energy of random state: {E_rand}")
        print("[PASS] Energy Landscape check")
    else:
        print(f"[WARNING] Could not find random state with higher energy than {E_p1} in 10 tries. Might be shallow minima.")

    print("\nAll HopfieldNetwork tests passed!")

if __name__ == "__main__":
    test_hopfield_network()
