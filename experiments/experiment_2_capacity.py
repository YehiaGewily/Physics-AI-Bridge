
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern, corrupt_pattern

def generate_mix_patterns(n_patterns, size=10):
    """
    Generates a list of 'n_patterns' patterns.
    Starts with 'YEHIA' letters, then adds random patterns if needed.
    """
    letters = ['Y', 'E', 'H', 'I', 'A']
    patterns = []
    
    # Add letters first
    for i in range(min(n_patterns, len(letters))):
        patterns.append(create_letter_pattern(letters[i], size))
        
    # Fill remainder with random patterns
    while len(patterns) < n_patterns:
        # Generate random +1/-1 pattern
        p = np.random.choice([-1, 1], size=size*size)
        patterns.append(p)
        
    return patterns

def run_experiment_2():
    print("=== Experiment 2: Network Capacity ===")
    
    L_grid = 10
    N = L_grid * L_grid
    
    # Experimental Setup
    p_counts = [1, 2, 3, 5, 7, 10]
    n_trials_per_config = 5
    corruption_rate = 0.25
    
    # Theoretical Capacity Limit (approx 0.138 * N)
    capacity_limit = 0.138 * N
    
    results = [] # To store mean accuracy per P
    errors = []  # To store std dev per P
    
    print(f"Network Size: N={N}")
    print(f"Theoretical Capacity (~0.14N): {capacity_limit:.2f} patterns")
    print("-" * 50)
    
    for P in p_counts:
        trial_accuracies = []
        
        for t in range(n_trials_per_config):
            # 1. Initialize & Train
            net = HopfieldNetwork(n_neurons=N)
            patterns = generate_mix_patterns(P, L_grid)
            net.train(patterns)
            
            # 2. Test Recall on ALL patterns
            successes = 0
            for pat in patterns:
                # Corrupt
                noisy = corrupt_pattern(pat, corruption_rate)
                
                # Recall
                recovered = net.recall(noisy, max_steps=50, mode='async')
                
                # Check (allow inverse too)
                if np.array_equal(recovered, pat) or np.array_equal(recovered, -pat):
                    successes += 1
            
            accuracy = successes / P
            trial_accuracies.append(accuracy)
            
        mean_acc = np.mean(trial_accuracies)
        std_acc = np.std(trial_accuracies)
        
        results.append(mean_acc)
        errors.append(std_acc)
        
        print(f"P={P:2d} | Mean Accuracy: {mean_acc*100:5.1f}% (±{std_acc*100:4.1f}%)")

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Data
    plt.errorbar(p_counts, results, yerr=errors, fmt='-o', 
                 capsize=5, color='blue', label='Measured Accuracy', linewidth=2)
    
    # Theoretical Limit Line
    plt.axvline(x=capacity_limit, color='red', linestyle='--', 
                label=f'Capacity Limit (~{capacity_limit:.1f})')
    
    # Formatting
    plt.title(f"Hopfield Network Capacity (N={N}, Noise={corruption_rate*100}%)")
    plt.xlabel("Number of Stored Patterns (P)")
    plt.ylabel("Recall Accuracy")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save
    output_path = "results/figures/exp2_capacity_results.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nCapacity plot saved to {output_path}")

if __name__ == "__main__":
    run_experiment_2()
