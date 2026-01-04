
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern, corrupt_pattern

def run_experiment_1():
    print("=== Experiment 1: Single Pattern Robustness ===")
    
    # 1. Initialize
    L_grid = 10
    N = L_grid * L_grid
    net = HopfieldNetwork(n_neurons=N)
    
    # 2. Store Pattern 'Y'
    target_pattern = create_letter_pattern('Y', size=L_grid)
    net.train([target_pattern])
    print(f"Network trained on pattern 'Y' (Size: {N} neurons)")
    
    # Parameters
    corruption_levels = [0.1, 0.25, 0.5]
    n_trials = 10
    max_steps = 50 # Epochs (each epoch is N updates)
    
    results = []

    # 3. Test Loops
    for noise in corruption_levels:
        print(f"\nTesting Corruption Level: {noise*100}%")
        success_count = 0
        total_steps = 0
        total_energy = 0
        
        for t in range(n_trials):
            # Generate Corrupted Input
            noisy_input = corrupt_pattern(target_pattern, corruption_rate=noise)
            
            # Run Recall Manually to track steps
            net.state = noisy_input.flatten().copy()
            prev_state = net.state.copy()
            
            steps_taken = max_steps
            converged = False
            
            for epoch in range(max_steps):
                # 1 Epoch = separate sync or N async updates
                # Let's use mode='sync' for deterministic step counting in this experiment,
                # or async. User didn't specify, but async is more physical.
                # However, async requires N updates to be comparable to 1 sync step.
                # The prompt says "max 50 iterations", usually implies epochs.
                
                # Perform 1 Epoch of updates
                # Using async updates (N single flips per epoch)
                for _ in range(N):
                     net.update(net.state, mode='async')
                
                # Check Convergence
                if np.array_equal(net.state, prev_state):
                    steps_taken = epoch
                    converged = True
                    break
                prev_state = net.state.copy()
            
            # Check Success
            # Success if final state matches target (or inverted target, technically, 
            # but for single pattern stored, expected is exact match)
            is_success = np.array_equal(net.state, target_pattern)
            # Also check inverted state (Hopfield networks store +/- pairs)
            # But single pattern loading usually makes that pattern the global minimum.
            if not is_success and np.array_equal(net.state, -target_pattern):
                 # Consider inverted state a "success" structurally? 
                 # Usually yes, but let's be strict for "Recall".
                 pass
            
            if is_success:
                success_count += 1
            
            total_steps += steps_taken
            total_energy += net.energy()
            
        # Stats
        avg_success = success_count / n_trials
        avg_steps = total_steps / n_trials
        avg_energy = total_energy / n_trials
        
        results.append({
            "Noise Level": noise,
            "Success Rate": avg_success,
            "Avg Steps": avg_steps,
            "Avg Energy": avg_energy
        })
        
        print(f"  Success Rate: {avg_success*100:.1f}%")
        print(f"  Avg Steps:    {avg_steps:.1f}")
        print(f"  Avg Energy:   {avg_energy:.2f}")

    # 4. Summary Plot
    df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Success Rate
    ax1.plot(df["Noise Level"], df["Success Rate"], marker='o', color='green', linewidth=2)
    ax1.set_title("Robustness: Success Rate vs Noise")
    ax1.set_xlabel("Corruption Rate")
    ax1.set_ylabel("Success Rate (0-1)")
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True)
    
    # Steps
    ax2.plot(df["Noise Level"], df["Avg Steps"], marker='s', color='blue', linewidth=2)
    ax2.set_title("Dynamics: Convergence Speed")
    ax2.set_xlabel("Corruption Rate")
    ax2.set_ylabel("Average Epochs to Converge")
    ax2.grid(True)
    
    plt.tight_layout()
    output_path = "results/figures/exp1_single_pattern_results.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"\nSummary plot saved to {output_path}")
    
    # 5. Print Table
    print("\n=== Final Results Table ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_experiment_1()
