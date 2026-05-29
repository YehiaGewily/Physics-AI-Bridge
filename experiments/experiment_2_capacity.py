
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
    Starts with 'NOVER' letters, then adds random patterns if needed.
    """
    letters = ['N', 'O', 'V', 'E', 'R']
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
    p_counts = [1, 2, 3, 5, 7, 10, 12, 14]
    n_trials_per_config = 50
    corruption_rate = 0.25
    
    # Theoretical Capacity Limit (approx 0.138 * N)
    capacity_limit = 0.138 * N
    
    # --- Pattern Quality Check ---
    print("\n--- Pattern Overlap Matrix (NOVER) ---")
    letters = ['N', 'O', 'V', 'E', 'R']
    letter_patterns = [create_letter_pattern(l, L_grid) for l in letters]
    overlap_matrix = np.zeros((len(letters), len(letters)))
    for i in range(len(letters)):
        for j in range(len(letters)):
            overlap_matrix[i, j] = (1.0 / N) * np.dot(letter_patterns[i], letter_patterns[j])
    
    print("     " + "  ".join(f"{l:>6s}" for l in letters))
    for i, l in enumerate(letters):
        row_str = "  ".join(f"{overlap_matrix[i, j]:6.3f}" for j in range(len(letters)))
        print(f"  {l}  {row_str}")
    
    # Warn if any off-diagonal overlap exceeds 0.3
    for i in range(len(letters)):
        for j in range(i + 1, len(letters)):
            if abs(overlap_matrix[i, j]) > 0.3:
                print(f"  WARNING: High overlap between {letters[i]} and {letters[j]}: {overlap_matrix[i, j]:.3f}")
    print("  Pattern overlap check complete.")
    
    results = []     # mean accuracy per P
    sems = []        # standard error of mean per P
    tolerant_results = []  # inversion-tolerant accuracy per P
    trial_rows = []  # per-trial CSV data
    
    print(f"\nNetwork Size: N={N}")
    print(f"Theoretical Capacity (~0.14N): {capacity_limit:.2f} patterns")
    print("-" * 50)
    
    for P in p_counts:
        trial_accuracies = []
        trial_tolerant = []
        
        for t in range(n_trials_per_config):
            # 1. Initialize & Train
            net = HopfieldNetwork(n_neurons=N)
            patterns = generate_mix_patterns(P, L_grid)
            net.train(patterns)
            
            # 2. Test Recall on ALL patterns
            strict_successes = 0
            tolerant_successes = 0
            for pat_idx, pat in enumerate(patterns):
                # Corrupt
                noisy = corrupt_pattern(pat, corruption_rate)
                
                # Recall
                recovered = net.recall(noisy, max_steps=50, mode='async')
                
                # Check — strict (pattern only) vs tolerant (pattern or inverse)
                is_strict = np.array_equal(recovered, pat)
                is_tolerant = is_strict or np.array_equal(recovered, -pat)
                
                if is_strict:
                    strict_successes += 1
                if is_tolerant:
                    tolerant_successes += 1
                
                trial_rows.append({
                    "n_patterns": P,
                    "pattern_index": pat_idx,
                    "trial": t,
                    "strict_success": int(is_strict),
                    "inversion_tolerant_success": int(is_tolerant),
                    "final_energy": net.energy(),
                })
            
            strict_accuracy = strict_successes / P
            tolerant_accuracy = tolerant_successes / P
            trial_accuracies.append(strict_accuracy)
            trial_tolerant.append(tolerant_accuracy)
        
        mean_acc = np.mean(trial_accuracies)
        sem_acc = np.std(trial_accuracies, ddof=1) / np.sqrt(n_trials_per_config)
        mean_tolerant = np.mean(trial_tolerant)
        
        results.append(mean_acc)
        sems.append(sem_acc)
        tolerant_results.append(mean_tolerant)
        
        print(f"P={P:2d} | Strict: {mean_acc*100:5.1f}% (±SEM {sem_acc*100:4.1f}%) | Tolerant: {mean_tolerant*100:5.1f}%")

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left: Capacity curve with SEM error bars ---
    ax1.errorbar(p_counts, results, yerr=sems, fmt='-o',
                 capsize=5, color='blue', label='Strict Accuracy', linewidth=2)
    # Shaded ±1 SEM region
    ax1.fill_between(p_counts,
                     [m - s for m, s in zip(results, sems)],
                     [m + s for m, s in zip(results, sems)],
                     alpha=0.2, color='blue', label='±1 SEM')
    # Inversion-tolerant curve (dashed, for comparison)
    ax1.plot(p_counts, tolerant_results, '--s', color='blue', alpha=0.4,
             markersize=5, label='Tolerant (pattern or inverse)')
    
    # Theoretical limit line
    ax1.axvline(x=capacity_limit, color='red', linestyle='--',
                label=f'Capacity Limit (~{capacity_limit:.1f})')
    
    # 50% accuracy reference line
    ax1.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, label='50% accuracy')
    
    # Annotate point where accuracy first drops below 50%
    for i, (p, acc) in enumerate(zip(p_counts, results)):
        if acc < 0.5:
            ax1.annotate(f'First <50%\nat P={p}',
                         xy=(p, acc), xytext=(p + 1, acc + 0.15),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=9, color='red', fontweight='bold')
            break
    
    ax1.set_title(f"Hopfield Network Capacity (N={N}, Noise={corruption_rate*100:.0f}%)", fontsize=12)
    ax1.set_xlabel("Number of Stored Patterns (P)")
    ax1.set_ylabel("Recall Accuracy")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # --- Right: Overlap matrix heatmap ---
    im = ax2.imshow(overlap_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(letters)))
    ax2.set_yticks(range(len(letters)))
    ax2.set_xticklabels(letters)
    ax2.set_yticklabels(letters)
    ax2.set_title("Pattern Overlap Matrix (NOVER)", fontsize=12)
    # Annotate cells with values
    for i in range(len(letters)):
        for j in range(len(letters)):
            color = 'white' if abs(overlap_matrix[i, j]) > 0.5 else 'black'
            ax2.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=9, color=color)
    plt.colorbar(im, ax=ax2, label='Overlap')
    
    plt.tight_layout()
    
    # Save to both output paths
    output_paths = [
        "results/hopfield/exp2_capacity_results.png",
        "experiments/results/figures/exp2_capacity_results.png",
    ]
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path}")
    plt.close()
    
    # --- Save CSV ---
    csv_path = "results/hopfield/exp2_results.csv"
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(csv_path, index=False)
    print(f"Trial data saved to {csv_path}")

if __name__ == "__main__":
    run_experiment_2()
