
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
    
    # 2. Store Pattern 'N'
    target_pattern = create_letter_pattern('N', size=L_grid)
    net.train([target_pattern])
    print(f"Network trained on pattern 'N' (Size: {N} neurons)")
    
    # Parameters
    corruption_levels = [0.1, 0.25, 0.5]
    n_trials = 50
    max_steps = 50  # Epochs (each epoch is N async updates)
    
    results = []
    energy_traces = {}  # {noise: [(step, energy), ...]} for one example trial per noise level
    trial_rows = []     # per-trial data for CSV
    recovery_examples = {}  # {noise: (corrupted, recalled)} for successful trial

    # 3. Test Loops
    for noise in corruption_levels:
        print(f"\nTesting Corruption Level: {noise*100:.0f}%")
        strict_successes = []    # exact pattern match only
        tolerant_successes = []  # pattern OR inverse
        total_steps = 0
        total_energy = 0
        example_trace = []  # energy trace for one trial
        found_recovery = False
        
        for t in range(n_trials):
            # Generate Corrupted Input
            noisy_input = corrupt_pattern(target_pattern, corruption_rate=noise)
            corrupted_state = noisy_input.copy()
            
            # Run Recall Manually to track steps
            net.state = noisy_input.flatten().copy()
            prev_state = net.state.copy()
            initial_energy = net.energy()
            
            steps_taken = max_steps
            trace = []  # track energy per epoch for this trial
            
            for epoch in range(max_steps):
                # Record energy before update
                trace.append((epoch, net.energy()))
                
                # Perform 1 Epoch of async updates (N single flips)
                for _ in range(N):
                    net.update(net.state, mode='async')
                
                # Check Convergence
                if np.array_equal(net.state, prev_state):
                    steps_taken = epoch + 1
                    trace.append((epoch + 1, net.energy()))
                    break
                prev_state = net.state.copy()
            else:
                # Reached max_steps without convergence
                trace.append((max_steps, net.energy()))
            
            # Save energy trace from first trial as example
            if t == 0:
                example_trace = trace
            
            # Check Success — two metrics
            final_energy = net.energy()
            is_strict = np.array_equal(net.state, target_pattern)
            is_tolerant = is_strict or np.array_equal(net.state, -target_pattern)
            
            strict_successes.append(int(is_strict))
            tolerant_successes.append(int(is_tolerant))
            
            # Save first strictly successful trial for recovery visualization
            if is_strict and not found_recovery:
                recovery_examples[noise] = (corrupted_state, net.state.copy())
                found_recovery = True
            
            total_steps += steps_taken
            total_energy += final_energy
            
            # Per-trial CSV row
            trial_rows.append({
                "corruption_level": noise,
                "trial": t,
                "strict_success": int(is_strict),
                "inversion_tolerant_success": int(is_tolerant),
                "steps_to_converge": steps_taken,
                "initial_energy": initial_energy,
                "final_energy": final_energy,
            })
        
        # Stats — strict success rate with SEM
        strict_arr = np.array(strict_successes)
        tolerant_arr = np.array(tolerant_successes)
        strict_rate = np.mean(strict_arr)
        tolerant_rate = np.mean(tolerant_arr)
        strict_sem = np.std(strict_arr, ddof=1) / np.sqrt(n_trials)
        avg_steps = total_steps / n_trials
        avg_energy = total_energy / n_trials
        
        results.append({
            "Noise Level": noise,
            "Success Rate": strict_rate,
            "SEM": strict_sem,
            "Tolerant Rate": tolerant_rate,
            "Avg Steps": avg_steps,
            "Avg Energy": avg_energy
        })
        energy_traces[noise] = example_trace
        
        print(f"  Strict Success:    {strict_rate*100:.1f}% (±SEM {strict_sem*100:.1f}%)")
        print(f"  Tolerant Success:  {tolerant_rate*100:.1f}% (pattern or inverse)")
        print(f"  Avg Steps:         {avg_steps:.1f}")
        print(f"  Avg Energy:        {avg_energy:.2f}")

    # Debug: print exact values being plotted
    df = pd.DataFrame(results)
    print("\n--- Debug: Plotting Data ---")
    print(f"  Noise Levels:  {df['Noise Level'].tolist()}")
    print(f"  Success Rates: {df['Success Rate'].tolist()}")
    print(f"  Avg Steps:     {df['Avg Steps'].tolist()}")

    # 4. Summary Plot (3 subplots)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 10))
    
    noise_labels = [f"{n*100:.0f}%" for n in df["Noise Level"]]
    x_pos = np.arange(len(corruption_levels))
    
    # --- Left: Success Rate (bar chart with SEM error bars) ---
    bars = ax1.bar(x_pos, df["Success Rate"], width=0.35, color=['#2ecc71', '#f39c12', '#e74c3c'],
                   edgecolor='black', yerr=df["SEM"], capsize=5, label='Strict (pattern only)')
    # Inversion-tolerant bars offset
    bars2 = ax1.bar(x_pos + 0.35, df["Tolerant Rate"], width=0.35,
                    color=['#2ecc71', '#f39c12', '#e74c3c'], edgecolor='black', alpha=0.4,
                    label='Tolerant (pattern or inverse)')
    ax1.set_xticks(x_pos + 0.175)
    ax1.set_xticklabels(noise_labels)
    ax1.set_title("Success Rate vs Corruption", fontsize=12)
    ax1.set_xlabel("Corruption Rate")
    ax1.set_ylabel("Success Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='50% baseline')
    ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='80% target')
    ax1.legend(fontsize=7)
    # Add value labels on bars
    for bar, val in zip(bars, df["Success Rate"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val*100:.0f}%', ha='center', va='bottom', fontweight='bold', fontsize=8)
    for bar, val in zip(bars2, df["Tolerant Rate"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val*100:.0f}%', ha='center', va='bottom', fontsize=7, alpha=0.7)
    
    # --- Middle: Convergence Speed (line chart with labels) ---
    ax2.plot(df["Noise Level"], df["Avg Steps"], marker='s', color='#3498db', linewidth=2, markersize=8)
    ax2.set_title("Convergence Speed", fontsize=12)
    ax2.set_xlabel("Corruption Rate")
    ax2.set_ylabel("Avg Epochs to Converge")
    ax2.grid(True, alpha=0.3)
    # Add data point labels
    for x, y in zip(df["Noise Level"], df["Avg Steps"]):
        ax2.annotate(f'{y:.1f}', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontweight='bold')
    
    # --- Right: Energy Descent ---
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for i, noise in enumerate(corruption_levels):
        trace = energy_traces[noise]
        steps = [s for s, e in trace]
        energies = [e for s, e in trace]
        ax3.plot(steps, energies, marker='o', markersize=3, color=colors[i],
                 linewidth=1.5, label=f'{noise*100:.0f}% corruption')
    ax3.set_title("Energy Descent (Example Trial)", fontsize=12)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Energy E")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle("Hopfield Network — Single Pattern Robustness (N=100)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to both output paths
    output_paths = [
        "results/hopfield/exp1_single_pattern_results.png",
        "experiments/results/figures/exp1_single_pattern_results.png",
    ]
    for path in output_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {path}")
    plt.close()
    
    # 5. Recovery Visualization
    n_levels = len(corruption_levels)
    fig, axes = plt.subplots(3, n_levels, figsize=(3 * n_levels + 1, 9))
    row_labels = ["Original", "Corrupted", "Recalled"]
    
    for col, noise in enumerate(corruption_levels):
        # Original pattern (same for all columns)
        axes[0, col].imshow(target_pattern.reshape(L_grid, L_grid), cmap='binary', vmin=-1, vmax=1)
        axes[0, col].set_title(f"{noise*100:.0f}% noise", fontsize=10)
        axes[0, col].axis('off')
        
        if noise in recovery_examples:
            corrupted, recalled = recovery_examples[noise]
            axes[1, col].imshow(corrupted.reshape(L_grid, L_grid), cmap='binary', vmin=-1, vmax=1)
            axes[2, col].imshow(recalled.reshape(L_grid, L_grid), cmap='binary', vmin=-1, vmax=1)
        else:
            # No successful trial found — show blank with note
            for row in [1, 2]:
                axes[row, col].text(0.5, 0.5, 'No recovery', ha='center', va='center',
                                    transform=axes[row, col].transAxes, fontsize=9, color='red')
                axes[row, col].axis('off')
        
        for row in [1, 2]:
            axes[row, col].axis('off')
    
    # Row labels on left
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold')
    
    fig.suptitle("Hopfield Network — Pattern Recovery Examples", fontsize=13, fontweight='bold')
    plt.tight_layout()
    recovery_path = "results/hopfield/exp1_recovery_examples.png"
    os.makedirs(os.path.dirname(recovery_path), exist_ok=True)
    plt.savefig(recovery_path, dpi=150, bbox_inches='tight')
    print(f"Recovery visualization saved to {recovery_path}")
    plt.close()
    
    # 6. Save per-trial CSV
    csv_path = "results/hopfield/exp1_results.csv"
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_csv(csv_path, index=False)
    print(f"Trial data saved to {csv_path}")
    
    # 7. Print Summary Table
    print("\n=== Final Results Table ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_experiment_1()
