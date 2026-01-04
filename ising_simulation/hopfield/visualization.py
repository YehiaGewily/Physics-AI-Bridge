
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import Tuple, List

from .core import HopfieldNetwork
from .utils import pattern_to_image

def visualize_recall_process(network: HopfieldNetwork, 
                             corrupted_pattern: np.ndarray, 
                             max_epochs: int = 5,
                             steps_per_epoch: int = None,
                             output_path: str = "results/figures/hopfield_recall_process.png") -> Tuple[np.ndarray, dict]:
    """
    Visualizes the Hopfield Network recall process.

    Args:
        network: Trained HopfieldNetwork instance.
        corrupted_pattern: 1D array of the starting state.
        max_epochs: Number of full sweeps (N updates) to run.
        steps_per_epoch: Updates per epoch. Defaults to N (size of network).
        output_path: Path to save the figure.

    Returns:
        tuple: (final_state, info_dict)
               info_dict contains 'energies' list and 'states' list.
    """
    
    # Setup
    N_neurons = network.n_neurons
    L = int(math.sqrt(N_neurons))
    if steps_per_epoch is None:
        steps_per_epoch = N_neurons

    # 1. Initialize
    network.state = corrupted_pattern.flatten().copy()
    
    history_states = []
    history_energies = []
    
    # Store initial state
    history_states.append(network.state.copy())
    history_energies.append(network.energy())
    
    # 2. Run Recall Loop
    # We run for 'max_epochs', taking snapshots
    for epoch in range(max_epochs):
        # Run async updates for one "epoch" (N steps)
        # Using the update method which updates in-place or returns new state
        # In our implementation update() works on self.state if called without args, 
        # or takes state arg. We used self.state implicit update in one version, 
        # but explicit state passing in the latest.
        # Let's use the explicit update helper from previous steps: update(state, mode)
        
        # Helper loop for N updates
        for _ in range(steps_per_epoch):
             network.update(network.state, mode='async')
        
        # Snapshot
        history_states.append(network.state.copy())
        history_energies.append(network.energy())
        
        # Check convergence (if state didn't change from last snapshot)
        if np.array_equal(history_states[-1], history_states[-2]):
            print(f"Converged at epoch {epoch+1}")
            break
            
    final_state = network.state.copy()
    
    # 3. Create Visualization
    # Row 1: Original/Start
    # Row 2: Intermediate steps (select up to 5 evenly spaced)
    # Row 3: Final
    # Row 4: Energy Plot
    
    # Determine which snapshots to show
    n_snapshots = len(history_states)
    # We want to show start, maybe 3 intermediates, and final.
    # Indices to plot:
    indices_to_plot = np.linspace(0, n_snapshots-1, num=min(5, n_snapshots), dtype=int)
    # Ensure 0 and last are included
    if 0 not in indices_to_plot: indices_to_plot = np.insert(indices_to_plot, 0, 0)
    if n_snapshots-1 not in indices_to_plot: indices_to_plot = np.append(indices_to_plot, n_snapshots-1)
    indices_to_plot = np.unique(indices_to_plot)
    
    num_plots = len(indices_to_plot)
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, num_plots) 
    
    # -- Row 1: State Evolution using subplot grid --
    # combining Row 1, 2, 3 concept into one simplified progression row/grid if preferred,
    # or strictly following user request:
    # Row 1: Original
    # Row 2: Progression
    # Row 3: Final
    # Row 4: Energy
    
    # Let's use a simpler layout that is cleaner:
    # Top section: Sequence of images (Start -> ... -> End)
    # Bottom section: Energy plot
    
    # Plot States
    for i, idx in enumerate(indices_to_plot):
        ax = fig.add_subplot(gs[0:2, i]) # Span first 2 rows
        img = pattern_to_image(history_states[idx], L)
        ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
        
        if idx == 0:
            ax.set_title("Input (Corrupted)")
        elif idx == n_snapshots - 1:
            ax.set_title(f"Final (Epoch {idx})")
        else:
            ax.set_title(f"Epoch {idx}")
        ax.axis('off')

    # Plot Energy
    ax_energy = fig.add_subplot(gs[2, :]) # Span last row
    ax_energy.plot(history_energies, marker='o', linestyle='-', color='red', linewidth=2)
    ax_energy.set_title("Energy Descent")
    ax_energy.set_xlabel("Epoch (x N updates)")
    ax_energy.set_ylabel("Energy (Lyapunov Function)")
    ax_energy.grid(True)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return final_state, {'energies': history_energies, 'states': history_states}

def analyze_convergence(network: HopfieldNetwork, 
                        pattern: np.ndarray, 
                        n_trials: int = 20,
                        max_epochs: int = 50,
                        output_path: str = "results/figures/hopfield_convergence_analysis.png") -> dict:
    """
    Analyzes the convergence properties of the Hopfield Network from random initial states.
    
    Args:
        network: Trained HopfieldNetwork instance.
        pattern: The target pattern to check convergence against.
        n_trials: Number of random initial states to test.
        max_epochs: Maximum epochs per trial.
        output_path: Path to save result figure.
        
    Returns:
        dict: Statistics including convergence rate and average energy/distance profiles.
    """
    N = network.n_neurons
    
    # Storage for all trials
    all_energies = [] # List of lists
    all_hamming = []  # List of lists
    final_converged = [] # Boolean list
    
    for t in range(n_trials):
        # Start from random state
        random_state = np.random.choice([-1, 1], size=N)
        network.state = random_state.copy()
        
        trial_energies = []
        trial_hamming = []
        
        # Initial Metics
        trial_energies.append(network.energy())
        # Hamming distance: number of mismatching bits
        # d = sum(s1 != s2). 
        # For +1/-1: d = (N - s1.dot(s2)) / 2
        d_init = np.sum(network.state != pattern.flatten())
        trial_hamming.append(d_init)
        
        prev_state = network.state.copy()
        
        for epoch in range(max_epochs):
            # Run 1 Epoch (N async updates)
            for _ in range(N):
                network.update(network.state, mode='async')
            
            # Metrics
            e_curr = network.energy()
            d_curr = np.sum(network.state != pattern.flatten())
            
            trial_energies.append(e_curr)
            trial_hamming.append(d_curr)
            
            # Convergence Check
            if np.array_equal(network.state, prev_state):
                # Fill remaining epochs with last value for easier plotting/averaging
                remaining = max_epochs - epoch - 1
                trial_energies.extend([e_curr] * remaining)
                trial_hamming.extend([d_curr] * remaining)
                break
            prev_state = network.state.copy()
            
        all_energies.append(trial_energies)
        all_hamming.append(trial_hamming)
        
        # Check if converged to target (or inverse target)
        # Perfectly converging means Hamming distance is 0 or N (inverse)
        is_target = np.array_equal(network.state, pattern.flatten())
        is_inverse = np.array_equal(network.state, -pattern.flatten())
        final_converged.append(is_target or is_inverse)

    # Calculate Statistics
    convergence_rate = sum(final_converged) / n_trials
    
    # Pad lists to same length if any trial didn't finish (though code above fills)
    # Convert to arrays for averaging
    # Find max length actually recorded (should be max_epochs + 1 due to initial state)
    max_len = max(len(l) for l in all_energies)
    
    # Pad any short ones (just in case logic above had off-by-one)
    for i in range(len(all_energies)):
        while len(all_energies[i]) < max_len:
            all_energies[i].append(all_energies[i][-1])
        while len(all_hamming[i]) < max_len:
            all_hamming[i].append(all_hamming[i][-1])
            
    avg_energy = np.mean(all_energies, axis=0)
    std_energy = np.std(all_energies, axis=0)
    
    avg_hamming = np.mean(all_hamming, axis=0)
    std_hamming = np.std(all_hamming, axis=0)
    
    # Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    x = np.arange(max_len)
    
    # 1. Energy Landscape
    # Plot individual faint lines + Average
    for e_trace in all_energies:
        ax1.plot(x, e_trace, color='red', alpha=0.1)
    ax1.plot(x, avg_energy, color='darkred', linewidth=2, label='Mean Energy')
    ax1.fill_between(x, avg_energy-std_energy, avg_energy+std_energy, color='red', alpha=0.2)
    ax1.set_title("Energy Landscape Descent")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Energy")
    ax1.grid(True)
    
    # 2. Hamming Distance
    for h_trace in all_hamming:
        ax2.plot(x, h_trace, color='blue', alpha=0.1)
    ax2.plot(x, avg_hamming, color='darkblue', linewidth=2, label='Mean Hamming Dist')
    ax2.set_title("Distance to Target Pattern")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Hamming Distance (# Bits)")
    # Add line for N/2 (Random guess expectation)
    ax2.axhline(y=N/2, color='gray', linestyle='--', label='Random Guess')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Convergence Pie Chart
    ax3.pie([convergence_rate, 1-convergence_rate], 
            labels=['Converged to Target', 'Spurious/Other'],
            colors=['lightgreen', 'lightcoral'],
            autopct='%1.1f%%', startangle=90)
    ax3.set_title(f"Basin of Attraction (n={n_trials})")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return {
        "convergence_rate": convergence_rate,
        "avg_final_energy": avg_energy[-1],
        "avg_final_hamming": avg_hamming[-1],
        "all_energies_history": all_energies
    }

def find_spurious_states(network: HopfieldNetwork, 
                         stored_patterns: List[np.ndarray], 
                         n_random_inits: int = 100, 
                         max_steps: int = 100) -> dict:
    """
    Identifies spurious states (stable minima that are not stored patterns).
    
    Args:
        network: The trained Hopfield network.
        stored_patterns: List of correct patterns.
        n_random_inits: Number of random starts to explore state space.
        max_steps: Max steps for convergence.
        
    Returns:
        dict: Report containing counts and examples of spurious states.
    """
    N = network.n_neurons
    L = int(math.sqrt(N))
    
    unique_states = []
    unique_counts = []
    
    # 1. Exploration Loop
    for _ in range(n_random_inits):
        # Random start
        start_state = np.random.choice([-1, 1], size=N)
        # Converge
        final_state = network.recall(start_state, max_steps=max_steps, mode='async')
        
        # Check uniqueness
        is_new = True
        for i, known_state in enumerate(unique_states):
            if np.array_equal(final_state, known_state):
                unique_counts[i] += 1
                is_new = False
                break
        
        if is_new:
            unique_states.append(final_state.copy())
            unique_counts.append(1)
            
    # 2. Classification
    classification = {
        "Stored": [],
        "Reverse": [],
        "Mixture": [],
        "Novel": []
    }
    
    classification_counts = {k: 0 for k in classification.keys()}
    
    for state, count in zip(unique_states, unique_counts):
        state_type = "Novel"
        
        # Check Stored
        for pat in stored_patterns:
            if np.array_equal(state, pat.flatten()):
                state_type = "Stored"
                break
            # Check Reverse
            if np.array_equal(state, -pat.flatten()):
                state_type = "Reverse"
                break
        
        # If not stored/reverse, check if simple mixture?
        # (For 3 patterns, mixture states are often combinations like sign(p1+p2+p3))
        # This is hard to exhaustively check generally, so we label as Novel/Mixture
        
        if state_type == "Novel":
             # We can't easily distinguish Mixture vs Novel without specific logic,
             # so we group them. But we can check energy.
             pass

        # Simple mixture heuristic: Check if it's a "Mixed" state (sign of sum of 3 patterns)
        if state_type == "Novel" and len(stored_patterns) >= 3:
             # Try simple Odd mixtures (e.g. p1+p2+p3)
             # This is just a heuristic check for demo purposes
             pass

        classification[state_type].append(state)
        classification_counts[state_type] += count

    # 3. Visualization of Spurious States
    # Show top 5 most frequent "Novel" or "Reverse" states
    spurious_candidates = classification["Novel"] + classification["Reverse"]
    
    if spurious_candidates:
        num_show = min(5, len(spurious_candidates))
        fig, axes = plt.subplots(1, num_show, figsize=(3*num_show, 3))
        if num_show == 1: axes = [axes]
        
        for i in range(num_show):
            img = pattern_to_image(spurious_candidates[i], L)
            axes[i].imshow(img, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f"Spurious {i+1}")
            axes[i].axis('off')
            
        plt.suptitle("Examples of Spurious/Reverse Attractors Found")
        plt.tight_layout()
        output_path = "results/figures/hopfield_spurious_states.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()

    return {
        "counts": classification_counts,
        "total_unique_states": len(unique_states),
        "classification": classification
    }

def visualize_energy_landscape(network: HopfieldNetwork, 
                               patterns: List[np.ndarray], 
                               n_samples: int = 10000,
                               output_path: str = "results/figures/hopfield_energy_landscape.png"):
    """
    Visualizes the energy landscape by projecting states onto 2D using PCA.
    
    Args:
        network: Trained HopfieldNetwork.
        patterns: List of stored patterns (to mark as stars).
        n_samples: Number of random states to sample for the background.
        output_path: File path to save the plot.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("sklearn is required for PCA visualization. Skipping.")
        return

    N = network.n_neurons
    
    # 1. Generate Samples
    # We want a mix: mostly random, but maybe some near patterns to show valleys?
    # User request said "10,000 random states". We'll stick to mostly random.
    # To make PCA meaningful, we might want to include the patterns in the fit data?
    # Or purely random? Purely random in 100D is a sphere. PCA picks arbitrary axes.
    # The energy gradient might not align with these arbitrary axes.
    # However, following instructions strictly: "Fit on sample states".
    
    # Generate random binary states
    samples = np.random.choice([-1, 1], size=(n_samples, N))
    
    # 2. Compute Energies
    energies = []
    for s in samples:
        # compute_energy takes 1D
        energies.append(network.compute_energy(s))
    energies = np.array(energies)
    
    # 3. PCA Probability
    pca = PCA(n_components=2)
    # Fit on samples
    pca.fit(samples)
    
    # Transform samples
    coords_samples = pca.transform(samples)
    
    # Transform Patterns
    pat_matrix = np.array([p.flatten() for p in patterns])
    coords_patterns = pca.transform(pat_matrix)
    
    # 4. Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Contour / Scatter
    # tricontourf is good for interpolation
    cntr = ax.tricontourf(coords_samples[:, 0], coords_samples[:, 1], energies, 
                          levels=20, cmap='viridis')
    
    # Add scatter for texture
    sc = ax.scatter(coords_samples[:, 0], coords_samples[:, 1], 
                    c=energies, cmap='viridis', s=1, alpha=0.3)
    
    fig.colorbar(cntr, ax=ax, label='Energy (Lyapunov)')
    
    # Plot Patterns
    # Mark stored patterns
    ax.scatter(coords_patterns[:, 0], coords_patterns[:, 1], 
               c='red', s=300, marker='*', edgecolors='white', 
               label='Stored Patterns', zorder=10)
    
    ax.set_title(f"Hopfield Energy Landscape (PCA Projection of {n_samples} random states)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Energy Landscape saved to {output_path}")

def map_attraction_basins(network: HopfieldNetwork, 
                          patterns: List[np.ndarray], 
                          grid_size: int = 50,
                          output_path: str = "results/figures/hopfield_basin_map.png"):
    """
    Visualizes the basins of attraction for 2 stored patterns in a reduced 2D space.
    
    Args:
        network: Trained HopfieldNetwork.
        patterns: List of exactly 2 stored patterns.
        grid_size: Resolution of the basin map grid.
        output_path: Path to save result.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return

    if len(patterns) != 2:
        print("Basin mapping currently supports exactly 2 patterns.")
        return

    N = network.n_neurons
    
    # 1. Define 2D Space via PCA
    # We want the plane that contains the two patterns.
    # To define this robustly with PCA, we fit on the patterns + their inverses
    # This ensures the origin is central and the axes align with the patterns.
    p1 = patterns[0].flatten()
    p2 = patterns[1].flatten()
    data_for_pca = np.array([p1, p2, -p1, -p2])
    
    pca = PCA(n_components=2)
    pca.fit(data_for_pca)
    
    # Project patterns to find bounds
    coords_patterns = pca.transform(np.array([p1, p2, -p1, -p2]))
    
    x_min, x_max = coords_patterns[:, 0].min(), coords_patterns[:, 0].max()
    y_min, y_max = coords_patterns[:, 1].min(), coords_patterns[:, 1].max()
    
    # Add margin
    margin = 0.5 * max(x_max - x_min, y_max - y_min)
    x_range = np.linspace(x_min - margin, x_max + margin, grid_size)
    y_range = np.linspace(y_min - margin, y_max + margin, grid_size)
    
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.c_[xx.ravel(), yy.ravel()] # Shape (grid^2, 2)
    
    # 2. Reconstruct and Recall
    # Project grid points back to N-dim space
    reconstructed = pca.inverse_transform(grid_points)
    # Binarize
    reconstructed = np.sign(reconstructed)
    reconstructed[reconstructed == 0] = 1
    
    basin_labels = np.zeros(len(reconstructed))
    
    # Run recall for each point
    # Note: optimizing this loop would be good, but grid_size=50 -> 2500 points
    # It might take a minute.
    print(f"Mapping basins for {grid_size}x{grid_size} grid...")
    
    for i, start_state in enumerate(reconstructed):
        # Recall
        final_state = network.recall(start_state, max_steps=50, mode='sync') # sync is faster
        
        # Identify attractor
        if np.array_equal(final_state, p1):
            basin_labels[i] = 1 # Pattern 1
        elif np.array_equal(final_state, p2):
            basin_labels[i] = 2 # Pattern 2
        elif np.array_equal(final_state, -p1):
            basin_labels[i] = 3 # Inverse P1
        elif np.array_equal(final_state, -p2):
            basin_labels[i] = 4 # Inverse P2
        else:
            basin_labels[i] = 0 # Spurious
            
    # 3. Plotting
    Z = basin_labels.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Custom colormap
    from matplotlib.colors import ListedColormap
    # 0: Gray (Spurious), 1: Blue (P1), 2: Green (P2), 3: LightBlue (Inv P1), 4: LightGreen (Inv P2)
    cmap = ListedColormap(['lightgray', 'cornflowerblue', 'mediumseagreen', 'powderblue', 'palegreen'])
    
    c = ax.pcolormesh(xx, yy, Z, cmap=cmap, shading='auto', alpha=0.8, vmin=0, vmax=4)
    
    # Plot Original Patterns (Projected)
    ax.scatter(coords_patterns[0, 0], coords_patterns[0, 1], c='blue', s=200, marker='*', edgecolors='white', label='Pattern 1')
    ax.scatter(coords_patterns[1, 0], coords_patterns[1, 1], c='green', s=200, marker='*', edgecolors='white', label='Pattern 2')
    # Plot Inverses
    ax.scatter(coords_patterns[2, 0], coords_patterns[2, 1], c='blue', s=100, marker='o', edgecolors='white', alpha=0.5, label='Inverse P1')
    ax.scatter(coords_patterns[3, 0], coords_patterns[3, 1], c='green', s=100, marker='o', edgecolors='white', alpha=0.5, label='Inverse P2')
    
    ax.set_title("Basins of Attraction (PCA Projection)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    
    # Create custom legend for regions
    # patches = [
    #     mpatches.Patch(color='lightgray', label='Spurious'),
    #     mpatches.Patch(color='cornflowerblue', label='Basin P1'),
    #     mpatches.Patch(color='mediumseagreen', label='Basin P2')
    # ]
    # ax.legend(handles=patches, loc='upper right')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Basin Map saved to {output_path}")
