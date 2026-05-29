
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern
from matplotlib.colors import SymLogNorm
from sklearn.decomposition import PCA


def plot_energy_landscape(net, pattern_N, pattern_O, L_grid):
    """Generate the energy landscape scatter plot + energy distribution."""
    N = L_grid * L_grid

    # Generate 5000 random binary states
    n_samples = 5000
    random_states = np.random.choice([-1, 1], size=(n_samples, N))
    print(f"Generated {n_samples} random states")

    # Compute energy for each (vectorized for speed)
    print("Computing energies...")
    SW = random_states @ net.W  # (n_samples, N)
    energies = -0.5 * np.sum(SW * random_states, axis=1)  # (n_samples,)

    # Energies of stored patterns and their inverses
    energy_N = net.compute_energy(pattern_N)
    energy_O = net.compute_energy(pattern_O)
    energy_invN = net.compute_energy(-pattern_N)
    energy_invO = net.compute_energy(-pattern_O)

    # PCA projection
    print("Running PCA projection...")
    pca = PCA(n_components=2)
    all_states = np.vstack([random_states, pattern_N.reshape(1, -1), pattern_O.reshape(1, -1),
                            (-pattern_N).reshape(1, -1), (-pattern_O).reshape(1, -1)])
    projected = pca.fit_transform(all_states)

    proj_random = projected[:n_samples]
    proj_N = projected[n_samples]
    proj_O = projected[n_samples + 1]
    proj_invN = projected[n_samples + 2]
    proj_invO = projected[n_samples + 3]

    explained_var = pca.explained_variance_ratio_
    print(f"PCA explained variance: PC1={explained_var[0]*100:.1f}%, PC2={explained_var[1]*100:.1f}%")

    # --- Terminal Output ---
    print("\n" + "=" * 50)
    print("ENERGY LANDSCAPE RESULTS")
    print("=" * 50)
    print(f"Energy of stored pattern N: {energy_N:.2f}")
    print(f"Energy of stored pattern O: {energy_O:.2f}")
    print(f"Energy of inverted -N:      {energy_invN:.2f}")
    print(f"Energy of inverted -O:      {energy_invO:.2f}")
    print(f"Mean energy of random states: {np.mean(energies):.2f}")
    print(f"Std energy of random states:  {np.std(energies):.2f}")

    frac_lower_N = np.mean(energies < energy_N)
    frac_lower_O = np.mean(energies < energy_O)
    print(f"\nFraction of random states with energy < pattern N ({energy_N:.2f}): {frac_lower_N:.4f}")
    print(f"Fraction of random states with energy < pattern O ({energy_O:.2f}): {frac_lower_O:.4f}")
    print(f"(Should be close to 0 — stored patterns should be near the energy minimum)")

    # --- Plotting ---
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 3, width_ratios=[2, 0.05, 1], wspace=0.48)

    ax1 = fig.add_subplot(gs[0, 0])
    # SymLogNorm to handle wide dynamic range between random states (~-5) and stored patterns (~-50)
    all_energies = np.concatenate([energies, [energy_N, energy_O, energy_invN, energy_invO]])
    norm = SymLogNorm(linthresh=5, vmin=all_energies.min(), vmax=all_energies.max())
    sc = ax1.scatter(proj_random[:, 0], proj_random[:, 1], c=energies,
                     cmap='RdYlBu_r', s=8, alpha=0.6, edgecolors='none', norm=norm)
    ax1.scatter(proj_N[0], proj_N[1], marker='*', s=300, c='gold',
                edgecolors='black', linewidths=1.0, label='Pattern N', zorder=5)
    ax1.scatter(proj_O[0], proj_O[1], marker='*', s=300, c='cyan',
                edgecolors='black', linewidths=1.0, label='Pattern O', zorder=5)
    ax1.scatter(proj_invN[0], proj_invN[1], marker='o', s=200, facecolors='none',
                edgecolors='gold', linewidths=2, label='Inverted -N', zorder=5)
    ax1.scatter(proj_invO[0], proj_invO[1], marker='o', s=200, facecolors='none',
                edgecolors='cyan', linewidths=2, label='Inverted -O', zorder=5)
    ax1.set_xlabel('PCA Component 1', fontsize=11)
    ax1.set_ylabel('PCA Component 2', fontsize=11)
    ax1.set_title('Hopfield Energy Landscape (PCA Projection)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='best')

    cbar_ax = fig.add_subplot(gs[0, 1])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Energy', fontsize=10)

    # Histogram with prominent energy-gap annotation
    ax2 = fig.add_subplot(gs[0, 2])
    counts, _, _ = ax2.hist(energies, bins=40, orientation='horizontal', color='#3498db',
                            alpha=0.55, edgecolor='black', linewidth=0.5,
                            label='Random states')

    stored_energy = min(energy_N, energy_O)
    random_cluster_energy = np.mean(energies)
    max_count = counts.max()
    x_arrow = max_count * 0.72

    ax2.axhline(y=stored_energy, color='black', linestyle='--', linewidth=2.5,
                label=f'Stored attractors (E={stored_energy:.1f})')
    ax2.text(max_count * 0.08, stored_energy + 2.0,
             f'Stored attractors (E={stored_energy:.1f})',
             fontsize=8, fontweight='bold', color='black',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85))

    ax2.annotate('', xy=(x_arrow, stored_energy), xytext=(x_arrow, random_cluster_energy),
                 arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=2.0))
    ax2.text(x_arrow * 1.05, (stored_energy + random_cluster_energy) / 2,
             f'≈{abs(random_cluster_energy - stored_energy):.0f} energy unit gap',
             fontsize=9, color='#c0392b', fontweight='bold', rotation=90,
             va='center', ha='left')

    ax2.set_xlabel('Count', fontsize=11)
    ax2.set_ylabel('')
    ax2.set_title('Energy Distribution', fontsize=12, fontweight='bold', pad=14)
    ax2.legend(fontsize=7, loc='upper right')

    fig.suptitle('Hopfield Network — Energy Landscape (N=100, P=2: N, O)',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.96)

    output_path = "results/hopfield/exp3_energy_landscape.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.close()

    return pca, random_states, energies, proj_random, proj_N, proj_O, proj_invN, proj_invO


def plot_recall_trajectories(net, pattern_N, pattern_O, L_grid, pca, proj_random, energies,
                             proj_N, proj_O, proj_invN, proj_invO):
    """Plot recall trajectories overlaid on the energy landscape."""
    N = L_grid * L_grid
    n_trajectories = 10
    max_steps = 50

    print("\n--- Recall Trajectories ---")

    trajectories = []  # list of (states_list, attractor_type)

    for t in range(n_trajectories):
        # Random initial state
        init_state = np.random.choice([-1, 1], size=N)
        net.state = init_state.copy()

        # Record state at each step
        state_history = [net.state.copy()]
        prev_state = net.state.copy()

        for epoch in range(max_steps):
            # 1 epoch = N async updates
            for _ in range(N):
                net.update(net.state, mode='async')

            state_history.append(net.state.copy())

            if np.array_equal(net.state, prev_state):
                break
            prev_state = net.state.copy()

        # Classify final state
        final = net.state
        threshold = 0.10  # 10% Hamming distance

        if np.sum(final != pattern_N) / N < threshold:
            attractor = 'N'
        elif np.sum(final != pattern_O) / N < threshold:
            attractor = 'O'
        elif np.sum(final != (-pattern_N)) / N < threshold:
            attractor = '-N'
        elif np.sum(final != (-pattern_O)) / N < threshold:
            attractor = '-O'
        else:
            attractor = 'spurious'

        trajectories.append((state_history, attractor))
        print(f"  Trial {t+1}: converged to {attractor} in {len(state_history)-1} steps")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: energy landscape scatter (faded) with SymLogNorm
    all_energies_traj = np.concatenate([energies,
                                         np.array([net.compute_energy(pattern_N),
                                                   net.compute_energy(pattern_O),
                                                   net.compute_energy(-pattern_N),
                                                   net.compute_energy(-pattern_O)])])
    norm_traj = SymLogNorm(linthresh=5, vmin=all_energies_traj.min(), vmax=all_energies_traj.max())
    ax.scatter(proj_random[:, 0], proj_random[:, 1], c=energies,
               cmap='RdYlBu_r', s=4, alpha=0.3, edgecolors='none', norm=norm_traj)

    # Mark stored patterns
    ax.scatter(proj_N[0], proj_N[1], marker='*', s=300, c='gold',
               edgecolors='black', linewidths=1.0, label='Pattern N', zorder=5)
    ax.scatter(proj_O[0], proj_O[1], marker='*', s=300, c='cyan',
               edgecolors='black', linewidths=1.0, label='Pattern O', zorder=5)
    ax.scatter(proj_invN[0], proj_invN[1], marker='o', s=200, facecolors='none',
               edgecolors='gold', linewidths=2, label='Inverted -N', zorder=5)
    ax.scatter(proj_invO[0], proj_invO[1], marker='o', s=200, facecolors='none',
               edgecolors='cyan', linewidths=2, label='Inverted -O', zorder=5)

    # Color/style mapping by attractor type
    style_map = {
        'N':     {'color': 'gold',          'linestyle': '-',  'label': '→ N'},
        'O':     {'color': 'cyan',          'linestyle': '-',  'label': '→ O'},
        '-N':    {'color': 'gold',          'linestyle': '--', 'label': '→ -N'},
        '-O':    {'color': 'cyan',          'linestyle': '--', 'label': '→ -O'},
        'spurious': {'color': 'grey',      'linestyle': '-',  'label': '→ spurious'},
    }

    # Track which labels have been added to legend
    added_labels = set()

    for state_history, attractor in trajectories:
        # Project all states in trajectory to PCA space
        traj_2d = pca.transform(np.array(state_history))
        style = style_map[attractor]

        # Use label only once for legend
        label = style['label'] if style['label'] not in added_labels else None
        added_labels.add(style['label'])

        # Draw arrow path
        ax.plot(traj_2d[:, 0], traj_2d[:, 1],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=1.5, alpha=0.8, label=label)

        # Start point: small circle
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], s=40, color=style['color'],
                   edgecolors='black', linewidths=0.5, zorder=6)

        # End point: large X
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], marker='X', s=120,
                   color=style['color'], edgecolors='black', linewidths=0.5, zorder=6)

        # Draw arrow at midpoint to show direction
        if len(traj_2d) > 2:
            mid = len(traj_2d) // 2
            dx = traj_2d[mid, 0] - traj_2d[mid - 1, 0]
            dy = traj_2d[mid, 1] - traj_2d[mid - 1, 1]
            ax.annotate('', xy=(traj_2d[mid, 0], traj_2d[mid, 1]),
                        xytext=(traj_2d[mid - 1, 0], traj_2d[mid - 1, 1]),
                        arrowprops=dict(arrowstyle='->', color=style['color'], lw=1.5))

    ax.set_xlabel('PCA Component 1', fontsize=11)
    ax.set_ylabel('PCA Component 2', fontsize=11)
    ax.set_title('Recall Trajectories on Energy Landscape\nArrows show state evolution during recall',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='best')

    fig.suptitle('Hopfield Network — Recall Dynamics (N=100, P=2: N, O)', fontsize=14, fontweight='bold')
    fig.subplots_adjust(top=0.88)

    output_path = "results/hopfield/exp3_recall_trajectories.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.close()


def run_experiment_3():
    print("=== Experiment 3: Energy Landscape & Recall Trajectories ===")

    L_grid = 10
    N = L_grid * L_grid

    # Train on 2 patterns: N and O
    pattern_N = create_letter_pattern('N', L_grid)
    pattern_O = create_letter_pattern('O', L_grid)

    net = HopfieldNetwork(n_neurons=N)
    net.train([pattern_N, pattern_O])
    print(f"Network trained on N, O (N={N})")

    # Part 1: Energy landscape
    pca, random_states, energies, proj_random, proj_N, proj_O, proj_invN, proj_invO = \
        plot_energy_landscape(net, pattern_N, pattern_O, L_grid)

    # Part 2: Recall trajectories on landscape
    plot_recall_trajectories(net, pattern_N, pattern_O, L_grid, pca, proj_random, energies,
                             proj_N, proj_O, proj_invN, proj_invO)


if __name__ == "__main__":
    run_experiment_3()
