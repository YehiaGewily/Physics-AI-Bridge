
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# spin-equilibrium has a hyphen (not importable as module), add its path directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))

from core.ising_model import IsingSimulation
from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern, corrupt_pattern


def run_comparison():
    print("=== Ising ↔ Hopfield Comparison ===")

    L_grid = 10
    N_hopfield = L_grid * L_grid

    # =========================================================
    # ROW 1: Ising Model
    # =========================================================

    # Panel (0,0): Ordered spin lattice at T=1.5
    print("Running Ising simulation at T=1.5...")
    ising_ordered = IsingSimulation(size=32, J=1.0, temperature=1.5)
    # Equilibrate
    ising_ordered.metropolis_step(steps_per_sweep=500)
    grid_ordered = ising_ordered.grid.copy()

    # Panel (0,1) & (0,2): Energy & magnetization descent from hot start
    print("Running Ising equilibration from hot start...")
    ising_hot = IsingSimulation(size=32, J=1.0, temperature=1.5)
    # Start from random (already random by default)

    n_sweeps = 200
    ising_energies = []
    ising_magnetizations = []
    n_spins = 32 * 32

    for s in range(n_sweeps):
        ising_hot.metropolis_step(steps_per_sweep=1)
        ising_energies.append(ising_hot.energy() / n_spins)
        ising_magnetizations.append(abs(ising_hot.magnetization) / n_spins)

    # Equilibrium magnetization (approximate for T=1.5)
    eq_mag = np.mean(ising_magnetizations[-50:])

    # =========================================================
    # ROW 2: Hopfield Network
    # =========================================================

    print("Running Hopfield recall from corrupted input...")
    pattern_N = create_letter_pattern('N', L_grid)
    net = HopfieldNetwork(n_neurons=N_hopfield)
    net.train([pattern_N])

    # Panel (1,0): Stored pattern N as spin grid
    pattern_grid = pattern_N.reshape(L_grid, L_grid)

    # Panel (1,1) & (1,2): Energy & similarity during recall
    corruption_rate = 0.4
    noisy_input = corrupt_pattern(pattern_N, corruption_rate=corruption_rate)

    net.state = noisy_input.copy()
    max_steps = 50

    hopfield_energies = []
    hopfield_similarities = []

    # Initial state
    hopfield_energies.append(net.energy())
    hamming = np.sum(net.state != pattern_N) / N_hopfield
    hopfield_similarities.append(1.0 - hamming)

    prev_state = net.state.copy()
    for step in range(max_steps):
        # 1 epoch = N async updates
        for _ in range(N_hopfield):
            net.update(net.state, mode='async')

        hopfield_energies.append(net.energy())
        hamming = np.sum(net.state != pattern_N) / N_hopfield
        hopfield_similarities.append(1.0 - hamming)

        if np.array_equal(net.state, prev_state):
            print(f"  Converged at step {step + 1}")
            break
        prev_state = net.state.copy()

    # =========================================================
    # PLOTTING: 2x3 figure
    # =========================================================
    print("Generating comparison figure...")

    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(2, 3, hspace=0.40, wspace=0.30,
                  left=0.06, right=0.96, top=0.88, bottom=0.06)

    # --- Row labels ---
    fig.text(0.01, 0.72, 'ISING\nMODEL', fontsize=14, fontweight='bold',
             ha='center', va='center', rotation=90, color='#2c3e50')
    fig.text(0.01, 0.28, 'HOPFIELD\nNETWORK', fontsize=14, fontweight='bold',
             ha='center', va='center', rotation=90, color='#2c3e50')

    # ===================== ROW 1: Ising =====================

    # Panel (0,0): Spin lattice
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.imshow(grid_ordered, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
    ax00.set_title('Ordered Phase (T=1.5 < Tc)', fontsize=11, fontweight='bold')
    ax00.set_xlabel('x')
    ax00.set_ylabel('y')
    ax00.text(0.02, 0.98, r'$\sigma_i \in \{\pm 1\}$',
              transform=ax00.transAxes, fontsize=10, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel (0,1): Energy descent
    ax01 = fig.add_subplot(gs[0, 1])
    ax01.plot(range(n_sweeps), ising_energies, color='#e74c3c', linewidth=1.5)
    ax01.set_title('Ising: Energy During Equilibration', fontsize=11, fontweight='bold')
    ax01.set_xlabel('Monte Carlo Sweeps')
    ax01.set_ylabel('Energy per Spin')
    ax01.grid(True, alpha=0.3)

    # Panel (0,2): Magnetization convergence
    ax02 = fig.add_subplot(gs[0, 2])
    ax02.plot(range(n_sweeps), ising_magnetizations, color='#3498db', linewidth=1.5)
    ax02.axhline(y=eq_mag, color='gray', linestyle='--', linewidth=1,
                 label=f'Equilibrium |M| ≈ {eq_mag:.2f}')
    ax02.set_title('Order Parameter: |M| vs Time', fontsize=11, fontweight='bold')
    ax02.set_xlabel('Monte Carlo Sweeps')
    ax02.set_ylabel('|M|')
    ax02.legend(fontsize=8)
    ax02.grid(True, alpha=0.3)

    # ===================== ROW 2: Hopfield =====================

    # Panel (1,0): Stored pattern as spin grid
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.imshow(pattern_grid, cmap='binary', vmin=-1, vmax=1, interpolation='nearest')
    ax10.set_title('Stored Memory Pattern N', fontsize=11, fontweight='bold')
    ax10.set_xlabel('x')
    ax10.set_ylabel('y')
    ax10.text(0.02, 0.98, r'$s_i \in \{\pm 1\}$',
              transform=ax10.transAxes, fontsize=10, va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel (1,1): Energy descent during recall
    ax11 = fig.add_subplot(gs[1, 1])
    ax11.plot(range(len(hopfield_energies)), hopfield_energies,
              color='#2ecc71', linewidth=1.5, marker='o', markersize=3)
    ax11.set_title('Hopfield: Energy During Recall', fontsize=11, fontweight='bold')
    ax11.set_xlabel('Recall Steps')
    ax11.set_ylabel('Energy E')
    ax11.grid(True, alpha=0.3)

    # Panel (1,2): Pattern similarity convergence
    ax12 = fig.add_subplot(gs[1, 2])
    ax12.plot(range(len(hopfield_similarities)), hopfield_similarities,
              color='#9b59b6', linewidth=1.5, marker='o', markersize=3)
    ax12.axhline(y=1.0, color='gray', linestyle='--', linewidth=1,
                 label='Perfect recall')
    ax12.set_title('Pattern Similarity vs Time', fontsize=11, fontweight='bold')
    ax12.set_xlabel('Recall Steps')
    ax12.set_ylabel('Similarity (1 − Hamming/N)')
    ax12.set_ylim(-0.05, 1.1)
    ax12.legend(fontsize=8)
    ax12.grid(True, alpha=0.3)

    # --- Global title ---
    fig.suptitle('The Ising–Hopfield Isomorphism', fontsize=16, fontweight='bold', y=0.97)

    # --- Central annotation box ---
    fig.text(0.52, 0.47,
             r'Same Math:  $E = -\sum J_{ij}\, \sigma_i \sigma_j$'
             r'  $\longleftrightarrow$'
             r'  $E = -\frac{1}{2}\sum W_{ij}\, s_i s_j$',
             fontsize=12, ha='center', va='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='#2c3e50', linewidth=2))

    output_path = "results/hopfield/comparison_ising_hopfield.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    run_comparison()
