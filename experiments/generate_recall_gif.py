import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ising_simulation.hopfield.core import HopfieldNetwork
from ising_simulation.hopfield.utils import create_letter_pattern, corrupt_pattern

def main():
    print("Generating Hopfield Recall Animation...")
    # Setup network
    L_grid = 10
    N = L_grid * L_grid
    net = HopfieldNetwork(n_neurons=N)
    
    # Target pattern
    target_pattern = create_letter_pattern('N', size=L_grid)
    net.train([target_pattern])
    
    # Corrupt
    noise = 0.30
    noisy_input = corrupt_pattern(target_pattern, corruption_rate=noise)
    
    # === FIX 1: Diagnostic prints BEFORE the animation loop ===
    print(f"Energy of stored pattern N: {net.compute_energy(target_pattern):.2f}")
    print(f"Should be approximately -49.3")
    
    n_flipped = np.sum(noisy_input != target_pattern)
    print(f"Pixels flipped during corruption: {n_flipped} (expected 30 for 30%)")
    
    initial_energy = net.compute_energy(noisy_input)
    print(f"Initial energy after corruption: {initial_energy:.2f}")
    print(f"Should be approximately -25 to -30")
    
    # Prepare for simulation
    net.state = noisy_input.flatten().copy()
    
    states = [net.state.copy()]
    energies = [net.energy()]
    
    # === FIX 2: Increase num_updates to enable full convergence ===
    # 2000 single-neuron updates = 20 full sweeps of 100 neurons
    num_updates = 2000
    stored_energy = net.compute_energy(target_pattern)
    
    for _ in range(num_updates):
        net.update(net.state, mode='async')
        states.append(net.state.copy())
        energies.append(net.energy())
    
    # === FIX 3: Verify convergence ===
    final_energy = energies[-1]
    print(f"Final energy: {final_energy:.2f}")
    print(f"Stored pattern energy: {stored_energy:.2f}")
    if abs(final_energy - stored_energy) > 2.0:
        print("WARNING: Recall did NOT converge to stored pattern!")
        print("Check the recall logic before saving animation.")
        return  # DO NOT save the GIF
    
    # === FIX 4: Subsample frames to keep GIF reasonable size ===
    # Pick ~80 frames: every 25th step from 2000 updates
    subsample_step = max(1, num_updates // 80)
    step_indices = list(range(0, num_updates + 1, subsample_step))
    states = [states[i] for i in step_indices]
    energies = [energies[i] for i in step_indices]
    
    # Setup Figure
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle("Hopfield Network Recall: 30% Corruption \u2192 Pattern N", fontsize=14)
    
    # Left subplot: Grid
    ax1 = plt.subplot(1, 2, 1)
    im = ax1.imshow(states[0].reshape(L_grid, L_grid), cmap='binary', vmin=-1, vmax=1)
    title1 = ax1.set_title(f"Step 0 \u2014 Energy: {energies[0]:.1f}")
    ax1.axis('off')
    
    # Right subplot: Energy curve
    ax2 = plt.subplot(1, 2, 2)
    line, = ax2.plot([], [], 'r-', lw=2)
    ax2.set_xlim(0, step_indices[-1])
    
    # Give some padding to ylim based on energy range
    min_e = min(energies)
    max_e = max(energies)
    # Handle case where min_e == max_e
    if min_e == max_e:
        ax2.set_ylim(min_e - 5, max_e + 5)
    else:
        padding = (max_e - min_e) * 0.1
        ax2.set_ylim(min_e - padding, max_e + padding)
        
    ax2.set_title("Energy E")
    ax2.set_xlabel("Step (Single Neuron Update)")
    ax2.set_ylabel("Energy")
    ax2.grid(True)
    
    def init():
        im.set_data(states[0].reshape(L_grid, L_grid))
        title1.set_text(f"Step 0 \u2014 Energy: {energies[0]:.1f}")
        line.set_data([], [])
        return [im, title1, line]
        
    def update(frame):
        im.set_data(states[frame].reshape(L_grid, L_grid))
        title1.set_text(f"Step {step_indices[frame]} \u2014 Energy: {energies[frame]:.1f}")
        line.set_data(step_indices[:frame+1], energies[:frame+1])
        return [im, title1, line]

    anim = FuncAnimation(fig, update, frames=len(states), init_func=init, blit=True, repeat=True)
    
    # Save outputs
    os.makedirs('results/hopfield', exist_ok=True)
    
    # Save GIF
    gif_path = 'results/hopfield/recall_animation.gif'
    print(f"Saving GIF to {gif_path} ...")
    anim.save(gif_path, writer=PillowWriter(fps=8))
    print("GIF saved.")
    
    # Save MP4
    mp4_path = 'results/hopfield/recall_animation.mp4'
    print(f"Saving MP4 to {mp4_path} ...")
    try:
        anim.save(mp4_path, writer=FFMpegWriter(fps=8))
        print("MP4 saved.")
    except Exception as e:
        print(f"Could not save MP4 (FFmpeg might not be installed or configured): {e}")

if __name__ == "__main__":
    main()
