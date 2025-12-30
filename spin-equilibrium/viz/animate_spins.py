import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Add path to spin-equilibrium root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.ising_model import IsingSimulation

def create_animation():
    # Configuration
    L = 128
    temps = [1.0, 2.269, 4.0]
    temp_labels = ["Ordered (T=1.0)", "Critical (T=2.269)", "Disordered (T=4.0)"]
    duration_sec = 10
    fps = 30
    total_frames = duration_sec * fps
    steps_per_frame = 4 # Approx 1200 steps total
    
    # Initialize Simulations
    sims = [IsingSimulation(size=L, temperature=T) for T in temps]
    
    # History buffers for plotting
    mag_histories = [[], [], []]
    time_steps = []
    
    # Setup Figure with GridSpec
    # Layout: Top row = 3 Lattice images
    # Bottom row = 3 Magnetization plots
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(2, 3, height_ratios=[3, 1], hspace=0.3)
    
    # Lattice Axes
    img_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # Plot Axes
    plot_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
    
    # Initial Drawings
    images = []
    lines = []
    
    # Common styling
    plt.rcParams.update({'font.size': 10})
    
    for i, (ax, sim, label) in enumerate(zip(img_axes, sims, temp_labels)):
        # Image
        # +1 -> White (255), -1 -> Black (0). 
        # Using binary cmap
        im = ax.imshow(sim.grid, cmap='gray', vmin=-1, vmax=1, interpolation='nearest')
        images.append(im)
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Plot setup
        p_ax = plot_axes[i]
        p_ax.set_xlim(0, total_frames * steps_per_frame)
        p_ax.set_ylim(0, 1.1)
        p_ax.set_xlabel("MC Sweeps")
        if i == 0:
            p_ax.set_ylabel("Magnetization |M|")
        p_ax.grid(True, alpha=0.3)
        
        line, = p_ax.plot([], [], lw=2, color='blue')
        lines.append(line)
        
    # Global Title and Step Counter
    fig.suptitle("2D Ising Model Evolution", fontsize=20)
    step_text = fig.text(0.5, 0.92, "Steps: 0", ha='center', fontsize=14)
    
    def init():
        return images + lines + [step_text]
        
    def update(frame):
        current_step = frame * steps_per_frame
        
        # Update Simulations
        for i, sim in enumerate(sims):
            sim.metropolis_step(steps_per_sweep=steps_per_frame)
            
            # Update grid image
            images[i].set_array(sim.grid)
            
            # Update plotting data
            mag = abs(sim.magnetization) / (L * L)
            mag_histories[i].append(mag)
            
            # Update Line
            # We construct logical x-axis based on appended history
            x_data = np.arange(len(mag_histories[i])) * steps_per_frame
            lines[i].set_data(x_data, mag_histories[i])
            
            # Show current M value in title or near plot?
            # Let's update title? No, expensive. 
            # Maybe text annotation on plot
            # Simple approach: Plot title
            plot_axes[i].set_title(f"|M| = {mag:.3f}", fontsize=10)
            
        step_text.set_text(f"Monte Carlo Sweeps: {current_step}")
        time_steps.append(current_step)
        
        return images + lines + [step_text] + [ax.title for ax in plot_axes]

    print("Generating animation...")
    anim = animation.FuncAnimation(
        fig, update, init_func=init, 
        frames=total_frames, interval=1000/fps, blit=False
    )
    
    # Save Output
    os.makedirs("results/animations", exist_ok=True)
    
    # Save as GIF (Available everywhere)
    print("Saving as GIF (ising_evolution.gif)...")
    anim.save('results/animations/ising_evolution.gif', writer='pillow', fps=fps)
    
    # Try saving as MP4 (Requires ffmpeg)
    # We check if ffmpeg is available
    import shutil
    if shutil.which("ffmpeg"):
        print("Saving as MP4 (ising_evolution.mp4)...")
        try:
            anim.save('results/animations/ising_evolution.mp4', writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
        except Exception as e:
            print(f"MP4 save failed: {e}")
    else:
        print("FFmpeg not found, skipping MP4.")
        
    print("Done!")

if __name__ == "__main__":
    create_animation()
