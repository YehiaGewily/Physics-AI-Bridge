import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Add spin-equilibrium to path explicitly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

def run_experiment():
    # Parameters
    GRID_SIZE = 64  # Smaller grid for faster data collection
    SWEEPS = 1000   # Sweeps to reach equilibrium
    TEMPS = np.linspace(1.0, 4.0, 20)
    
    # Results storage
    results = {
        "T": [],
        "E": [],
        "M": [],
        "Cv": [],
        "Chi": []
    }
    
    print(f"Starting Experiment: {len(TEMPS)} temperatures, {SWEEPS} sweeps each.")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")

    output_dir_snapshots = os.path.join("results", "snapshots")
    os.makedirs(output_dir_snapshots, exist_ok=True)

    for T in tqdm(TEMPS, desc="Simulating"):
        # Initialize model
        model = IsingSimulation(size=GRID_SIZE, temperature=T)
        
        # Equilibrate
        model.metropolis_step(steps_per_sweep=SWEEPS)
        
        # Collect Stats
        stats = model.get_statistics()
        
        # Normalize computations
        # Cv = Var(E)/T^2. The code calculates Var(E) over the history.
        # We need to ensure the history is populated. 
        # The model automatically tracks history in get_statistics logic if we stepped.
        # However, we only stepped 1000 times.
        # We should check if the history buffer is full or sufficient. 
        # IsingSimulation keeps last 1000 steps.
        
        N = GRID_SIZE * GRID_SIZE
        
        # Store intensive properties (per spin) for E and M
        results["T"].append(T)
        results["E"].append(stats["mean_E"] / N)
        results["M"].append(np.abs(stats["mean_M"]) / N) # Use absolute magnetization
        results["Cv"].append(stats["Cv"] / N)
        results["Chi"].append(stats["Chi"] / N)
        
        # Save Snapshot
        # Map -1 to 0, 1 to 1 for generic image saving
        plt.imsave(
            os.path.join(output_dir_snapshots, f"grid_T_{T:.2f}.png"), 
            model.grid, 
            cmap='coolwarm'
        )

    # Generate Plots
    print("Generating Plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Ising Model Thermodynamics ({GRID_SIZE}x{GRID_SIZE} Grid)", fontsize=16)
    
    # E vs T
    axes[0, 0].plot(results["T"], results["E"], 'o-', color='cyan')
    axes[0, 0].set_title("Average Energy vs Temperature")
    axes[0, 0].set_ylabel("Energy / Spin")
    axes[0, 0].set_xlabel("Temperature (T)")
    axes[0, 0].grid(True)

    # M vs T
    axes[0, 1].plot(results["T"], results["M"], 'o-', color='magenta')
    axes[0, 1].set_title("Magnetization vs Temperature")
    axes[0, 1].set_ylabel("|M| / Spin")
    axes[0, 1].set_xlabel("Temperature (T)")
    axes[0, 1].grid(True)
    
    # Cv vs T
    axes[1, 0].plot(results["T"], results["Cv"], 'o-', color='orange')
    axes[1, 0].set_title("Specific Heat (Cv) vs Temperature")
    axes[1, 0].set_ylabel("Cv / Spin")
    axes[1, 0].set_xlabel("Temperature (T)")
    axes[1, 0].grid(True)
    
    # Chi vs T
    axes[1, 1].plot(results["T"], results["Chi"], 'o-', color='green')
    axes[1, 1].set_title("Magnetic Susceptibility (Chi) vs Temperature")
    axes[1, 1].set_ylabel("Chi / Spin")
    axes[1, 1].set_xlabel("Temperature (T)")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join("results", "thermodynamic_curves.png"))
    print("Done! Results saved to results/")

if __name__ == "__main__":
    run_experiment()
