import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import seaborn as sns

# Add path to spin-equilibrium root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

def run_hysteresis_sweep(L, T, steps_equil=500, steps_meas=2000):
    """
    Run a full Hysteresis loop for a given Temperature.
    Protocol:
    1. Init all spins +1
    2. B: 2.0 -> -2.0 (40 steps)
    3. B: -2.0 -> 2.0 (40 steps)
    """
    # Initialize model with all spins up
    sim = IsingSimulation(size=L, temperature=T, B=2.0)
    sim.grid = np.ones((L, L), dtype=np.int8)
    
    # Define Field Sweep
    # 40 points from 2 to -2
    b_down = np.linspace(2.0, -2.0, 40)
    # 40 points from -2 to 2
    b_up = np.linspace(-2.0, 2.0, 40)
    
    # Concatenate but careful with potential overlap at -2.0
    # b_up starts at -2.0, same as b_down end. 
    # Usually we just run them sequentially.
    
    b_values = np.concatenate([b_down, b_up[1:]]) # Avoid double counting turning point
    
    m_values = []
    b_history = []
    
    N = L * L
    
    for B in tqdm(b_values, desc=f"Hysteresis T={T}", leave=False):
        sim.B = B
        
        # Equilibrate
        sim.metropolis_step(steps_per_sweep=steps_equil)
        
        # Measure
        # We can average over `steps_meas`
        # For efficiency, we can sample every 10 steps
        measurements = []
        for _ in range(steps_meas // 10):
            sim.metropolis_step(steps_per_sweep=10)
            measurements.append(sim.magnetization)
            
        avg_m = np.mean(measurements) / N
        m_values.append(avg_m)
        b_history.append(B)
        
    return np.array(b_history), np.array(m_values)

def analyze_hysteresis(b_vals, m_vals):
    """
    Calculate Coercivity (Bc), Remanence (Mr), and Loop Area.
    Assumes b_vals goes +2 -> -2 -> +2
    """
    # Split into down and up branches
    # Turn point is the minimum B (approx index len/2)
    turn_idx = np.argmin(b_vals)
    
    b_down = b_vals[:turn_idx+1]
    m_down = m_vals[:turn_idx+1]
    
    b_up = b_vals[turn_idx:]
    m_up = m_vals[turn_idx:]
    
    # Remanence Mr: |M| at B=0
    # We find M at B=0 for both branches and average magnitude or take top branch
    # Mr is usually defined from the Zero-Field crossing coming from Saturation.
    # So on b_down (2 -> -2), finding B=0 crossing gives +Mr
    # On b_up (-2 -> 2), finding B=0 crossing gives -Mr
    
    mr_down = interp1d(b_down, m_down)(0.0)
    mr_up = interp1d(b_up, m_up)(0.0)
    Mr = (abs(mr_down) + abs(mr_up)) / 2.0
    
    # Coercivity Bc: |B| at M=0
    # On b_down, M goes + to -. Find M=0.
    bc_down = 0.0
    try:
        bc_down = interp1d(m_down, b_down)(0.0)
    except:
        pass # M might not cross 0 if T is high (always 0) or always magnetized (impossible)
        
    bc_up = 0.0
    try:
        bc_up = interp1d(m_up, b_up)(0.0)
    except:
        pass
        
    Bc = (abs(bc_down) + abs(bc_up)) / 2.0
    
    # Loop Area: Integral M dB
    # Area = Integration using trapezoidal rule
    # Since the curve is closed (roughly), we can integrate over the full cycle
    # Area = Integral(M dB) (signed area)
    # Correct orientation (- loop) gives positive area usually
    area = -np.trapz(m_vals, b_vals)
    
    return Bc, Mr, abs(area)

def plot_hysteresis(results_dict):
    """
    Plot M vs B for all temperatures.
    """
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("deep", len(results_dict))
    
    summary_stats = []
    
    for i, (T, (b, m)) in enumerate(results_dict.items()):
        Bc, Mr, Area = analyze_hysteresis(b, m)
        summary_stats.append({"T": T, "Bc": Bc, "Mr": Mr, "Area": Area})
        
        label = f"T={T:.1f}\n$B_c$={Bc:.2f}, $M_r$={Mr:.2f}"
        ax.plot(b, m, label=label, color=colors[i], linewidth=2)
        
        # Add Arrows
        # Add arrow on down curve
        mid_down = len(b)//4
        ax.annotate('', xy=(b[mid_down-1], m[mid_down-1]), xytext=(b[mid_down+1], m[mid_down+1]),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))
        
        # Add arrow on up curve
        mid_up = 3*len(b)//4
        ax.annotate('', xy=(b[mid_up+1], m[mid_up+1]), xytext=(b[mid_up-1], m[mid_up-1]),
                    arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5))

    ax.set_xlabel("External Field $B$")
    ax.set_ylabel("Magnetization $M$")
    ax.set_title("Hysteresis Loops: Magnetic Memory vs Temperature")
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(title="Conditions", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Inset showing parameter decay? Or just text?
    # Let's add a small inset or just let the legend handle it.
    
    plt.tight_layout()
    plt.savefig("results/figures/hysteresis_loops.png", dpi=300)
    plt.savefig("results/figures/hysteresis_loops.pdf", dpi=300)
    print("Plot saved to results/figures/hysteresis_loops.png")
    
    # Save statistics
    df = pd.DataFrame(summary_stats)
    df.to_csv("results/hysteresis_stats.csv", index=False)
    print("Stats saved to results/hysteresis_stats.csv")
    print(df)

def main():
    L = 64
    temps = [1.0, 1.5, 2.0, 2.5]
    
    print(f"Starting Hysteresis Experiment (L={L}) for T={temps}")
    
    results = {}
    
    for T in temps:
        b, m = run_hysteresis_sweep(L, T)
        results[T] = (b, m)
        
    os.makedirs("results/figures", exist_ok=True)
    plot_hysteresis(results)

if __name__ == "__main__":
    main()
