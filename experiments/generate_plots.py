import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Ensure we can load the class definition
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
# We need to import the dataclass from run_simulation or define a dummy one if pickle needs it?
# Pickle needs the class definition to be available in the module scope described in the pickle stream.
# Since run_simulation is in experiments/, we can import it.
try:
    from experiments.run_simulation import SimulationResults
except ImportError:
    # If running from root
    from run_simulation import SimulationResults

def setup_plotting_style():
    """Configure matplotlib/seaborn for publication quality."""
    sns.set_style("ticks")
    sns.set_context("paper", font_scale=1.4)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman'],
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 1.5,
        'errorbar.capsize': 3
    })
    
    # Color palette
    colors = sns.color_palette("colorblind")
    return colors

def load_results(path="results/ising_data_partial.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def calculate_errors_and_correlation_length(res: SimulationResults):
    """
    Process raw results to add error bars and correlation lengths.
    Returns a dictionary with enhanced data arrays.
    """
    T = np.array(res.temperatures)
    M = np.array(res.magnetization)
    E = np.array(res.energy)
    Chi = np.array(res.susceptibility)
    Cv = np.array(res.specific_heat)
    
    # Calculate M_std from Chi
    # Chi = beta * N * Var(M) => Var(M) = Chi * T / N
    # std(M) = sqrt(Var(M))
    # Note: res.susceptibility is already normalized per spin? 
    # Check run_simulation.py: Chi = (beta / N) * var_M. 
    # Yes, it is intensive. Wait. Chi is usually extensive ~ N.
    # But in run_simulation: Chi = (beta / N_spins) * var_M. 
    # var_M was computed from Total Magnetization (M_series).
    # So Chi is intensive.
    # Therefore std_m (per spin) calculation:
    # var_m (per spin) = var_M / N^2
    # relation: Chi = (1/T/N) * var_M
    # So var_M = Chi * T * N
    # var_m = Chi * T * N / N^2 = Chi * T / N
    # std_m = sqrt(Chi * T / N)
    
    N = res.lattice_size ** 2
    std_M = np.sqrt(Chi * T / N)
    
    # Calculate std_E from Cv
    # Cv = (1/T^2/N) * var_E
    # var_E = Cv * T^2 * N
    # var_e (per spin) = Cv * T^2 / N
    # std_E = sqrt(Cv * T^2 / N)
    std_E = np.sqrt(Cv * (T**2) / N)
    
    # Calculate Correlation Length xi
    # Formula: xi = -1 / ln( |C(1)/C(0)| )
    # using connected correlations from res.spatial_correlations
    xi_list = []
    
    for t_val in T:
        # spatial_correlations is keyed by float T. 
        # Floating point matching might be tricky, use close match or assume key exists.
        # run_simulation puts specific floats.
        
        # We need to find the key in the dict that is closest to t_val
        closest_T = min(res.spatial_correlations.keys(), key=lambda k: abs(k - t_val))
        c_r = res.spatial_correlations[closest_T]
        
        # C(0) should be variance of spin = 1 - m^2 ? 
        # compute_spatial_correlation returns <si si+r> - <m>^2
        # C(0) = 1 - <m>^2.
        # C(1) = <si si+1> - <m>^2.
        
        c0 = c_r[0]
        c1 = c_r[1]
        
        # Avoid log(0)
        ratio = abs(c1 / c0) if c0 != 0 else 0
        if ratio <= 0 or ratio >= 1:
            # If ratio >= 1, correlation doesn't decay (ordered?) -> xi infinite
            # If ratio <= 0, anti-correlated or noise?
            xi = 0 # Placeholder for invalid
        else:
            xi = -1.0 / np.log(ratio)
            
        xi_list.append(xi)
        
    return {
        "T": T, "M": M, "E": E, "Chi": Chi, "Cv": Cv,
        "std_M": std_M, "std_E": std_E, "Xi": np.array(xi_list),
        "L": res.lattice_size
    }

def plot_figure_1(data_dict, colors):
    """Phase Transition Overview (2x2 subplot)"""
    # Use the largest system size for the overview if multiple available
    L_max = max(data_dict.keys())
    d = data_dict[L_max]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Tc line
    Tc = 2.269
    
    # 1. Magnetization
    ax = axes[0, 0]
    ax.errorbar(d['T'], d['M'], yerr=d['std_M'], fmt='o-', color=colors[0], label=f'L={d["L"]}', markersize=4, alpha=0.8)
    ax.axvline(Tc, color='k', linestyle='--', alpha=0.5, label=r'$T_c$')
    ax.set_ylabel(r'Magnetization $\langle |M| \rangle$')
    ax.set_xlabel(r'Temperature $T$')
    ax.legend()
    
    # 2. Energy
    ax = axes[0, 1]
    ax.errorbar(d['T'], d['E'], yerr=d['std_E'], fmt='o-', color=colors[1], label=f'L={d["L"]}', markersize=4, alpha=0.8)
    ax.axvline(Tc, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel(r'Energy $\langle E \rangle$')
    ax.set_xlabel(r'Temperature $T$')
    
    # 3. Susceptibility
    ax = axes[1, 0]
    ax.plot(d['T'], d['Chi'], 'o-', color=colors[2], label=f'L={d["L"]}')
    ax.axvline(Tc, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel(r'Susceptibility $\chi$')
    ax.set_xlabel(r'Temperature $T$')
    
    # 4. Specific Heat
    ax = axes[1, 1]
    ax.plot(d['T'], d['Cv'], 'o-', color=colors[3], label=f'L={d["L"]}')
    ax.axvline(Tc, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel(r'Specific Heat $C_v$')
    ax.set_xlabel(r'Temperature $T$')
    
    plt.tight_layout()
    plt.savefig('results/figures/Fig1_Transition_Overview.png')
    plt.savefig('results/figures/Fig1_Transition_Overview.pdf')
    plt.close()

def plot_figure_2(data_dict, colors):
    """Critical Point Detail (Zoomed)"""
    L_max = max(data_dict.keys())
    d = data_dict[L_max]
    
    # Filter for zoom range [2.0, 2.5]
    mask = (d['T'] >= 2.0) & (d['T'] <= 2.5)
    T_zoom = d['T'][mask]
    Chi_zoom = d['Chi'][mask]
    Cv_zoom = d['Cv'][mask]
    
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    # Chi on left axis
    line1 = ax1.plot(T_zoom, Chi_zoom, 'o-', color=colors[2], label=r'$\chi$', markersize=6)
    ax1.set_xlabel(r'Temperature $T$')
    ax1.set_ylabel(r'Susceptibility $\chi$', color=colors[2])
    ax1.tick_params(axis='y', labelcolor=colors[2])
    
    # Cv on right axis
    ax2 = ax1.twinx()
    line2 = ax2.plot(T_zoom, Cv_zoom, 's--', color=colors[3], label=r'$C_v$', markersize=6)
    ax2.set_ylabel(r'Specific Heat $C_v$', color=colors[3])
    ax2.tick_params(axis='y', labelcolor=colors[3])
    
    # Annotate Tc
    Tc = 2.269
    ax1.axvline(Tc, color='k', alpha=0.3, linestyle=':')
    ax1.annotate(r'$T_c \approx 2.269$', xy=(Tc, np.max(Chi_zoom)), xytext=(Tc+0.1, np.max(Chi_zoom)),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title(f'Critical Region Detail (L={d["L"]})')
    plt.tight_layout()
    plt.savefig('results/figures/Fig2_Critical_Detail.png')
    plt.savefig('results/figures/Fig2_Critical_Detail.pdf')
    plt.close()

def plot_figure_3(data_dict, colors):
    """Finite-Size Scaling (Magnetization)"""
    plt.figure(figsize=(10, 8))
    
    # Sort Ls
    Ls = sorted(data_dict.keys())
    
    for i, L in enumerate(Ls):
        d = data_dict[L]
        plt.plot(d['T'], d['M'], 'o-', color=colors[i % len(colors)], label=f'L={L}', markersize=4, alpha=0.8)
        
    plt.axvline(2.269, color='k', linestyle='--', alpha=0.5, label=r'$T_c$')
    plt.ylabel(r'Magnetization $\langle |M| \rangle$')
    plt.xlabel(r'Temperature $T$')
    plt.title('Finite-Size Effects on Magnetization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/Fig3_Finite_Size_Scaling.png')
    plt.savefig('results/figures/Fig3_Finite_Size_Scaling.pdf')
    plt.close()

def plot_figure_4(data_dict, colors):
    """Correlation Length vs T"""
    plt.figure(figsize=(10, 8))
    
    Ls = sorted(data_dict.keys())
    
    for i, L in enumerate(Ls):
        d = data_dict[L]
        plt.plot(d['T'], d['Xi'], 'o-', color=colors[i % len(colors)], label=f'L={L}', markersize=4, alpha=0.8)
        
    plt.axvline(2.269, color='k', linestyle='--', alpha=0.5, label=r'$T_c$')
    plt.ylabel(r'Correlation Length $\xi$')
    plt.xlabel(r'Temperature $T$')
    plt.yscale('log')
    plt.title(r'Correlation Length Divergence $\xi(T)$')
    plt.legend()
    plt.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig('results/figures/Fig4_Correlation_Length.png')
    plt.savefig('results/figures/Fig4_Correlation_Length.pdf')
    plt.close()

def main():
    os.makedirs("results/figures", exist_ok=True)
    colors = setup_plotting_style()
    
    # Try loading full data, fallback to partial
    path = "results/ising_data_full.pkl"
    if not os.path.exists(path):
        print("Full data not found, using partial data...")
        path = "results/ising_data_partial.pkl"
        
    raw_results = load_results(path)
    
    # Process all lattice sizes
    data_dict = {}
    for L, res in raw_results.items():
        data_dict[L] = calculate_errors_and_correlation_length(res)
        
    print(f"Loaded data for L = {list(data_dict.keys())}")
    
    print("Generating Figure 1...")
    plot_figure_1(data_dict, colors)
    
    print("Generating Figure 2...")
    plot_figure_2(data_dict, colors)
    
    print("Generating Figure 3...")
    plot_figure_3(data_dict, colors)
    
    print("Generating Figure 4...")
    plot_figure_4(data_dict, colors)
    
    print("Done! Figures saved to results/figures/")

if __name__ == "__main__":
    main()
