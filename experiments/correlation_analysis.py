import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
import seaborn as sns

# Add path to spin-equilibrium root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

def compute_correlation_fft(grid):
    """
    Compute full spatial correlation function using FFT.
    C(dx, dy) = < s(x, y) * s(x+dx, y+dy) >
    """
    L = grid.shape[0]
    
    # Grid is s(x, y)
    # FFT
    ft_grid = np.fft.fft2(grid)
    
    # Power spectrum: |FT(s)|^2
    # This corresponds to FT of autocorrelation
    power_spectrum = np.abs(ft_grid)**2
    
    # Inverse FFT to get Autocorrelation
    autocorr = np.fft.ifft2(power_spectrum).real
    
    # Normalize by number of sites (FFT scaling)
    autocorr /= (L * L)
    
    # This gives C(dx, dy) with periodic boundaries wrapped correctly.
    # C[0,0] is variance/self-correlation (always 1 for spins +/- 1)
    
    return autocorr

def radial_average(autocorr):
    """
    Compute radial average C(r) from 2D autocorrelation C(dx, dy).
    """
    L = autocorr.shape[0]
    y, x = np.indices((L, L))
    
    # Handle periodicity for distance calculation
    # Shortest distance on torus
    dx = np.minimum(x, L - x)
    dy = np.minimum(y, L - y)
    
    r = np.sqrt(dx**2 + dy**2)
    
    # Binning by integer radius
    r_int = r.astype(int)
    
    # Max radius to consider is L/2
    max_r = L // 2
    
    tbin = np.bincount(r_int.ravel(), autocorr.ravel())
    nr = np.bincount(r_int.ravel())
    
    radial_profile = tbin[:max_r+1] / nr[:max_r+1]
    
    return np.arange(len(radial_profile)), radial_profile

def measure_correlations(L, temps, sweeps_equil=1000, sweeps_meas=2000, sample_interval=10):
    results = {}
    
    for T in tqdm(temps, desc="Measuring C(r)"):
        sim = IsingSimulation(size=L, temperature=T)
        
        # Equilibrate
        sim.metropolis_step(steps_per_sweep=sweeps_equil)
        
        c_r_accum = None
        m_sq_accum = 0.0
        n_samples = 0
        
        for _ in range(sweeps_meas // sample_interval):
            sim.metropolis_step(steps_per_sweep=sample_interval)
            
            # Use FFT method
            corr_2d = compute_correlation_fft(sim.grid)
            r_vals, c_r = radial_average(corr_2d)
            
            if c_r_accum is None:
                c_r_accum = np.zeros_like(c_r)
            
            c_r_accum += c_r
            m_sq_accum += (np.mean(sim.grid))**2 # Squared magnetization
            n_samples += 1
            
        avg_c_r = c_r_accum / n_samples
        avg_m_sq = m_sq_accum / n_samples
        
        # Connected correlation G(r) = C(r) - <M>^2
        # C(r) from FFT is <s_i s_j>. 
        # So we subtract <M>^2
        connected_c_r = avg_c_r - avg_m_sq
        
        results[T] = (r_vals, connected_c_r)
        
    return results

def exponential_decay(r, xi, A):
    """Fit function: A * exp(-r / xi)"""
    return A * np.exp(-r / xi)

def power_exp_decay(r, xi, A, eta=0.25):
    """Fit function near Tc: A * r^(-eta) * exp(-r / xi)"""
    # Avoid r=0 singularity
    safe_r = np.copy(r)
    safe_r[0] = 1e-10 
    return A * (safe_r**(-eta)) * np.exp(-r / xi)

def extract_correlation_length(r, c_r, T, Tc=2.269):
    """
    Extract xi by fitting.
    if T > Tc: exponential decay.
    if T near Tc: power-law * exponential?
    We'll try simple exponential first for T > Tc.
    Power law dominates exactly at Tc.
    """
    # Fit range: cut off small r (lattice effects) and large r (noise)
    # Range r in [3, L/4] often good
    start_r = 2
    end_r = len(r) // 2 
    
    # Filter valid positive data for log fitting
    mask = (c_r > 1e-5) & (np.arange(len(c_r)) >= start_r) & (np.arange(len(c_r)) <= end_r)
    
    if np.sum(mask) < 4:
        return 0.0, 0.0 # Failed fit
        
    x_fit = r[mask]
    y_fit = c_r[mask]
    
    # Initial guess
    try:
        if abs(T - Tc) < 0.1:
            # Use fixed eta=0.25 (2D Ising)
            popt, _ = curve_fit(lambda x, xi, A: power_exp_decay(x, xi, A, eta=0.25), 
                               x_fit, y_fit, p0=[5.0, 1.0], maxfev=5000)
        else:
            popt, _ = curve_fit(exponential_decay, x_fit, y_fit, p0=[2.0, 1.0], maxfev=5000)
    except:
        return 0.0, 0.0
        
    return popt[0], popt[1] # xi, A

def plot_correlations(L, results, xi_data):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    colors = sns.color_palette("viridis", len(results))
    
    # PLOT 1: C(r) vs r
    plt.figure(figsize=(10, 8))
    for i, (T, (r, cr)) in enumerate(results.items()):
        plt.plot(r, cr, 'o-', color=colors[i], label=f"T={T:.2f}", markersize=4, alpha=0.7)
        # Plot fit if available
        if T in xi_data and xi_data[T] > 0:
            xi = xi_data[T]
            # Verify fit validity visually?
            # plt.plot(r, exponential_decay(r, xi, xi_data[T+"_A"]), '--', color=colors[i])
            pass
            
    plt.yscale('log')
    plt.xlabel(r"Distance $r$")
    plt.ylabel(r"Correlation Function $C(r)$")
    plt.title(f"Spin-Spin Correlations (L={L})")
    plt.xlim(0, L/2)
    plt.ylim(1e-4, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/Fig_C_r_Decay.png", dpi=300)
    plt.close()

    # PLOT 2: Xi vs T
    # Sort data
    ts = sorted([t for t in xi_data.keys() if isinstance(t, float)])
    xis = [xi_data[t] for t in ts]
    
    plt.figure(figsize=(10, 8))
    plt.plot(ts, xis, 's-', color='crimson', markersize=8)
    plt.axvline(2.269, color='k', linestyle='--', label=r"$T_c$")
    plt.xlabel(r"Temperature $T$")
    plt.ylabel(r"Correlation Length $\xi$")
    plt.title(r"Divergence of Correlation Length $\xi(T)$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/Fig_Correlation_Length_Divergence.png", dpi=300)
    plt.close()
    
    # Log-Log for Critical Exponent
    # Plot log(xi) vs log(|T-Tc|)
    tc = 2.269
    t_arr = np.array(ts)
    xi_arr = np.array(xis)
    
    # Filter T > Tc (disordered phase usually cleaner for xi fit)
    mask = (t_arr > tc) & (xi_arr > 0.1)
    if np.sum(mask) > 2:
        reduced_t = t_arr[mask] - tc
        
        plt.figure(figsize=(8, 6))
        plt.loglog(reduced_t, xi_arr[mask], 'o', color='blue')
        
        # Fit power law xi = a * t^(-nu)
        # log(xi) = log(a) - nu * log(t)
        def linear(x, m, c): return m * x + c
        popt, cov = curve_fit(linear, np.log(reduced_t), np.log(xi_arr[mask]))
        nu = -popt[0]
        
        plt.plot(reduced_t, np.exp(popt[1]) * reduced_t**(-nu), 'r--', label=f"Fit $\\nu = {nu:.2f}$")
        plt.xlabel(r"$|T - T_c|$")
        plt.ylabel(r"$\xi$")
        plt.title("Critical Scaling of Correlation Length")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        plt.savefig("results/figures/Fig_Critical_Exponent_Nu.png", dpi=300)
        plt.close()

    # PLOT 3: Data Collapse (Rescaled)
    # C(r) * r^eta vs r/xi
    # Theory: C(r) = r^(-eta) * F(r/xi)
    # So C(r) * r^eta = F(r/xi)
    # If we plot (C(r)*r^eta) vs (r/xi), curves should collapse
    plt.figure(figsize=(10, 8))
    eta = 0.25 # 2D Ising
    
    for T, (r, cr) in results.items():
        if T not in xi_data or xi_data[T] <= 0: continue
        xi = xi_data[T]
        
        # Avoid r=0
        mask = r > 0
        r_valid = r[mask]
        cr_valid = cr[mask]
        
        x_scaled = r_valid / xi
        y_scaled = cr_valid * (r_valid ** eta)
        
        plt.plot(x_scaled, y_scaled, '.', label=f"T={T}")
        
    plt.xlabel(r"$r / \xi$")
    plt.ylabel(r"$C(r) r^{\eta}$")
    plt.title(r"Scaling Collapse ($\eta=0.25$)")
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend() # Too messy if many T
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/Fig_Scaling_Collapse.png", dpi=300)
    plt.close()

def main():
    L = 64 # Use 64 or 128
    temps = [1.0, 1.5, 2.0, 2.2, 2.269, 2.35, 2.5, 3.0, 4.0]
    
    print(f"Running Correlation Analysis L={L}, T={temps}")
    results = measure_correlations(L, temps)
    
    xi_data = {}
    summary_data = []
    
    for T, (r, cr) in results.items():
        xi, A = extract_correlation_length(r, cr, T)
        xi_data[T] = xi
        summary_data.append({"T": T, "Xi": xi, "A": A})
        print(f"T={T:.3f}: Xi = {xi:.3f}")
        
    pd.DataFrame(summary_data).to_csv("results/correlation_lengths.csv", index=False)
    
    os.makedirs("results/figures", exist_ok=True)
    plot_correlations(L, results, xi_data)
    print("Analysis Complete. Figures saved to results/figures/")

if __name__ == "__main__":
    main()
