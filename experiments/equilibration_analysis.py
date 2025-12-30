import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
# from scipy.special import expit # Unused
import seaborn as sns

# Add path to spin-equilibrium root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

def run_equilibration_track(L, T, mode='cold', sweeps=10000):
    """
    Run simulation tracking E and M every sweep.
    mode: 'cold' (all up) or 'hot' (random)
    """
    sim = IsingSimulation(size=L, temperature=T)
    
    if mode == 'cold':
        sim.grid = np.ones((L, L), dtype=np.int8)
    else:
        # random start (default)
        sim.grid = np.random.choice([-1, 1], size=(L, L)).astype(np.int8)
        
    e_history = []
    m_history = []
    
    N = L*L
    
    for _ in range(sweeps):
        sim.metropolis_step(steps_per_sweep=1)
        # Record intensive quantities
        e_history.append(sim.energy() / N)
        m_history.append(abs(sim.magnetization) / N)
        
    return np.array(e_history), np.array(m_history)

def calculate_time_autocorrelation(series, max_lag=1000):
    """
    Compute normalized autocorrelation function A(t).
    A(t) = [ <X(0)X(t)> - <X>^2 ] / Variance
    """
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return np.ones(max_lag)
        
    # Subtract mean
    centered = series - mean
    
    # Compute using FFT for efficiency
    # Padding to avoid circular correlation
    ft = np.fft.rfft(centered, n=2*n)
    acf = np.fft.irfft(ft * np.conj(ft))
    
    # Normalize and truncate
    acf = acf[:max_lag] / acf[0]
    
    return acf

def integrated_autocorrelation_time(acf):
    """
    Sum A(t) until it hits zero/noise.
    tau_int = 0.5 + sum(acf[1:cutoff])
    """
    tau = 0.5
    for val in acf[1:]:
        if val <= 0:
            break
        tau += val
    return tau

def fit_exponential_decay(t, acf):
    """Fit A(t) ~ exp(-t/tau)"""
    # Simply log linear fit on initial part where A(t) > 0.1
    mask = (acf > 0.1) & (t > 0)
    if np.sum(mask) < 5:
        return integrated_autocorrelation_time(acf) # Fallback
        
    x = t[mask]
    y = np.log(acf[mask])
    
    # y = -1/tau * x
    try:
        popt, _ = curve_fit(lambda x, inv_tau: -inv_tau * x, x, y)
        return 1.0 / popt[0]
    except:
        return integrated_autocorrelation_time(acf)

def analyze_equilibration(L, temps, sweeps=10000):
    results = {}
    
    for T in tqdm(temps, desc="Equilibration Analysis"):
        # Cold Start
        e_cold, m_cold = run_equilibration_track(L, T, 'cold', sweeps)
        # Hot Start
        e_hot, m_hot = run_equilibration_track(L, T, 'hot', sweeps)
        
        results[T] = {
            'cold': {'E': e_cold, 'M': m_cold},
            'hot': {'E': e_hot, 'M': m_hot}
        }
        
    return results

def analyze_autocorrelation(L, temps, sweeps_meas=20000, sweeps_equil=2000):
    tau_results = {}
    acf_data = {}
    
    for T in tqdm(temps, desc="Autocorrelation Analysis"):
        sim = IsingSimulation(size=L, temperature=T)
        sim.metropolis_step(steps_per_sweep=sweeps_equil)
        
        # Measure M sequence
        m_series = []
        N = L*L
        for _ in range(sweeps_meas):
            sim.metropolis_step(steps_per_sweep=1)
            m_series.append(abs(sim.magnetization)/N)
            
        m_series = np.array(m_series)
        
        # Calculate ACF
        max_lag = min(1000, sweeps_meas // 10)
        acf = calculate_time_autocorrelation(m_series, max_lag)
        t_vals = np.arange(max_lag)
        
        # Extract Tau (Exponential fit usually better for measuring independence)
        tau_exp = fit_exponential_decay(t_vals, acf)
        tau_int = integrated_autocorrelation_time(acf)
        
        tau_results[T] = {'tau_exp': tau_exp, 'tau_int': tau_int}
        acf_data[T] = acf
        
    return tau_results, acf_data

def plot_equilibration(results, L):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 4*len(results)), sharex=True)
    if len(results) == 1: axes = [axes]
    
    for i, (T, data) in enumerate(results.items()):
        ax_m = axes[i][0]
        ax_e = axes[i][1]
        
        steps = np.arange(len(data['cold']['M']))
        
        # Magnetization
        ax_m.plot(steps, data['cold']['M'], label='Cold (All +1)', alpha=0.8, linewidth=1.5)
        ax_m.plot(steps, data['hot']['M'], label='Hot (Random)', alpha=0.8, linewidth=1.5)
        ax_m.set_ylabel(r"Magnetization $|M|$")
        ax_m.set_title(f"Equilibration T={T}")
        ax_m.grid(True, alpha=0.3)
        if i==0: ax_m.legend()
        
        # Energy
        ax_e.plot(steps, data['cold']['E'], alpha=0.8, linewidth=1.5)
        ax_e.plot(steps, data['hot']['E'], alpha=0.8, linewidth=1.5)
        ax_e.set_ylabel(r"Energy $E$")
        ax_e.set_title(f"Energy Convergence T={T}")
        ax_e.grid(True, alpha=0.3)
        
        if i == len(results)-1:
            ax_m.set_xlabel("MC Sweeps")
            ax_e.set_xlabel("MC Sweeps")
            
    plt.tight_layout()
    plt.savefig("results/figures/Fig_Equilibration_Curves.png", dpi=300)
    plt.close()

def plot_autocorrelation(acf_data, tau_results, L):
    plt.figure(figsize=(10, 8))
    
    colors = sns.color_palette("viridis", len(acf_data))
    
    for i, (T, acf) in enumerate(acf_data.items()):
        t = np.arange(len(acf))
        tau = tau_results[T]['tau_exp']
        plt.plot(t, acf, label=f"T={T} ($\\tau \\approx {tau:.1f}$)", color=colors[i], linewidth=2)
        
    plt.plot([0, 1000], [0, 0], 'k--', alpha=0.3)
    plt.plot([0, 1000], [np.exp(-1)]*2, 'k:', alpha=0.3, label="1/e")
    
    plt.xlabel(r"Time Lag $t$ (Sweeps)")
    plt.ylabel(r"Autocorrelation $A(t)$")
    plt.title(f"Magnetization Autocorrelation (L={L})")
    plt.xlim(0, 500)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/Fig_Autocorrelation_Functions.png", dpi=300)
    plt.close()
    
    # Tau vs T
    plt.figure(figsize=(8, 6))
    temps = sorted(tau_results.keys())
    taus = [tau_results[t]['tau_exp'] for t in temps]
    
    plt.plot(temps, taus, 'o-', color='crimson')
    plt.axvline(2.269, color='k', linestyle='--', label=r"$T_c$")
    plt.xlabel("Temperature T")
    plt.ylabel(r"Autocorrelation Time $\tau_{auto}$")
    plt.title("Critical Slowing Down")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/Fig_Tau_vs_T.png", dpi=300)
    plt.close()

def main():
    L = 64
    temps_eq = [1.0, 2.0, 2.269, 3.0]
    
    # 1. Equilibration Study
    print("Running Equilibration Study...")
    eq_results = analyze_equilibration(L, temps_eq, sweeps=5000) # 5k usually enough for L=64
    
    os.makedirs("results/figures", exist_ok=True)
    plot_equilibration(eq_results, L)
    
    # 2. Autocorrelation Study
    print("Running Autocorrelation Study...")
    # Need sufficient stats
    tau_results, acf_data = analyze_autocorrelation(L, temps_eq, sweeps_meas=10000, sweeps_equil=2000)
    
    plot_autocorrelation(acf_data, tau_results, L)
    
    # Save Table
    df = pd.DataFrame([
        {"T": T, "Tau_exp": d['tau_exp'], "Tau_int": d['tau_int']}
        for T, d in tau_results.items()
    ])
    df.to_csv("results/equilibration_stats.csv", index=False)
    print("Analysis Complete.")
    print(df)

if __name__ == "__main__":
    main()
