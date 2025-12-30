import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.optimize import curve_fit, minimize
import seaborn as sns

def load_data(path="results/fss_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def parabolic_fit_peak(T, Y):
    """
    Find peak by fitting parabola to the top points.
    Returns (T_peak, Y_peak)
    """
    # Find approx max
    idx_max = np.argmax(Y)
    
    # Take window around max
    window = 3
    start = max(0, idx_max - window)
    end = min(len(T), idx_max + window + 1)
    
    if end - start < 3:
        # Fallback to discrete max
        return T[idx_max], Y[idx_max]
        
    t_win = T[start:end]
    y_win = Y[start:end]
    
    # Fit y = a(t-t0)^2 + c  -> y = at^2 + bt + c
    p = np.polyfit(t_win, y_win, 2)
    
    # Peak is at -b / 2a
    # p[0]t^2 + p[1]t + p[2]
    a, b, c = p
    
    if a >= 0: # Convex, minimum not maximum
        return T[idx_max], Y[idx_max]
        
    t_peak = -b / (2*a)
    y_peak = a*(t_peak**2) + b*t_peak + c
    
    return t_peak, y_peak

def analyze_scaling_peaks(data_map):
    Ls = sorted(data_map.keys())
    
    chi_peaks = []
    T_chi_peaks = []
    
    cv_peaks = []
    T_cv_peaks = []
    
    for L in Ls:
        d = data_map[L]
        T = d['T']
        Chi = d['Chi_mean']
        Cv = d['Cv_mean']
        
        tc_chi, chi_max = parabolic_fit_peak(T, Chi)
        chi_peaks.append(chi_max)
        T_chi_peaks.append(tc_chi)
        
        tc_cv, cv_max = parabolic_fit_peak(T, Cv)
        cv_peaks.append(cv_max)
        T_cv_peaks.append(tc_cv)
        
    return {
        'L': np.array(Ls),
        'Chi_max': np.array(chi_peaks),
        'T_chi': np.array(T_chi_peaks),
        'Cv_max': np.array(cv_peaks),
        'T_cv': np.array(T_cv_peaks)
    }

def fit_power_law(L, Y):
    """Fit Y = A * L^exponent"""
    # log Y = log A + exp * log L
    slope, intercept = np.polyfit(np.log(L), np.log(Y), 1)
    exponent = slope
    A = np.exp(intercept)
    return A, exponent

def fit_tc_scaling(L, T_L):
    """Fit T(L) = Tc + a * L^(-1/nu) -> Fix nu=1, so T(L) = Tc + a/L"""
    # Fit y = mx + c where x = 1/L, y = T_L, c = Tc
    inv_L = 1.0 / L
    slope, intercept = np.polyfit(inv_L, T_L, 1)
    Tc_extrap = intercept
    return Tc_extrap, slope

def quality_of_collapse(params, data_map, quantity='Chi'):
    """
    Objective function for data collapse.
    params: [Tc, nu, exponent]
    For Chi: exponent = gamma
    Scaled Y = Chi * L^(-gamma/nu)
    Scaled X = (T - Tc) * L^(1/nu)
    Goal: Minimize variance of Y at fixed X (points should lie on one curve).
    Implementation: Sort by X, calculate local variance.
    """
    Tc, nu, exponent = params
    
    all_x = []
    all_y = []
    
    for L, d in data_map.items():
        T = d['T']
        if quantity == 'Chi':
            Y = d['Chi_mean']
        elif quantity == 'M':
            Y = d['M_mean']
            
        # x_scaled = t * L^(1/nu), t = (T-Tc)/Tc
        t = (T - Tc) / Tc
        x_s = t * (L**(1/nu))
        
        # y_scaled = Y * L^(-exponent/nu)
        y_s = Y * (L**(-exponent/nu))
        
        all_x.extend(x_s)
        all_y.extend(y_s)
        
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Sort
    idx = np.argsort(all_x)
    sx = all_x[idx]
    sy = all_y[idx]
    
    # Calculate difference between adjacent points (rough smoothness metric)
    # Better: sum of squared differences of y values for close x values
    # Simple measure: Sum |y_{i+1} - y_i|
    # This favors simpler curves.
    diffs = np.sum(np.abs(np.diff(sy)))
    return diffs

def perform_data_collapse_fit(data_map, qty='Chi', p0=[2.269, 1.0, 1.75]):
    # Try to optimize parameters to minimize 'messiness' of curve
    # This is hard to automate robustly without good bounds.
    # We will use the provided theory values for plotting first (Plot 3), 
    # but strictly fitting collapse is advanced.
    # We'll stick to manual parameters for "Proof" unless requested to fit.
    # User asked: "All L curves should collapse... This is the smoking gun".
    # User didn't strictly ask to blind-extract exponents from collapse, just "Show data collapse".
    # So we use best known Tc, nu, gamma/beta.
    return p0

def plot_fss(data_map, peak_data, output_dir):
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    
    Ls = peak_data['L']
    
    # PLOT 1: Peak Scaling (Chi_max vs L)
    plt.figure(figsize=(8, 6))
    plt.loglog(Ls, peak_data['Chi_max'], 'o', markersize=8, color='blue')
    
    A, gamma_nu = fit_power_law(Ls, peak_data['Chi_max'])
    plt.plot(Ls, A * Ls**gamma_nu, 'r--', label=f"Fit $\gamma/\\nu = {gamma_nu:.3f}$")
    plt.plot(Ls, A * Ls**1.75, 'k:', label=r"Theory $\gamma/\nu = 1.75$")
    
    plt.xlabel(r"Lattice Size $L$")
    plt.ylabel(r"Peak Susceptibility $\chi_{max}$")
    plt.title(r"Finite-Size Scaling of Susceptibility")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig_FSS_Peak_Scaling.png", dpi=300)
    plt.close()
    
    # PLOT 2: Tc Extrapolation
    plt.figure(figsize=(8, 6))
    plt.plot(1/Ls, peak_data['T_chi'], 's', markersize=8, color='green')
    
    Tc_extrap, slope = fit_tc_scaling(Ls, peak_data['T_chi'])
    x_fit = np.linspace(0, max(1/Ls)*1.1, 10)
    plt.plot(x_fit, Tc_extrap + slope*x_fit, 'r--', label=f"$T_c(\\infty) = {Tc_extrap:.4f}$")
    plt.axhline(2.269, color='k', linestyle=':', label=r"Onsager $T_c = 2.269$")
    
    plt.xlabel(r"$1/L$")
    plt.ylabel(r"Pseudo-critical Temperature $T_{\chi}(L)$")
    plt.title(r"Extrapolation of Critical Temperature")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, max(1/Ls)*1.2)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig_FSS_Tc_Scaling.png", dpi=300)
    plt.close()
    
    # PLOT 3: Data Collapse Chi
    # y = Chi * L^(-gamma/nu), x = t * L^(1/nu)
    # Theory: gamma=1.75, nu=1.0, Tc=2.269
    # Use extracted Tc if better? Let's use extraction.
    Tc = Tc_extrap
    nu = 1.0
    gamma = 1.75
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("viridis", len(Ls))
    
    for i, L in enumerate(Ls):
        d = data_map[L]
        T = d['T']
        Chi = d['Chi_mean']
        
        t = (T - Tc) / Tc
        x_s = t * (L**(1/nu))
        y_s = Chi * (L**(-gamma/nu))
        
        plt.plot(x_s, y_s, '.', label=f"L={L}", color=colors[i])
        
    plt.xlabel(r"Scaled Temperature $t L^{1/\nu}$")
    plt.ylabel(r"Scaled Susceptibility $\chi L^{-\gamma/\nu}$")
    plt.title(r"Susceptibility Data Collapse ($\gamma=1.75, \nu=1$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig_FSS_Chi_Collapse.png", dpi=300)
    plt.close()

    # PLOT 4: Data Collapse M
    # y = M * L^(beta/nu), x = t * L^(1/nu)
    # Theory: beta=0.125
    beta = 0.125
    
    plt.figure(figsize=(10, 8))
    
    for i, L in enumerate(Ls):
        d = data_map[L]
        T = d['T']
        M = d['M_mean']
        
        t = (T - Tc) / Tc
        x_s = t * (L**(1/nu))
        y_s = M * (L**(beta/nu))
        
        # Only plot near transition to see collapse clearly
        # mask = (x_s > -5) & (x_s < 5)
        plt.plot(x_s, y_s, '.', label=f"L={L}", color=colors[i])
        
    plt.xlabel(r"Scaled Temperature $t L^{1/\nu}$")
    plt.ylabel(r"Scaled Magnetization $M L^{\beta/\nu}$")
    plt.title(r"Magnetization Data Collapse ($\beta=0.125, \nu=1$)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Fig_FSS_M_Collapse.png", dpi=300)
    plt.close()
    
    return gamma_nu, Tc_extrap

def main():
    print("Loading FSS Data...")
    if not os.path.exists("results/fss_data.pkl"):
        print("Error: results/fss_data.pkl not found. Run fss_run.py first.")
        return
        
    data_map = load_data("results/fss_data.pkl")
    print(f"Loaded L={list(data_map.keys())}")
    
    print("Analyzing Peaks...")
    peak_data = analyze_scaling_peaks(data_map)
    
    print("Generating Plots...")
    os.makedirs("results/figures", exist_ok=True)
    gamma_nu, Tc_fit = plot_fss(data_map, peak_data, "results/figures")
    
    # Output Table
    print("\n=== Critical Exponents Analysis ===")
    print(f"Tc (Extrapolated): {Tc_fit:.4f} (Theory: 2.2692)")
    print(f"gamma/nu (Measured): {gamma_nu:.3f} (Theory: 1.75)")
    print(f"nu assumed: 1.0")
    print(f"beta assumed: 0.125")
    print("===================================")
    
    # Save to CSV
    df = pd.DataFrame([{
        "Metric": "Tc", "Measured": Tc_fit, "Theory": 2.2692
    }, {
        "Metric": "gamma/nu", "Measured": gamma_nu, "Theory": 1.75
    }])
    df.to_csv("results/critical_exponents.csv", index=False)

if __name__ == "__main__":
    main()
