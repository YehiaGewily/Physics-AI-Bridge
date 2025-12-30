import sys
import os
import numpy as np
import pickle
import time
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Dict

# Add spin-equilibrium to path explicitly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'spin-equilibrium')))
from core.ising_model import IsingSimulation

@dataclass
class SimulationResults:
    lattice_size: int
    temperatures: List[float]
    magnetization: List[float] = field(default_factory=list)
    energy: List[float] = field(default_factory=list)
    susceptibility: List[float] = field(default_factory=list)
    specific_heat: List[float] = field(default_factory=list)
    autocorrelation_times: List[float] = field(default_factory=list)
    spatial_correlations: Dict[float, np.ndarray] = field(default_factory=dict) # T -> C(r) array

def compute_integrated_autocorrelation_time(series: np.ndarray) -> float:
    """
    Compute integrated autocorrelation time using a windowing method (e.g. Sokal's adaptive window).
    Simple definition: tau = 1 + 2 * sum_{t=1}^W rho(t)
    where rho(t) is normalized autocorrelation function.
    """
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    
    if var == 0:
        return 0.0
        
    # Subtract mean
    centered = series - mean
    
    # Compute Autocorrelation Function (ACF) via FFT for speed
    ft = np.fft.fft(centered)
    acf = np.fft.ifft(ft * np.conj(ft)).real
    acf = acf[:n//2] / acf[0] # Normalize, take first half
    
    # Estimate tau
    # We sum until the correlation drops below zero or negligible noise
    # Simple heuristic: sum until rho < 0 or use adaptive window
    tau = 1.0
    for t in range(1, len(acf)):
        if acf[t] <= 0:
            break
        tau += 2 * acf[t]
        
    return tau

def run_single_lattice(L: int, temps: np.ndarray, 
                       equilibration_sweeps: int = 1000, 
                       measurement_sweeps: int = 5000,
                       sample_interval: int = 10) -> SimulationResults:
    
    print(f"--> Simulating Lattice Size L={L}")
    
    results = SimulationResults(lattice_size=L, temperatures=temps.tolist())
    
    # We can reuse the model to avoid reallocation, but resetting spins at high T is better
    # or start with random for each T.
    # Standard practice: Start high T, cool down (simulated annealing like) 
    # OR random start each time. The prompt implies independent points but sequential T is often faster for equilibration if close.
    # Let's do random start for each T to be safe and independent.
    
    for T in tqdm(temps, desc=f"Sweeping T (L={L})", leave=False):
        model = IsingSimulation(size=L, temperature=T)
        
        # 1. Equilibration
        model.metropolis_step(steps_per_sweep=equilibration_sweeps)
        
        # 2. Measurement Phase
        measurements_E = []
        measurements_M = []
        measurements_abs_M = []
        
        # Number of actual samples
        num_samples = measurement_sweeps // sample_interval
        
        for _ in range(num_samples):
            model.metropolis_step(steps_per_sweep=sample_interval)
            
            # Record properties
            E_val = model.energy()
            M_val = model.magnetization
            
            measurements_E.append(E_val)
            measurements_M.append(M_val) 
            measurements_abs_M.append(abs(M_val))
            
        # Convert to numpy arrays for vectorized math
        E_series = np.array(measurements_E) # Total Energy
        M_series = np.array(measurements_M) # Total Magnetization (signed)
        abs_M_series = np.array(measurements_abs_M) # Total Absolute Magnetization
        
        N_spins = L * L
        
        # Thermodynamic Averages (per spin)
        mean_E = np.mean(E_series)
        mean_M_abs = np.mean(abs_M_series) # Order parameter Usually <|M|> for finite finite size
        mean_M2 = np.mean(M_series**2)
        mean_E2 = np.mean(E_series**2)
        
        # Store basic averages (per spin)
        results.energy.append(mean_E / N_spins)
        results.magnetization.append(mean_M_abs / N_spins)
        
        # Derived Quantities
        # Cv = (beta^2 / N) * (<E^2> - <E>^2)
        beta = 1.0 / T
        var_E = mean_E2 - mean_E**2
        Cv = (beta**2 / N_spins) * var_E
        results.specific_heat.append(Cv)
        
        # Chi = (beta / N) * (<M^2> - <|M|>^2)
        # Note: Often defined with <|M|> for finite systems to detect transition
        var_M = mean_M2 - mean_M_abs**2
        Chi = (beta / N_spins) * var_M
        results.susceptibility.append(Chi)
        
        # Autocorrelation Time (on Magnetization series)
        tau = compute_integrated_autocorrelation_time(M_series)
        results.autocorrelation_times.append(tau)
        
        # Spatial Correlation (Snapshot at end of run)
        # We compute correlation up to L/4
        max_r = L // 4
        corr_func = model.compute_spatial_correlation(max_r=max_r)
        results.spatial_correlations[T] = corr_func
        
    return results

def main():
    # Parameters according to requirements
    LATTICE_SIZES = [32, 64, 128, 256]
    TEMPS = np.linspace(0.5, 4.0, 50)
    
    print("==================================================")
    print("Starting Comprehensive 2D Ising Model Simulation")
    print("==================================================")
    print(f"Lattice Sizes: {LATTICE_SIZES}")
    print(f"Temperature Range: {TEMPS[0]} - {TEMPS[-1]} (50 points)")
    print(f"Output Directory: results/")
    
    all_results = {}
    
    os.makedirs("results", exist_ok=True)
    
    start_time = time.time()
    
    for L in LATTICE_SIZES:
        sim_result = run_single_lattice(L, TEMPS)
        all_results[L] = sim_result
        
        # Intermediate Save
        with open(os.path.join("results", "ising_data_partial.pkl"), "wb") as f:
            pickle.dump(all_results, f)
            
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nSimulation Complete in {elapsed:.2f} seconds.")
    
    # Final Save
    final_path = os.path.join("results", "ising_data_full.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {final_path}")
    
    # Also save as JSON for portability (without numpy arrays if possible, or skip)
    # We stick to pickle for NumPy structures as requested "pickle/json"
    
if __name__ == "__main__":
    main()
