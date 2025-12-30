"""
Statistical analysis tools for Ising Simulation data.
"""

import numpy as np
import pandas as pd

def calculate_errors(series):
    """
    Calculate standard error of the mean using jackknife or bootstrap?
    For uncorrelated data, simple std/sqrt(N).
    For correlated data (Monte Carlo), we need binning or autocorrelation time.
    Here we assume uncorrelated (subsampled) data.
    """
    return np.std(series) / np.sqrt(len(series))

def compute_autocorrelation(series, max_lag=1000):
    """
    Compute time-displaced autocorrelation using FFT.
    A(t) = <(x_0 - mean)(x_t - mean)> / var
    """
    n = len(series)
    mean = np.mean(series)
    var = np.var(series)
    if var < 1e-10: return np.ones(max_lag)
    
    centered = series - mean
    ft = np.fft.rfft(centered, n=2*n)
    acf = np.fft.irfft(ft * np.conj(ft))
    acf = acf[:max_lag] / acf[0]
    return acf

def integrate_autocorrelation_time(acf):
    """Tau_int = 0.5 + sum_{t=1}^cutoff A(t)"""
    tau = 0.5
    for val in acf[1:]:
        if val <= 0: break
        tau += val
    return tau
