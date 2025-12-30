"""
Visualization utilities for the Ising Model Simulation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def set_style():
    """Configure matplotlib/seaborn for publication-quality plots."""
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("ticks")
    plt.rc('text', usetex=False) # Use standard fonts if LaTeX not avail
    plt.rc('font', family='serif')

def plot_thermodynamics(results_df, output_path: str = None):
    """
    Plot energy, magnetization, susceptibility, and specific heat.
    
    Args:
        results_df (pd.DataFrame): DataFrame containing columns:
            ['T', 'M_mean', 'E_mean', 'Chi', 'Cv', 'L']
        output_path (str): If provided, save figure to this path.
    """
    set_style()
    
    Ls = results_df['L'].unique()
    colors = sns.color_palette("viridis", len(Ls))
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    
    for i, L in enumerate(Ls):
        data = results_df[results_df['L'] == L]
        data = data.sort_values('T')
        c = colors[i]
        
        # Magnetization
        axes[0, 0].plot(data['T'], data['M_mean'], 'o-', label=f"L={L}", color=c, markersize=4)
        axes[0, 0].set_ylabel(r"$|M|$")
        
        # Energy
        axes[0, 1].plot(data['T'], data['E_mean'], 'o-', color=c, markersize=4)
        axes[0, 1].set_ylabel(r"$E$")
        
        # Susceptibility
        axes[1, 0].plot(data['T'], data['Chi'], 'o-', color=c, markersize=4)
        axes[1, 0].set_ylabel(r"$\chi$")
        axes[1, 0].set_xlabel("Temperature T")
        
        # Specific Heat
        axes[1, 1].plot(data['T'], data['Cv'], 'o-', color=c, markersize=4)
        axes[1, 1].set_ylabel(r"$C_v$")
        axes[1, 1].set_xlabel("Temperature T")

    axes[0, 0].legend()
    axes[0, 0].set_title("Magnetization")
    axes[0, 1].set_title("Energy")
    axes[1, 0].set_title("Susceptibility")
    axes[1, 1].set_title("Specific Heat")
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    return fig

def plot_lattice(grid, title=None):
    """
    Quickly visualize a lattice state.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='gray', vmin=-1, vmax=1)
    if title:
        plt.title(title)
    plt.axis('off')
    return plt.gcf()
