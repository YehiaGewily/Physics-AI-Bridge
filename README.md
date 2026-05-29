# Physics-AI Bridge

A structured research project tracing the mathematical foundation of artificial intelligence back to its roots in statistical mechanics. Each phase proves that the same energy minimization principle that governs physical systems also governs neural computation.

## The Big Idea

Inspired by the ScienceClic English video "The Physics of A.I.", this project proves that AI is not "just code" — it is the laws of statistical mechanics in action. We trace a single mathematical thread from 19th-century physics through modern deep learning to quantum field theory.

```
Ising Model (Phase 1) ✅
↓ Replace fixed coupling J with learned weights Wᵢⱼ
Hopfield Networks (Phase 2) ✅
↓ Add hidden units and stochasticity
Boltzmann Machines (Phase 2.5) 🔜
↓ Stack layers + backpropagation
Deep Belief Networks
↓ Take width to infinity
Neural Network Gaussian Processes (Phase 3) 🔜
↓ Recognize as free quantum field
Quantum Field Theory Connection (Phase 4) 🔜
```

## Progress Tracker

| Phase | Title | Status | Key Result |
|-------|-------|--------|------------|
| 1 | 2D Ising Model | ✅ Complete | Tc = 2.2677 (0.05% error vs Onsager exact) |
| 2 | Hopfield Networks | ✅ Complete | 68% strict recall at 25% corruption; Z₂ symmetry empirically confirmed (201/201 split) |
| 2.5 | Boltzmann Machines | 🔜 Next | Hidden units + Contrastive Divergence |
| 3 | Neural Network Gaussian Processes | 🔜 Planned | Finite → infinite width convergence |
| 4 | Quantum Field Theory Connection | 🔜 Planned | NN ↔ Free quantum field theory |

## Published Papers

### Phase 1: 2D Ising Model

- **PDF:** [`docs/Computational_2D_Ising_Simulation_Phase1.pdf`](docs/Computational_2D_Ising_Simulation_Phase1.pdf)
- Validated Tc to 0.05% accuracy against Onsager's exact solution
- 60 FPS real-time simulation of 65,536 spins
- Critical exponents within 4.5% of theory

### Phase 2: Hopfield Networks

- **PDF:** [`docs/Hopfield Networks as Learnable Ising Models - Phase 2.pdf`](docs/Hopfield%20Networks%20as%20Learnable%20Ising%20Models%20-%20Phase%202.pdf)
- **LaTeX source:** [`docs/phase2_project/`](docs/phase2_project/)
- Demonstrated formal Ising–Hopfield isomorphism
- Empirically confirmed Z₂ symmetry through 500-trial spurious states analysis
- 9.65-unit energy gap between stored attractors and spurious states

## Authors

- **Yehia Said Gewily** — Software & Data Engineer, Machine Learning  
  [yehia@nover.studio](mailto:yehia@nover.studio)
- **Omar Hosney Mahmoud** (Phase 2 co-author) — Co-founder and CTO, Nover Studio  
  [omar@nover.studio](mailto:omar@nover.studio)

The five stored memory patterns used in Phase 2 (N, O, V, E, R) spell "NOVER" — the authors' AI-driven media production startup.

## Repository Structure

```
Physics-AI-Bridge/
├── docs/                                    # Papers and documentation
│   ├── Computational_2D_Ising_Simulation_Phase1.pdf
│   ├── Hopfield Networks as Learnable Ising Models - Phase 2.pdf
│   ├── Physics_of_AI_Research_Project.pdf  # Original project guide
│   └── phase2_project/                     # Phase 2 LaTeX project
│       ├── main.tex                        # LaTeX source
│       ├── main.pdf                        # Compiled paper
│       ├── references.bib                  # Bibliography
│       ├── README.md                       # Build instructions
│       ├── figures/                        # All 8 paper figures (PNG, 300 DPI)
│       └── data/                           # Raw experimental data (CSVs)
├── spin-equilibrium/                       # Phase 1: Interactive dashboard
│   ├── core/ising_model.py                 # IsingSimulation class
│   └── viz/                                # Streamlit + Pygame visualizers
├── ising_simulation/                       # Modular simulation package
│   ├── ising/                              # Phase 1: Modular Ising components
│   │   ├── core.py                         # SimulationParams, IsingLattice
│   │   └── simulation.py                   # MetropolisEngine
│   └── hopfield/                           # Phase 2: Hopfield Network
│       ├── core.py                         # HopfieldNetwork class
│       ├── utils.py                        # Pattern utilities (NOVER letters)
│       └── visualization.py               # Visualization tools
├── experiments/                            # Reproducible experiments
│   ├── # Phase 1 experiments
│   ├── run_simulation.py                   # Full thermodynamic suite
│   ├── hysteresis_loop.py                  # Hysteresis analysis
│   ├── correlation_analysis.py             # Spatial correlations
│   ├── equilibration_analysis.py           # Equilibration dynamics
│   ├── fss_run.py / fss_analyze.py         # Finite-size scaling
│   ├── # Phase 2 experiments
│   ├── experiment_1_single_pattern.py      # Robustness vs corruption
│   ├── experiment_2_capacity.py            # Storage capacity
│   ├── experiment_2b_spurious.py           # Spurious states + Z₂ symmetry
│   ├── experiment_3_energy_landscape.py    # PCA energy landscape
│   ├── comparison_ising_hopfield.py        # Isomorphism figure
│   └── generate_recall_gif.py             # Recall animation
└── results/
    ├── ising/                              # Phase 1 figures
    ├── hopfield/                           # Phase 2 figures, CSVs, GIF
    └── animations/                         # MP4 + GIF animations
```

## Quick Start

### Run the Phase 1 Interactive Dashboard

```bash
pip install -r spin-equilibrium/requirements.txt
streamlit run spin-equilibrium/viz/dashboard.py
```

The dashboard lets you adjust temperature, coupling, and external field in real time, and overlays your simulation against the exact Onsager solution.

### Reproduce Phase 1 Results

```bash
python experiments/run_simulation.py        # Full thermodynamic suite
python experiments/hysteresis_loop.py       # Hysteresis loops
python experiments/fss_run.py               # Finite-size scaling data
python experiments/fss_analyze.py           # Critical exponents
```

### Reproduce Phase 2 Results

```bash
python experiments/experiment_1_single_pattern.py    # Single pattern robustness
python experiments/experiment_2_capacity.py           # Storage capacity
python experiments/experiment_2b_spurious.py          # Spurious states
python experiments/experiment_3_energy_landscape.py   # Energy landscape
python experiments/comparison_ising_hopfield.py       # Isomorphism comparison
python experiments/generate_recall_gif.py             # Recall animation
```

### Compile the Phase 2 LaTeX Paper

```bash
cd docs/phase2_project
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `docs/phase2_project/main.pdf`

## Key Results

### Phase 1: 2D Ising Model

| Metric | Measured | Theory | Error |
|--------|----------|--------|-------|
| Critical temperature Tc | 2.2677 ± 0.001 | 2.269 | 0.05% |
| Critical exponent γ/ν | 1.672 ± 0.05 | 1.75 | 4.5% |
| Peak susceptibility | 63.2 at T=2.269 | — | — |
| Autocorrelation at Tc | ~85 sweeps | — | 160× from T=1.0 |

### Phase 2: Hopfield Networks

| Metric | Value |
|--------|-------|
| Strict recall at 25% corruption | 68.0% ± 6.7% |
| Convergence steps (avg) | 4.4 |
| Capacity collapse (<50%) | Between P=3 and P=5 |
| Theoretical capacity (AGS) | 0.138N = 13.8 patterns |
| Stored vs Inverted convergence | 201/201 (exact Z₂ symmetry) |
| Energy gap (spurious vs stored) | 9.65 units |

## Inspiration & Acknowledgments

This project was inspired by the YouTube channel **ScienceClic English** and their video "The Physics of A.I." which sketches the conceptual bridge from statistical mechanics to modern AI.

## License

MIT License.

## Contact

For questions, collaborations, or discussion:
- Yehia Said Gewily: [yehia@nover.studio](mailto:yehia@nover.studio)
- Omar Hosney Mahmoud: [omar@nover.studio](mailto:omar@nover.studio)
