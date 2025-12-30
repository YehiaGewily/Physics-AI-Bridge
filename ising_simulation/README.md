# 2D Ising Model Simulation Package

A professional Python package for simulating, analyzing, and visualizing the 2D Ising Model using the Metropolis-Hastings algorithm.

## Features

- **Efficient Simulation**: Vectorized checkerboard updates using NumPy.
- **Thermodynamic Analysis**: Calculate Energy, Magnetization, Specific Heat, and Susceptibility.
- **Critical Phenomena**: Finite-Size Scaling tools to extract critical exponents.
- **Advanced Visualization**: Interactive Streamlit dashboard and publication-quality static plots.
- **Extensible Architecture**: Modular design separating core logic, simulation engine, and analysis.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from ising.core import IsingLattice
from ising.simulation import MetropolisEngine

# Initialize Lattice
model = IsingLattice(L=64, T=2.269)
engine = MetropolisEngine(model)

# Equilibrate
engine.step(steps=1000)

# Measure
print(f"Magnetization: {model.magnetization}")
```

## Directory Structure

- `ising/`: Core package source code.
- `scripts/`: Ready-to-run experiments (Phase Transition, Hysteresis, etc.).
- `notebooks/`: Jupyter notebooks for exploration.
- `tests/`: Unit tests using pytest.
- `figures/`: Generated plots.

## Interactive Dashboard

Run the real-time visualization dashboard:

```bash
streamlit run ising/viz/dashboard.py
```

(Note: You may need to copy the dashboard script from the old structure or update the path).

## License

MIT
