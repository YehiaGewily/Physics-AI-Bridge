# Interactive Ising Model Dashboard

This dashboard allows you to explore the 2D Ising Model in real-time using an interactive web interface.

## Features

- **Real-time Simulation**: Watch spins evolve at >20 FPS.
- **Interactive Controls**: Adjust Temperature, Field, and Coupling on the fly.
- **Live Analysis**: Track Magnetization, Energy, and Phase Diagram trajectory.
- **Phase Diagram Tracker**: See where the system lies compared to the theoretical Onsager solution.

## Prerequisites

Ensure you have the requirements installed:

```bash
pip install -r spin-equilibrium/requirements.txt
```

## How to Run

Navigate to the project root and run:

```bash
streamlit run spin-equilibrium/viz/dashboard.py
```

The dashboard will open in your default web browser (usually at <http://localhost:8501>).

## Usage Tips

- **Start**: Check the "Run Simulation" box in the sidebar.
- **Pause**: Uncheck the box to pause.
- **Reset**: Click "Reset / Randomize" to start fresh (random spins).
- **Critical Slowing Down**: Try setting $T \approx 2.27$ and watch the large clusters form and fluctuate slowly.
- **Hysteresis**: Set $T=1.0$, start with $B=2.0$, then slowly slide $B$ to $-2.0$ and observe the reluctance to flip.
