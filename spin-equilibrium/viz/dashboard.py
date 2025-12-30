import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os

# Add path to spin-equilibrium root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.ising_model import IsingSimulation

# Set Page Layout
st.set_page_config(
    page_title="Ising Model Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Onsager Solution for Phase Diagram ---
def onsager_magnetization(T):
    representation = np.zeros_like(T)
    Tc = 2.269
    mask = T < Tc
    # M = (1 - (sinh(2/T))^-4 )^(1/8)
    # Avoid div by zero
    T_valid = T[mask]
    with np.errstate(over='ignore', invalid='ignore'):
        sinh_term = np.sinh(2.0 / T_valid)
        term = 1.0 - (sinh_term)**(-4)
        m_vals = np.power(term, 1.0/8.0)
        # NaN safe
        m_vals = np.nan_to_num(m_vals)
    representation[mask] = m_vals
    return representation

# --- Helper Functions ---
def init_simulation():
    """Re-initialize simulation based on Sidebar inputs."""
    L = st.session_state.L
    T = st.session_state.T
    B = st.session_state.B
    J = st.session_state.J
    
    st.session_state.sim = IsingSimulation(size=L, temperature=T, B=B, J=J)
    st.session_state.history = {
        'M': [], 'E': [], 'steps': []
    }
    st.session_state.step_count = 0

# --- Sidebar Controls ---
st.sidebar.title("Controls")

# 1. System Parameters
st.sidebar.header("System Parameters")

L = st.sidebar.selectbox("Lattice Size (L)", [32, 64, 128, 256], index=1, key='L', on_change=init_simulation)
J = st.sidebar.slider("Coupling (J)", 0.5, 2.0, 1.0, 0.1, key='J') # Update sim dynamically later if L matches

# 2. Thermodynamic Variables
st.sidebar.header("Thermodynamics")
T = st.sidebar.slider("Temperature (T)", 0.5, 4.0, 2.27, 0.01, key='T')
B = st.sidebar.slider("External Field (B)", -2.0, 2.0, 0.0, 0.1, key='B')

# 3. Simulation Control
st.sidebar.header("Simulation")
speed = st.sidebar.slider("Speed (Sweeps/Frame)", 1, 50, 5)
run_btn = st.sidebar.checkbox("Run Simulation", value=False)
reset_btn = st.sidebar.button("Reset / Randomize", on_click=init_simulation)

# --- Session State Management ---
if 'sim' not in st.session_state:
    init_simulation()

# Update Sim Parameters if they changed but L didn't force reset
sim = st.session_state.sim
if sim.size != L:
    init_simulation()
    sim = st.session_state.sim

# Live Parameter Update (T, B, J)
sim.set_temperature(T)
sim.B = B
sim.J = J

# --- Main Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Lattice View")
    
    # Placeholder for the Image
    lattice_placeholder = st.empty()
    
    # Overlay Stats
    stats_placeholder = st.empty()

with col2:
    st.header("Live Analysis")
    
    # 1. Magnetization Plot
    chart_m_placeholder = st.empty()
    
    # 2. Phase Diagram
    st.subheader("Phase Diagram Tracker")
    fig_phase, ax_phase = plt.subplots(figsize=(5, 3))
    t_range = np.linspace(0.1, 4.0, 100)
    m_onsager = onsager_magnetization(t_range)
    ax_phase.plot(t_range, m_onsager, 'k--', label='Onsager (Theory)')
    ax_phase.set_xlabel("Temperature T")
    ax_phase.set_xlim(0, 4.0)
    ax_phase.set_ylim(-0.1, 1.1)
    ax_phase.axvline(2.269, color='r', alpha=0.3, linestyle=':')
    
    # Current point
    point, = ax_phase.plot([], [], 'ro', markersize=8)
    
    phase_plot_placeholder = st.pyplot(fig_phase)

# --- Simulation Loop ---
if run_btn:
    # We update in a loop
    while True:
        # Step
        sim.metropolis_step(steps_per_sweep=speed)
        st.session_state.step_count += speed
        
        # Measure
        m = abs(sim.magnetization) / (L*L)
        e = sim.energy() / (L*L)
        
        # Update History
        st.session_state.history['M'].append(m)
        st.session_state.history['E'].append(e)
        st.session_state.history['steps'].append(st.session_state.step_count)
        
        # Keep history manageable
        limit = 200
        if len(st.session_state.history['M']) > limit:
            st.session_state.history['M'] = st.session_state.history['M'][-limit:]
            st.session_state.history['E'] = st.session_state.history['E'][-limit:]
            st.session_state.history['steps'] = st.session_state.history['steps'][-limit:]
        
        # --- Render ---
        
        # 1. Lattice Image
        # Map -1 -> 0 (Black), 1 -> 255 (White)
        img_data = ((sim.grid + 1) * 127.5).astype(np.uint8)
        # Use simple nearst neighbor via st.image (auto)
        lattice_placeholder.image(img_data, caption=f"Step {st.session_state.step_count}", width="stretch", output_format='PNG', clamp=True, channels='GRAY')
        
        # 2. Stats Overlay
        stats_placeholder.markdown(f"""
        **M**: {m:.4f} | **E**: {e:.4f} | **T**: {T:.2f}
        """)
        
        # 3. Line Charts
        # Create DataFrame for Streamlit chart
        df = pd.DataFrame({
            'Magnetization': st.session_state.history['M'],
            'Energy': st.session_state.history['E']
        })
        chart_m_placeholder.line_chart(df)
        
        # 4. Phase Diagram Update
        # It's expensive to redraw matplotlib every frame.
        # Maybe skip every 10 frames?
        if st.session_state.step_count % (speed * 5) == 0:
            point.set_data([T], [m])
            phase_plot_placeholder.pyplot(fig_phase)
            
        # Throttling
        # time.sleep(0.01)
        
        # Check if button unchecked? 
        # Streamlit doesn't update 'run_btn' variable inside the loop until rerun.
        # So manual stop is usually implicitly done by interacting with UI which triggers rerun.
        # But for 'clean' experience, we just let it run.
        
else:
    # If not running, just show current state
    img_data = ((sim.grid + 1) * 127.5).astype(np.uint8)
    lattice_placeholder.image(img_data, caption=f"Step {st.session_state.step_count} (Paused)", width="stretch", output_format='PNG', clamp=True, channels='GRAY')
    
    m = abs(sim.magnetization) / (L*L)
    e = sim.energy() / (L*L)
    stats_placeholder.markdown(f"**M**: {m:.4f} | **E**: {e:.4f} | **T**: {T:.2f}")

    df = pd.DataFrame({'Magnetization': st.session_state.history['M']})
    chart_m_placeholder.line_chart(df)
    
    point.set_data([T], [m])
    phase_plot_placeholder.pyplot(fig_phase)
