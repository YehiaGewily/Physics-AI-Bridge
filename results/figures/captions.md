# Figure Captions for 2D Ising Model Simulation Results

## Phase Transition & Thermodynamics

**Figure 1** (`Fig1_Transition_Overview.png`)
**Thermodynamic Order Parameters across the Phase Transition.**
(a) Total Magnetization $|M|$ vs. Temperature $T$. The system transitions from an ordered ferromagnetic state ($M \approx 1$) to a disordered paramagnetic state ($M \approx 0$) near the critical temperature $T_c \approx 2.27$.
(b) Energy per spin $E$ vs. $T$, showing a continuous increase in internal energy.
(c) Magnetic Susceptibility $\chi$ and (d) Specific Heat $C_v$ vs. $T$. Both response functions exhibit sharp peaks that diverge with increasing lattice size $L$, characteristic of a second-order phase transition.

**Figure 2** (`Fig2_Critical_Detail.png`)
**Critical Divergence near $\mathbf{T_c}$.**
Detailed view of the Susceptibility $\chi$ (left axis, blue) and Specific Heat $C_v$ (right axis, red) in the critical region $T \in [2.0, 2.5]$. The vertical dashed line marks the Onsager solution $T_c = 2.269$. The peaks align closely with the theoretical prediction, with slight finite-size scaling shifts.

## Finite-Size Scaling (FSS)

**Figure 5** (`Fig_FSS_Peak_Scaling.png`)
**Finite-Size Scaling of Peak Susceptibility.**
Log-log plot of the maximum susceptibility $\chi_{max}(L)$ versus lattice size $L$. The solid red line represents the power-law fit $\chi_{max} \sim L^{\gamma/\nu}$. The extracted slope $\gamma/\nu \approx 1.67$ is in close agreement with the exact 2D Ising exponent of $1.75$.

**Figure 6** (`Fig_FSS_Tc_Scaling.png`)
**Extrapolation of Critical Temperature.**
The pseudo-critical temperature $T_{\chi}(L)$ (defined by the peak of susceptibility) plotted against inverse lattice size $1/L$. The linear extrapolation to the thermodynamic limit ($1/L \to 0$) yields $T_c(\infty) = 2.2677 \pm 0.002$, matching the Onsager value ($2.2692$).

**Figure 7** (`Fig_FSS_Chi_Collapse.png`)
**Universal Scaling Data Collapse.**
Rescaled susceptibility $\chi L^{-\gamma/\nu}$ plotted against the scaled reduced temperature $t L^{1/\nu}$ (where $t = (T-T_c)/T_c$), for lattice sizes $L=16, 32, 48, 64$. The collapse of all distinct data points onto a single master curve validates the scaling hypothesis and the universality class of the simulated system.

## Magnetic Hysteresis

**Figure 3** (`hysteresis_loops.png`)
**Magnetic Hysteresis Loops.**
Magnetization $M$ vs. External Field $B$ cycles for temperatures $T=1.0, 1.5, 2.0, 2.5$.
At low temperatures ($T=1.0$, blue), the system exhibits a wide hysteresis loop with high coercivity, indicating stable magnetic memory. As $T$ approaches $T_c$ ($T=2.0$, green), the loop area shrinks significantly. Above $T_c$ ($T=2.5$, red), the hysteresis vanishes, and the response becomes linear (paramagnetic).

## Correlation & Dynamics

**Figure 4** (`Fig_C_r_Decay.png`)
**Spatial Correlation Decay.**
The spin-spin correlation function $G(r) = \langle s_0 s_r \rangle - \langle M \rangle^2$ as a function of distance $r$ for various temperatures. At high temperatures ($T=4.0$), correlations decay exponentially (linear on log-linear plot). Near $T_c$ ($T \approx 2.27$), the decay becomes slower (power-law), indicating the emergence of long-range order.

**Figure 8** (`Fig_Equilibration_Curves.png`)
**Thermal Equilibration Dynamics.**
Time evolution of Magnetization (left) and Energy (right) for Cold (all aligned) and Hot (random) initial conditions. At $T=2.269$ (critical point), the two initial conditions take significantly longer to converge compared to the off-critical temperatures, demonstrating Critical Slowing Down.

**Figure 9** (`Fig_Autocorrelation_Functions.png`)
**Integrated Autocorrelation Time.**
(Top) Normalized autocorrelation function $A(t)$ of magnetization vs. time lag (sweeps).
(Bottom) The integrated autocorrelation time $\tau$ as a function of temperature. The sharp peak near $T_c$ confirms the divergence of relaxation times, necessitating longer simulations in the critical region.
