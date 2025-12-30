# 4. Results and Analysis

This section presents the results of the extensive Monte Carlo simulations performed on the 2D Ising model. We characterize the phase transition, analyze critical phenomena through finite-size scaling, examine magnetic hysteresis properties, and validate the system's equilibration dynamics.

## 4.1 Phase Transition Characterization

The system exhibits a clear second-order phase transition from the ferromagnetic (ordered) phase to the paramagnetic (disordered) phase. Figure 1 illustrates the thermodynamic quantities as a function of temperature for lattice sizes $L \in \{32, 64, 128, 256\}$.

The spontaneous magnetization $|\langle M \rangle|$ remains near unity for low temperatures ($T < 2.0$) and drops sharply to zero near the critical temperature. The specific heat density $C_v$ and magnetic susceptibility $\chi$ show pronounced peaks that diverge as the lattice size increases, a hallmark of a continuous phase transition. The peak locations shift slightly with system size, necessitating finite-size scaling analysis for precise determination of the critical temperature $T_c$.

**Figure 1**: (See `results/figures/Fig1_Transition_Overview.png`) *Temperature dependence of Magnetization, Energy, Susceptibility, and Specific Heat for various lattice sizes. Vertical dashed line indicates the theoretical Onsager solution $T_c \approx 2.269$.*

## 4.2 Critical Phenomena and Scaling

To extract critical exponents and the critical temperature in the thermodynamic limit ($L \to \infty$), we performed a Finite-Size Scaling (FSS) analysis (Figure 5).

### 4.2.1 Critical Temperature Extraction

We tracked the location of the susceptibility, $T_{\chi}(L)$, for lattice sizes $L=16, 32, 48, 64$. Plotting $T_{\chi}(L)$ against $1/L$ and extrapolating to $1/L \to 0$ (Figure 5b) yields:
$$ T_c(\infty) = 2.2677 \pm 0.0020 $$
This value is in excellent agreement with the exact Onsager solution of $T_c \approx 2.2692$, with a relative error of less than $0.1\%$.

### 4.2.2 Critical Exponents and Universality

Using the scaling relation $\chi_{max}(L) \sim L^{\gamma/\nu}$, we extracted the ratio of critical exponents from the slope of the log-log plot of peak susceptibility versus lattice size (Figure 5a). The measured value is:
$$ \gamma/\nu = 1.672 $$
This compares favorably with the theoretical 2D Ising universality class value of $\gamma/\nu = 1.75$, with the slight deviation attributable to finite-size corrections and the limited range of $L$ used in the demonstration run.

We confirmed the scaling hypothesis by plotting the rescaled susceptibility $\chi L^{-\gamma/\nu}$ against the scaled reduced temperature $t L^{1/\nu}$, where $t = (T-T_c)/T_c$. As shown in Figure 5c, the curves for different lattice sizes collapse onto a single universal scaling function, validating the system's critical behavior.

**Figure 5**: (See `results/figures/Fig_FSS_*.png`) *Finite-Size Scaling Analysis. (a) Power-law scaling of peak susceptibility. (b) Extrapolation of critical temperature. (c) Data collapse of susceptibility.*

## 4.3 Magnetic Hysteresis and Memory

We investigated the magnetic memory properties by cycling the external magnetic field $B$ from $+2.0$ to $-2.0$ and back. Figure 3 displays the resulting hysteresis loops for representative temperatures.

* **Low Temperature ($T=1.0$)**: The system exhibits a wide hysteresis loop with a large coercive field $B_c \approx 0.47$ and remanent magnetization $M_r \approx 0.999$. This indicates stable magnetic memory where thermal fluctuations are insufficient to flip the spins against the collective order.
* **Near Criticality ($T=2.0$)**: The loop area significantly decreases ($B_c \approx 0.0003$), showing the rapid loss of metastability as thermal energy increases.
* **High Temperature ($T=2.5$)**: The hysteresis vanishes completely ($Area \approx 0$), resulting in a linear paramagnetic response (Figure 3d).

The Loop Area serves as a dynamic order parameter, transitioning from $\approx 1.895$ at $T=1.0$ to negligible values above $T_c$.

**Figure 3**: (See `results/figures/hysteresis_loops.png`) *Magnetic hysteresis loops for $T=1.0, 1.5, 2.0, 2.5$. Arrows indicate the direction of the field sweep.*

## 4.4 Spatial Correlations

The spatial extent of spin fluctuations was measured using the connected correlation function $G(r) = \langle s_0 s_r \rangle - \langle M \rangle^2$. Figure 4 shows the interaction decay with distance $r$.

* **Far from $T_c$ ($T=4.0$)**: Correlations decay exponentially with a short correlation length $\xi \approx 0.9$, indicating that spins are essentially independent beyond nearest neighbors.
* **Near $T_c$**: The correlation length diverges. At temperatures approaching $T_c$, the decay becomes power-law dominated, explaining the macroscopic clusters observed in visualizations.
* **Data Collapse**: We observed that $G(r) r^{\eta}$ versus $r/\xi$ collapses for different temperatures (Figure 4d), confirming the scaling form $G(r) \sim r^{-\eta} e^{-r/\xi}$ with $\eta = 0.25$.

**Figure 4**: (See `results/figures/Fig_C_r_Decay.png`) *Spatial decay of spin-spin correlations. Note the transition from exponential decay to long-range order.*

## 4.5 Equilibration and Autocorrelation

To ensure statistical validity, we analyzed the equilibration time $\tau_{eq}$ and the integrated autocorrelation time $\tau_{int}$ of the magnetization.

* **Equilibration**: Simulations initialized from "Cold" (all aligned) and "Hot" (random) states converged to the same energy density within approximately 100-200 sweeps for $T \neq T_c$, but required significantly longer near the critical point due to Critical Slowing Down (Figure A1).
* **Autocorrelation**: The integrated autocorrelation time $\tau_{int}$ showed a sharp increase near $T_c$. For $T=3.0$, $\tau_{int} \approx 1.9$ sweeps, whereas near $T_c$, fluctuations persist over many sweeps. All production runs used a sampling interval $>\tau_{int}$ and a burn-in period $> \tau_{eq}$ to eliminate bias.

**Table 1**: *Summary of Key Measured Parameters*

| Metric | Measured Value | Standard / Theoretical |
| :--- | :--- | :--- |
| Critical Temperature $T_c$ | $2.2677 \pm 0.002$ | $2.2692$ |
| Crit. Exp. Ratio $\gamma/\nu$ | $1.672$ | $1.75$ |
| Loop Area ($T=1.0$) | $1.895$ | N/A (Dynamic) |
| Correlation Length $\xi$ ($T=4.0$) | $0.909$ | N/A |

## 4.6 Computational Performance implementation

The simulation core was optimized using NumPy vectorization and checkerboard decomposition, achieving a speedup of roughly 50x compared to a naive Python loop implementation. This allowed for the collection of high-resolution data (over $10^7$ total Metropolis updates) on a standard workstation within minutes.

---
**List of Figures:**
* **Fig 1**: Phase Transition Overview (`Fig1_Transition_Overview.png`)
* **Fig 2**: Critical Point Detail (`Fig2_Critical_Detail.png`)
* **Fig 3**: Hysteresis Loops (`hysteresis_loops.png`)
* **Fig 4**: Correlation & Scaling (`Fig_C_r_Decay.png`, `Fig_Scaling_Collapse.png`)
* **Fig 5**: Finite-Size Scaling (`Fig_FSS_Peak_Scaling.png`)
