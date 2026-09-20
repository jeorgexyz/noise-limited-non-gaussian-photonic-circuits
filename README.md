# Noise-Limited Photonic Circuit Depth & Non-Gaussianity Collapse

> ### Status: V1 prototype — phenomenological scaffold, **not** a physics simulation
>
> This repository currently contains a **research scaffold**, not validated numerical
> results. Every number it produces comes from an *assumed* exponential decay ansatz
>
> ```
> N_W(d) = N_0 · exp(-γ_eff · d),   γ_eff = 2.0·loss + 1.5·n_th + 0.5·(1-η_sq)
> ```
>
> whose coefficients (2.0, 1.5, 0.5) were chosen by hand. No Fock-space state is ever
> constructed, no channel is ever applied, and no Wigner function is ever computed.
>
> **Therefore:** the collapse depths, platform rankings, and phase diagrams in
> `PROJECT_SUMMARY.md`, `noise_collapse_report.json`, `sweep_results.json`, and the
> committed `.png` figures are **demonstrations that the analysis/plotting harness
> works**. They are not defensible physical predictions and must not be cited as such.
>
> V2 replaces the ansatz with real mixed-state Fock simulation on a
> [Piquasso](https://github.com/Budapest-Quantum-Computing-Group/piquasso) backend.
> See `RESEARCH_PLAN.md`. The original MrMustard integration target was archived by
> Xanadu on 2026-07-29 and is no longer viable.

## Research Overview

This project investigates how realistic noise mechanisms destroy non-Gaussian quantum advantage in continuous-variable (CV) photonic circuits. It serves as the photonic analogue of noise-adaptive depth threshold and coherence collapse studies in discrete-variable quantum computing.

## V2 Validation (in progress)

The first real simulation result in this repository. `src/ngphotonic/` now contains a
pure-NumPy mixed-state Fock backend, a pure-loss channel, and a Wigner/negativity metric
stack, validated against closed forms rather than against an assumption.

![Three-panel validation figure: simulated versus analytic integral of |W| against
transmissivity showing a kink at eta = 1/2; the residual jumping from 1e-16 to 1e-5 at
that same point; and cutoff convergence flat across Fock cutoffs 5 to
40](figures/00_validation_loss_threshold.png)

**The check.** A single photon through a pure-loss channel of transmissivity `eta`
becomes `eta|1><1| + (1-eta)|0><0|`, whose Wigner function is

```
W(x, p) = (1/pi) exp(-r^2) [2 eta r^2 - 2 eta + 1],    r^2 = x^2 + p^2
```

negative exactly where `r^2 < (2 eta - 1)/(2 eta)`. Integrating gives

```
int |W| = 4 eta exp(-(1 - 1/(2 eta))) - 1     for eta >= 1/2
int |W| = 1                                    for eta <= 1/2
```

so **Wigner negativity of a lossy single photon vanishes at exactly `eta = 1/2`**. One
closed form exercises the channel, the Wigner transform, and the negativity integral at
once. Measured against it:

| quantity | result |
| --- | --- |
| max \|simulated - analytic\| over `eta` in [0, 1] | `2.8e-05` |
| negativity threshold, located by bisection | `eta = 0.50502` (analytic `0.5`) |
| worst Gaussian-state `W_log` (must be 0) | `0.0` |
| tests passing | 160 |

The middle panel is the most informative: the residual sits at machine precision
(`1e-16`) below `eta = 1/2`, where the state is *exactly* Wigner-positive and the
integral is trivially 1, then jumps to `1e-5` the instant a kink appears in `|W|` and
trapezoidal quadrature has something to struggle with. The threshold is visible in the
error structure, not just in the curve.

### What validation caught

Two failure modes that would have silently corrupted later results:

**Truncation manufactures the resource.** By Hudson's theorem a *pure* state has a
non-negative Wigner function if and only if it is Gaussian. A truncated, renormalised
squeezed ket is not Gaussian, so it necessarily shows Wigner negativity that is pure
numerics. Measured spurious `W_log` for squeezed vacuum:

```
r = 0.6:  cutoff 20 -> 2.0e-3,  40 -> 1.0e-6,  60 -> 0
r = 1.0:  cutoff 20 -> 9.4e-2,  40 -> 4.8e-3,  80 -> 0
```

At `r = 1.0` and cutoff 20 a **Gaussian** state reports `W_log = 0.094` -- about 26% of
the genuine `0.355` of a single photon. That is large enough to be mistaken for a
physical finding. `squeezed_ket` now warns when its tail weight is too large, and the
effect is pinned by a test.

**Grid extent versus cutoff.** The displaced-parity Wigner method displaces by the full
grid extent, so it needs a far larger cutoff than the state does. Accuracy tracks
`ratio = cutoff / |alpha|^2_max`:

```
ratio    3      5      6.5     10      15
error    1e-2   2e-4   1e-7    2e-9    1e-14
```

Below ratio 8 the routine now warns rather than returning confident nonsense.

Neither failure is exotic. Both produce smooth, plausible curves. Both are exactly the
kind of thing the V1 ansatz could never have surfaced, because V1 never built a state.

### Running it

```bash
python -m pytest tests/ -q                          # 160 tests, ~48s
python experiments/00_validation/run_validation.py  # regenerates the figure above
```

### Not yet done

Piquasso backend, thermal/phase/squeezing-error channels, cubic-phase and Kerr gates,
the optimized matched Gaussian baseline, and every sweep. See `RESEARCH_PLAN.md`.

## V1 Figures

The prototype generates three figures. They are reproduced here **with the caveat that
they visualise the assumed decay law, not simulated dynamics** — and because each one
shows, fairly legibly, exactly where the model runs out.

### Non-Gaussianity collapse vs. circuit depth

![Four-panel figure: Wigner negativity decay, state purity evolution, quantum advantage
loss, and non-classical statistics, each plotted against circuit depth for six noise
scenarios](noise_collapse_plots.png)

Note the top-left panel: on a log axis every negativity curve is a perfectly straight
line. That straightness *is* the ansatz — `N_W(d) = N_0 exp(-g_eff d)` cannot produce
anything else. Real Fock-space dynamics under a pure-loss channel do not give a clean
single exponential in the negativity, because the negative volume depends non-linearly on
how the whole density matrix reshapes. Curvature, or its absence, is the diagnostic to
watch when V2 replaces this panel.

### Parameter sweeps

![Four-panel figure: Wigner negativity at depth 30 over loss and thermal photons, collapse
depth over the same plane, negativity evolution over depth and loss, and a viable
operating region boundary](experimental_parameter_sweeps.png)

The contours in the top two panels are nearly vertical — thermal occupation appears to
barely matter. That is not a physical finding either. It follows from the chosen sweep
ranges against the chosen coefficients: loss spans 0-0.1 weighted at 2.0 (so up to 0.2),
while thermal photons span 0-0.02 weighted at 1.5 (so up to 0.03). The axis ranges decide
the apparent hierarchy.

### Platform comparison (withdrawn)

![Two-panel figure: negativity decay for four photonic platforms, and a bar chart of
maximum viable circuit depth showing an identical value of 31 for every
platform](platform_comparison.png)

**This figure is retained only as a record of what not to publish.** Two things are wrong
with it. First, the ranking of the four platforms restates four input loss constants
through one invented decay law. Second — visible in the figure itself — no curve in the
left panel ever crosses the dashed 10^-2 threshold within the plotted range, yet the right
panel reports a "maximum viable circuit depth" of **31 for all four platforms**. That
identical 31 is the sweep boundary, not a measured collapse. The bar chart has no
information content.

A real platform comparison needs calibrated per-platform channel parameters and the V2
simulation. It is out of scope for V1.

## Scientific Questions

### Primary Questions
1. **At what circuit depth does Wigner negativity vanish?**
   - How does this threshold depend on specific noise sources?
   - Can we predict collapse depth from device parameters?

2. **When do non-Gaussian states become classically simulable?**
   - What is the relationship between Wigner negativity and simulation complexity?
   - Can we identify a sharp classical-quantum boundary?

3. **What is the "last viable non-Gaussian layer"?**
   - Maximum useful circuit depth for quantum advantage
   - Optimal resource allocation for noisy circuits

### Secondary Questions
4. How do different noise sources (loss vs. thermal vs. squeezing imperfection) compare in their destructive impact?
5. Can error mitigation strategies extend the collapse threshold?
6. What circuit architectures are most robust to noise?

## Technical Components

### 1. State Preparation and Characterization
- **Initial states**: Squeezed vacuum, Fock states, cat states, displaced squeezed states
- **Metrics**: Wigner negativity, Fock purity, photon statistics, phase space structure

### 2. Noise Models

#### Loss Mechanisms
```
Transmission: η = 1 - loss_per_layer
Beamsplitter model: |ψ⟩ → BS(θ)|ψ⟩|vac⟩
```

#### Thermal Noise
```
Thermal occupation: n̄_th
Channel: ρ → ε_thermal(ρ, n̄_th)
```

#### Imperfect Squeezing
```
Ideal: S(r_ideal)
Actual: S(r_ideal × η_squeezing)
```

#### Detector Inefficiency
```
Detection: η_det < 1
Dark counts: Poisson(λ_dark)
```

### 3. Circuit Architecture

**Layered Structure:**
```
Layer i: Squeeze → BeamSplit → Rotate → Thermal_Noise
         (imperfect)  (lossy)    (ideal)   (environmental)
```

**Depth Sweep:** 0 to 50+ layers

### 4. Metrics and Observables

#### Wigner Negativity
```python
N_W = ∫∫ |W(x,p)| dx dp  where W(x,p) < 0
```
- Primary indicator of non-classicality
- Necessary resource for quantum computational advantage

#### Fock State Purity
```python
Purity = Tr(ρ²) = Σ_n P(n)²
```
- Measures decoherence and mixing
- Pure state: Purity = 1
- Maximally mixed: Purity → 0

#### Photon Number Statistics
```python
⟨n⟩ = Σ_n n P(n)
Var(n) = ⟨n²⟩ - ⟨n⟩²
```
- Sub-Poissonian: Var(n) < ⟨n⟩ (quantum)
- Poissonian: Var(n) = ⟨n⟩ (coherent)
- Super-Poissonian: Var(n) > ⟨n⟩ (thermal)

#### Classical Simulability
```python
Hardness ∝ N_W × Purity
```
- Combines negativity and coherence
- Threshold: below which classical algorithms dominate

## Experimental Protocol

### Phase 1: Single Noise Source Analysis
**Objective:** Isolate effect of each noise mechanism

1. **Loss-only scenario**
   - Vary loss: 0.001 to 0.1 per layer
   - Track negativity vs. depth
   - Identify collapse threshold

2. **Thermal-only scenario**
   - Vary thermal photons: 10⁻⁴ to 10⁻²
   - Measure thermalization rate
   - Find mixing threshold

3. **Squeezing imperfection-only**
   - Vary efficiency: 0.7 to 0.99
   - Assess impact on initial negativity
   - Determine minimum viable squeezing

### Phase 2: Combined Noise Analysis
**Objective:** Understand noise interaction effects

1. **Lab-realistic parameters**
   - Loss: 0.01/layer, Thermal: 0.001, Squeezing: 0.95
   - Establish baseline collapse depth
   
2. **Parameter sweeps**
   - 2D sweeps: loss × thermal
   - Identify dominant noise source
   - Map viable operating region

3. **Platform comparison**
   - Silicon photonics
   - Silicon nitride
   - Fiber-based systems
   - Free-space optics

### Phase 3: Mitigation Strategies
**Objective:** Extend collapse threshold

1. **Adaptive circuits**
   - Depth-dependent operation selection
   - Noise-aware compilation
   
2. **Error mitigation**
   - Post-selection strategies
   - Probabilistic error cancellation
   
3. **Optimal architectures**
   - Minimize critical path depth
   - Parallel vs. serial processing

## Expected Results

### Quantitative Predictions

> These are the *hypotheses the project sets out to test*, and in V1 they are also
> what the code assumes as input. Confirming them requires the V2 Fock simulation;
> until then they are not results.

1. **Collapse Depth Scaling**
   ```
   d_collapse ∝ 1/γ_eff
   γ_eff = α·loss + β·n_thermal + δ·(1-η_squeeze)
   ```
   
2. **Negativity Decay**
   ```
   N_W(d) = N_0 exp(-γ_eff · d)
   ```

3. **Threshold Estimates (Lab-Grade)**
   - Loss = 0.01: d_collapse ≈ 20-30 layers
   - Thermal = 0.001: d_collapse ≈ 30-40 layers
   - Combined: d_collapse ≈ 15-25 layers

### Qualitative Insights

1. **Critical Transitions**
   - Sharp vs. gradual negativity loss
   - Existence of "cliff" behavior
   
2. **Noise Hierarchy**
   - Relative importance: Loss > Thermal > Squeezing
   - Platform-dependent rankings
   
3. **Practical Limits**
   - Maximum useful depth for NISQ-era photonics
   - Comparison to discrete-variable circuits

## Implementation

> **V1 (this repo):** pure NumPy/Matplotlib. The `mrmustard_integration.py` module is a
> stub — every method body is commented out. It documents *where* real simulation
> should go; it does not perform any.
>
> **V2 (planned):** Piquasso general Fock simulator, which provides the attenuator
> channel, Kerr and cubic-phase gates, squeezing, interferometers, and JAX integration
> this project needs. A thin project-owned channel/experiment layer sits on top — the
> repo should not become "scripts that invoke Piquasso."

### Code Structure

```
noise_collapse_study.py          # Framework and analysis
mrmustard_integration.py         # MrMustard interface
experiments/
  ├── single_noise_analysis.py   # Phase 1 experiments
  ├── combined_noise_study.py    # Phase 2 experiments
  └── mitigation_tests.py        # Phase 3 experiments
results/
  ├── collapse_depths.json
  ├── negativity_curves.png
  └── platform_comparison.pdf
```

### Key Functions

```python
# State preparation
state = create_initial_state('squeezed', r=1.0)

# Noisy layer
state = apply_layer(state, noise_params)

# Metrics
negativity = compute_wigner_negativity(state)
purity = compute_fock_purity(state)

# Analysis
collapse_depth = find_collapse_threshold(negativity_history)
```

### Computational Requirements

- **Fock cutoff**: 20-30 (depends on photon number)
- **Wigner grid**: 100×100 points in phase space
- **Depth range**: 0-50 layers
- **Scenarios**: 6-10 different noise configurations
- **Runtime**: ~30 minutes per full analysis (estimated)

## Validation and Benchmarking

### Analytical Checks
1. Gaussian limit (no non-linearity): negativity = 0
2. Unitary evolution (no noise): negativity constant
3. Known results: compare to literature values

### Numerical Checks
1. Convergence: vary Fock cutoff
2. Resolution: vary Wigner grid density
3. Reproducibility: multiple random seeds

### Experimental Comparison
Compare to published data:
- Squeezed state decoherence (Vahlbruch et al.)
- Photon loss in waveguides (Matthews et al.)
- Cat state lifetime (Ourjoumtsev et al.)

## Extensions and Future Work

### Immediate Extensions
1. Multi-mode entanglement collapse
2. Continuous-variable graph states
3. GKP (Gottesman-Kitaev-Preskill) states

### Advanced Topics
1. Fault-tolerance threshold for CV codes
2. Noise-adaptive circuit compilation
3. Resource theory of non-Gaussianity

### Cross-Connections
**To your previous work:**
- Noise-adaptive depth optimization
- Coherence timescale analysis
- Circuit architecture comparison

**Intended contributions (V2, not yet established):**
- Separation of *resource survival* from *task usefulness* from *classical simulation
  difficulty* — Wigner negativity can persist after operational advantage is gone
- Collapse boundary `D*(ν) = max{D : A(D,ν) > ε}` against an **optimized, resource-matched**
  Gaussian baseline, not an arbitrary weak one
- Optimal non-Gaussian resource placement `k*(D,ν)` at fixed depth

None of these are demonstrated by the V1 code in this repository.

## References and Resources

### Key Papers
1. Wigner negativity: Kenfack & Życzkowski, J. Opt. B (2004)
2. CV noise models: Braunstein & van Loock, Rev. Mod. Phys. (2005)
3. Photonic loss: Carolan et al., Science (2015)

### Simulation Backends
- **Piquasso** (V2 target): https://github.com/Budapest-Quantum-Computing-Group/piquasso
- ~~MrMustard~~ — archived by Xanadu 2026-07-29, no longer maintained
- QuTiP — general quantum toolbox, useful for cross-checking single-mode channels

### Related Projects
- Strawberry Fields (Xanadu's photonic framework)
- Prior noise-adaptive depth-threshold work (discrete-variable analogue)

## Getting Started

### Installation
```bash
# Create environment
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies (V1 needs only the analysis stack)
pip install -e .

# V2 simulation backend, once that work lands
pip install -e ".[sim]"
```

### Quick Start
```bash
# Run basic analysis
python noise_collapse_study.py

# Generate all plots
python -c "from noise_collapse_study import main; main()"

# Note: mrmustard_integration.py is a stub and produces no simulation output.
```

### Next Steps (V2)
1. Stand up `src/ngphotonic/` with a Piquasso-backed Fock simulator
2. Validate against the analytic single-photon pure-loss channel:
   `|1><1| -> η|1><1| + (1-η)|0><0|`
3. Add Wigner logarithmic negativity `W_log(ρ) = log ∫|W_ρ(q,p)| dq dp`
4. Build the optimized, resource-matched Gaussian baseline
5. Run depth and resource-placement sweeps with cutoff-convergence checks
6. Only then revisit any claim about collapse depths

---

**Scope note.** Fock-space dimension grows combinatorially with mode count and cutoff
(4 modes at cutoff 20 is ~8,855 pure amplitudes and ~78M density-matrix elements), so
this is deliberately a **1–4 mode** study, not a general multimode simulator. Every
headline result must pass a cutoff-convergence check `|M_{c+Δc} - M_c| < ε`.
