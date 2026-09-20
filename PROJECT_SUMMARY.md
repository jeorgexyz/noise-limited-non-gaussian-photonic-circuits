# Project Summary: Noise-Limited Photonic Circuit Depth (V1 prototype)

> **Read this first.** Everything below describes a *phenomenological scaffold*.
> The numbers are outputs of an assumed decay law, not of a quantum simulation.
> See the status banner in `README.md`.

## What We Built

A scaffold for studying how realistic noise destroys non-Gaussian quantum advantage in
photonic circuits: a scenario-comparison harness, a parameter-sweep driver, and a
plotting/JSON-export layer. The *physics* underneath is a hand-parameterised exponential
decay, deliberately placeholder, to be replaced in V2 by mixed-state Fock simulation.

## Key Files Created

### Core Framework
1. **noise_collapse_study.py** - Main analysis framework
   - `NoiseParameters`: Physical noise model
   - `CircuitMetrics`: Track negativity, purity, simulability
   - `WignerNegativityTracker`: Depth-dependent evolution
   - `NoiseComparison`: Multi-scenario analysis
   - 6 standard scenarios from near-ideal to worst-case

2. **mrmustard_integration.py** - MrMustard interface (ready for integration)
   - `PhotonicCircuitSimulator`: High-fidelity CV simulations
   - Wigner function computation
   - Layered circuit construction
   - Platform-specific noise models

3. **experimental_analysis.py** - Advanced parameter sweeps
   - 2D sweeps: loss × thermal, depth × loss
   - Platform comparisons (SiN, Si, Fiber, Free-space)
   - Boundary identification

### Documentation
4. **README.md** - Complete research roadmap
   - Scientific questions
   - Technical protocol
   - Expected results
   - Integration guide

## Framework Demonstration Output

> These are **not physical results.** Each "collapse depth" below is simply
> `log(N_0 / 0.01) / g_eff` evaluated for a different choice of the input noise floats,
> where `g_eff = 2.0*loss + 1.5*n_th + 0.5*(1 - eta_sq)` is an assumption, not a
> derivation. They demonstrate that the sweep and reporting machinery runs end to end.
> Nothing more.

### From noise_collapse_study.py

**Collapse depths produced by the ansatz:**
- Near-Ideal: >50 layers (loss=0.001, thermal=0.0001)
- Lab-Grade: >50 layers (realistic parameters)
- Moderate Loss: 37 layers (loss=0.05)
- Poor Squeezing: 38 layers (efficiency=0.80)
- Worst-Case: 13 layers (all noise high)

**Apparent ordering:** loss dominates, then squeezing imperfection. Note this ordering is
*built into* the coefficients `(2.0, 1.5, 0.5)` rather than discovered - it restates the
input assumption. Testing whether real CV dynamics reproduce it is a V2 objective.

**Files:**
- `noise_collapse_plots.png` - 4-panel comparison
- `noise_collapse_report.json` - Detailed metrics

### From experimental_analysis.py

**2D Parameter Sweeps:**
- Loss × Thermal noise mapping
- Depth × Loss evolution
- Viable operating region boundaries
- Critical transition identification

**Platform comparison - withdrawn as a result.**
The earlier ranking (fiber > SiN > free-space > silicon) was produced by substituting four
different loss floats into the same invented decay constant. It carries no information
about real hardware beyond "lower assumed loss ranks higher," and should not be quoted.
Worse, `platform_comparison.png` reports a "maximum viable circuit depth" of **31 for all
four platforms simultaneously**, while no negativity curve in its companion panel ever
crosses the 10^-2 threshold in the plotted range. That 31 is the sweep boundary, not a
measured collapse, so the bar chart carries no information at all.

A meaningful platform comparison requires calibrated per-platform channel parameters plus
the V2 simulation, and is out of scope for V1.

**Files:**
- `experimental_parameter_sweeps.png` - 4-panel 2D sweeps
- `platform_comparison.png` - Platform rankings
- `sweep_results.json` - Raw sweep data

## Framework Capabilities

> Framing note: these are things the *code can express*, not findings it has established.

### 1. Threshold extraction machinery
Given any negativity-vs-depth curve, extracts a collapse depth against a threshold.
In V1 the curve is supplied by the ansatz:
```
N_W(d) = N_0 exp(-γ_eff · d)
γ_eff = 2.0·loss + 1.5·n_thermal + 0.5·(1-η_squeeze)
```

### 2. Noise-hierarchy parameterisation
The relative weights of loss / thermal / squeezing error are **inputs** (2.0 / 1.5 / 0.5),
not measured quantities. V2 must derive the effective hierarchy from actual channels.

### 3. Per-scenario sweep configuration
Accepts arbitrary noise-parameter scenarios and runs them through a common pipeline.

### 4. 2D map rendering
Generates 2D diagrams showing:
- Quantum advantage region
- Classical simulability boundary
- Critical transition lines

## Connection to Your Research

This project directly extends your work on:

1. **Noise-Adaptive Depth Thresholds**
   - CV analogue of discrete-variable depth limits
   - Similar exponential decay with noise
   - Platform-specific optimization strategies

2. **Coherence Collapse Studies**
   - Wigner negativity = CV coherence measure
   - Tracks decoherence through circuit
   - Identifies "last viable layer"

3. **Resource-Efficient Quantum Algorithms**
   - Informs CV algorithm design
   - Guides error mitigation strategies
   - Optimizes resource allocation

## Next Steps: V2 on Piquasso

MrMustard was archived by Xanadu on 2026-07-29, so `mrmustard_integration.py` is retained
only as a record of the intended interface. V2 targets **Piquasso**, whose general Fock
simulator already provides the attenuator channel, Kerr and cubic-phase gates, squeezing,
interferometers, and JAX integration.

```bash
pip install -e ".[sim]"
```

The V1 stub is not a drop-in starting point: it has no working method bodies. The V2
package is built fresh under `src/ngphotonic/`, with the prototype scripts moved aside.

**First milestone is validation, not results:** reproduce the analytic single-photon
pure-loss channel `|1><1|  ->  eta*|1><1| + (1-eta)*|0><0|`, confirm trace preservation
and positivity, and establish cutoff convergence before computing anything
headline-worthy.

## Research Extensions

### Immediate
1. **Multi-mode entanglement** - How does noise affect CV cluster states?
2. **GKP codes** - Fault-tolerance thresholds for bosonic codes
3. **Error mitigation** - Extend negativity with post-selection

### Advanced
1. **Adaptive compilation** - Noise-aware circuit optimization
2. **Resource theory** - Quantify non-Gaussian resource consumption
3. **Experimental validation** - Compare to real device data

## How to Use This Project

### Run Basic Analysis
```bash
python noise_collapse_study.py
# Generates: noise_collapse_plots.png, noise_collapse_report.json
```

### Run Parameter Sweeps
```bash
python experimental_analysis.py
# Generates: experimental_parameter_sweeps.png, platform_comparison.png
```

### Customize Scenarios
```python
from noise_collapse_study import NoiseParameters, NoiseComparison

comparison = NoiseComparison()
comparison.add_scenario("My Scenario", NoiseParameters(
    loss_per_layer=0.02,
    thermal_photons=0.005,
    squeezing_imperfection=0.88,
    detector_efficiency=0.92
))
comparison.run_comparison(max_depth=40)
comparison.plot_comparison(save_path='my_results.png')
```

### Simulation interface (stub - does not run)
`mrmustard_integration.py` exposes `PhotonicCircuitSimulator`, but every method body is
commented out. Calling it yields no simulation. It is kept as interface documentation.

## Publication Status

**Not publishable in its current form, and not close.** The results are generated by an
assumed decay model, so they cannot support claims about CV circuit depth limits, hardware
selection, or noise hierarchies.

What would be publishable is the V2 programme: mapping the boundary between *nominal*
non-Gaussianity and *operationally useful* non-Gaussianity under distributed realistic
noise, depth, and resource placement - against an optimized resource-matched Gaussian
baseline. That responds directly to recent work showing Wigner negativity alone is not
sufficient evidence of computational advantage, and it needs real simulation first.

## Code Quality Features

- **Modular design**: scenario/sweep/plot layers are separable and reusable in V2
- **Docstrings and type hints** throughout
- **JSON exports**: results are machine-readable
- **Reproducible figure generation** from a single entry point
- **Caveat**: the physics layer is a placeholder; see the README status banner

## Contact & Collaboration

This framework connects:
- Your noise-adaptive algorithm expertise
- CV quantum information theory
- Realistic photonic device physics

Perfect for collaboration on:
- NISQ-era photonic quantum computing
- Error mitigation for CV systems
- Hardware-software co-design

---

**Built:** February 2026  
**Framework:** Python 3.12 + NumPy/Matplotlib  
**Physics model:** phenomenological exponential ansatz (placeholder)  
**Next:** V2 - Piquasso-backed mixed-state Fock simulation  
**Status:** V1 prototype scaffold - runs, plots, and exports; results are illustrative, not physical
