# Research Plan — V2

**Working title:** Noise-induced collapse surfaces for operational non-Gaussianity in
finite-depth photonic circuits.

This document defines what V2 is, and what separates it from the V1 prototype in this
repository. V1 assumes an exponential decay law and plots its consequences. V2 simulates
the dynamics and measures the decay.

---

## 1. Central question

At what combinations of circuit depth, loss, thermal noise, phase noise, squeezing error,
and non-Gaussian resource *placement* does a non-Gaussian photonic circuit cease to
provide an operational advantage over a **resource-matched** Gaussian circuit?

## 2. The research object

For a depth-`D` circuit, evolution is

```
rho_{l+1} = N_l [ U_l(theta_l) rho_l U_l^dag(theta_l) ]
```

where `U_l` contains Gaussian or non-Gaussian operations and `N_l` is a physical
imperfection channel.

The framing commitment of this project is that three things are measured **separately**
and are not assumed to coincide:

```
resource survival  !=  task usefulness  !=  classical simulation difficulty
```

This is what distinguishes the work from plotting Wigner functions against loss. Recent
results indicate Wigner negativity can persist after it stops implying useful
computational resourcefulness, so a result of the form

```
W_log > 0   AND   A <= 0
```

— negativity intact, measurable task advantage gone — is a specific outcome to look for,
not a failure mode.

## 3. Key quantities

Wigner logarithmic negativity, an established non-Gaussian resource monotone:

```
W_log(rho) = log ∫ |W_rho(q,p)| dq dp
```

Advantage against the optimized Gaussian baseline, for noise configuration `nu`:

```
A(D, nu) = S_NG(D, nu) - S_G*(D, nu)
```

Collapse boundary — the rigorous form of the old "last viable non-Gaussian layer":

```
D*(nu) = max{ D : A(D, nu) > epsilon }
```

Optimal resource placement at fixed depth, and optimal resource strength:

```
k*(D, nu) = argmax_k A(D, nu, k)
gamma*(D, nu)   or   xi*(D, nu)
```

## 4. Simulator

**Backend: Piquasso** (`piquasso[jax]`). MrMustard was archived by Xanadu on 2026-07-29
and is not a viable target. Piquasso's general Fock simulator provides the attenuator
channel, Kerr and cubic-phase gates, beam splitters, squeezing, interferometers, photon
creation/annihilation, and JAX integration.

A thin project-owned channel/experiment layer sits on top. The repository must not become
"scripts that invoke Piquasso."

| Layer | V1 implementation |
| --- | --- |
| Gaussian operations | displacement, squeezing, rotation, beam splitter, interferometer |
| Non-Gaussian states | Fock, cat, photon-subtracted |
| Non-Gaussian gates | cubic phase, Kerr |
| Loss | pure-loss attenuator |
| Thermal loss | lossy coupling to thermal environment |
| Phase noise | stochastic phase diffusion |
| Squeezing error | `r -> r + delta_r` |
| Detector inefficiency | loss preceding ideal detector |
| Measurements | PNR and homodyne where appropriate |
| Simulation | mixed-state Fock representation |
| Optimization | JAX / SciPy |

## 5. Circuit families

Three, not twenty.

**Single-mode resource survival** — establishes clean noise thresholds.

```
|0> -> S(r) -> U_NG -> [N . G]^D
```

**Two-mode interferometric** — introduces resource spreading, correlations, mode mixing.

```
S_1 S_2 -> BS -> U_NG -> [N . U_MZI]^D
```

**Resource placement** — hold total depth `D` fixed, move the non-Gaussian operation,
sweep `k`:

```
G_1 ... G_k  U_NG  G_{k+1} ... G_D
```

This turns the project from "how fast does non-Gaussianity die?" into "where should
expensive non-Gaussian resources actually be placed?"

## 6. Noise model

Loss first: it dominates most photonic systems and has clean analytic checks. Do not
combine every imperfection at once. The progression is:

```
eta  ->  (eta, n_th)  ->  (eta, n_th, sigma_phi)  ->  (eta, n_th, sigma_phi, sigma_r)
```

Each step raises the dimension of the collapse surface.

## 7. Metrics

| Class | Metric |
| --- | --- |
| Non-Gaussian resource | Wigner logarithmic negativity |
| State quality | fidelity to ideal noiseless output |
| Energy | mean photon number |
| Decoherence | purity |
| Computational resource | Gaussian extent / rank, on tractable cases |
| Output behavior | total variation distance / distribution fidelity |
| Task utility | task-specific performance |
| Robustness | `D*`, `eta*`, collapse boundaries |
| Cost | Fock cutoff, runtime, memory |

Gaussian rank/extent matters because it measures something fundamentally different from
negativity, giving the study a genuine second axis.

## 8. The Gaussian baseline

Non-negotiable. Comparing against vacuum or a deliberately weak Gaussian circuit and
calling the difference "advantage" is the main way this kind of study goes wrong.

For each experiment the Gaussian baseline is **optimized under matched constraints**:
comparable mean photon number `n_G ~ n_NG`, same mode count, comparable depth, same
measurement family, comparable parameter budget. Then optimize squeezing, displacement,
rotations, and interferometer parameters.

This converts *"non-Gaussian circuit beats a Gaussian example"* into *"non-Gaussian
circuit beats the best Gaussian circuit found under defined physical constraints."*

## 9. Operational benchmarks

1. **Target-state preparation** — target fidelity and success probability. Cleanest test
   for cubic-phase, cat, and photon-subtracted resources. *(required)*
2. **Phase estimation** — classical Fisher information under an explicit measurement. A
   genuine operational quantity rather than another state-space metric. *(required)*
3. **Sampling** — how closely the noisy circuit preserves an ideal photon-count
   distribution, by total variation distance. *(later)*

Running more than one lets the project detect whether different definitions of "useful
non-Gaussianity" collapse at different depths.

## 10. Experimental programme

1. ~~Validate: analytic single-photon loss, noiseless Gaussian circuits, normalization,
   cutoff convergence~~ **done** -- see the V2 validation section of README.md. Also
   established: the Piquasso backend and its cross-validation against the reference,
   and the standing caveat that cross-backend agreement is not evidence where both
   backends share an algorithm (cubic phase).
2. ~~Non-Gaussian resource decay under pure loss~~ **done** -- experiment 01.
   Critical transmissivity per resource, reported with the survival threshold epsilon
   it depends on.
3. Depth sweep, identify `D*`
4. Collapse maps: `(D, eta)`, then `(D, eta, n_th)`
5. Resource-strength sweep: cubic-phase `gamma`, Kerr `xi`, cat amplitude `alpha`, `r`
6. Early vs. middle vs. late insertion of the non-Gaussian resource
7. One strong non-Gaussian layer vs. several weak ones
8. ~~Optimize Gaussian baselines~~ **done for the single-mode preparation task** --
   experiment 05. Energy-matched displaced-squeezed family, multi-start Nelder-Mead,
   validated against a brute-force grid. Taken out of order because it is what makes
   `A(D, nu)` well defined; the depth sweep (step 3) builds on it.
9. Optimize the non-Gaussian circuit for **robustness** rather than ideal performance
10. Scale 1 -> 2 -> 3-4 modes

Step 9 matters most: ask whether the ideal circuit and the noise-optimal circuit are
actually different. They are expected to be.

## 11. Numerical validation

Fock truncation can contribute to measured quantities, so this is not optional.

Pure-state dimension scales combinatorially:

| modes | cutoff | pure-state dim | density-matrix elements |
| --- | --- | --- | --- |
| 2 | 20 | 210 | 44,100 |
| 4 | 15 | 3,060 | 9.36M |
| 4 | 20 | 8,855 | 78.4M |
| 6 | 15 | 38,760 | 1.50B |

Hence: **a 1-4 mode research project**, not a general multimode simulator.

Every headline result requires cutoff convergence `|M_{c+dc} - M_c| < epsilon`. Also test
Wigner-grid resolution, Monte Carlo convergence for stochastic noise, trace preservation,
positivity, and seed sensitivity.

Primary unit test — the analytic single-photon pure-loss channel:

```
|1><1|  ->  eta |1><1| + (1 - eta) |0><0|
```

This checks the channel implementation and the Wigner calculation against a closed form.

## 12. Scope split

**V2 scope:** Piquasso backend, pure loss, phase diffusion, cubic phase / Kerr / photon
subtraction, Wigner logarithmic negativity, matched Gaussian baseline, depth and
resource-placement sweeps.

**Deferred to V3, so they do not contaminate the initial implementation:** thermal noise,
full operational task suite, robust optimization, multimode scaling.

## 13. Target repository layout

```
src/ngphotonic/
  circuits/     gaussian.py  non_gaussian.py  templates.py  placement.py
  noise/        loss.py  thermal_loss.py  phase_diffusion.py  squeezing_error.py  detection.py
  backends/     piquasso.py  reference.py
  metrics/      wigner.py  negativity.py  fidelity.py  gaussian_extent.py  operational.py
  baselines/    gaussian.py
  optimization/ gaussian_baseline.py  robust_design.py
  sweeps/       depth.py  noise.py  placement.py
  analysis/     thresholds.py  convergence.py  statistics.py
  visualization/ wigner.py  phase_diagrams.py

experiments/  00_validation  01_loss_threshold  02_depth_collapse  03_multinoise_surface
              04_resource_placement  05_gaussian_baseline  06_robust_optimization
              07_multimode_scaling

configs/  tests/  notebooks/  results/  figures/  paper/
```

Notebooks stay secondary. Experiments run from reproducible config files and scripts.

Natural dataset shape is `M[D, eta, n_th, sigma_phi, r, gamma, k, seed]` — an xarray/zarr
problem, not a CSV table.

## 14. Stack

Python 3.12, `uv` for environments.

Core: `piquasso[jax]`, `numpy`, `scipy`, `jax`, `matplotlib`, `pandas`, `xarray`, `zarr`,
`tqdm`, `pydantic`. Dev: `pytest`, `pytest-cov`, `ruff`, `mypy`.

## 15. Disposition of V1

The prototype scripts (`noise_collapse_study.py`, `experimental_analysis.py`,
`mrmustard_integration.py`) move to `legacy/` when the V2 package lands. They are retained
for the scenario/sweep/plot structure, which is reusable, and as a record of the framing.
Their numerical output is superseded and must not be carried forward into V2 results.
