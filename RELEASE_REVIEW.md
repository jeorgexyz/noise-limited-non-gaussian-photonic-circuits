# Public release review

The repository is ready to present as a **single-mode computational research project**
with validated resource-decay, task-comparison and Kerr-depth results. It does not yet
establish computational advantage, characterize all Gaussian comparison circuits, or
complete the resource-placement and multimode programme.

## Completed in the depth study

- Reproducible configuration, reusable depth-sweep code, CLI experiment and CI smoke run.
- Kerr → loss → dephasing evolution, with a separately optimized Gaussian preparation
  baseline experiencing the same number of noise exposures and matched input energy.
- All 658 integer-depth points across six noise configurations, with revival intervals,
  D* at epsilon 0.01, and an analytic bound excluding any later viable depth.
- Discrete depth/loss maps, Wigner diagnostics, and fixed-total-budget controls.
- Higher-cutoff NG fidelity checks at every depth; extra optimizer starts, cutoff and
  Wigner-grid checks at boundary candidates and representative points.
- Tracked numerical reports and figures, plus updated README, project page and roadmap.
- Corrected NumPy dependency floor to match use of `numpy.trapezoid`; optional Piquasso
  is loaded only when requested. Corrected malformed equations in the project page.

For input squeezing 0.6, Kerr strength 0.2 and advantage tolerance 0.01:

| Per-layer transmission | D*, loss only | D*, dephasing σ = 0.1 |
| --- | --- | --- |
| 0.80 | 2 | 2 |
| 0.90 | 6 | 6 |
| 0.95 | 14 | 10 |

At eta 0.95 without dephasing, the viable intervals are 1–3, 5–7, 9–10 and 14.
The first threshold crossing would therefore underestimate the last viable depth.

Validation on Python 3.12.3: **387 tests passed**, including Piquasso 8.0.1 comparisons;
Ruff passed; the analytic validation experiment and depth smoke experiment passed.
All six production convergence checks passed. The largest change in checked advantage
under cutoff/start refinement was **2.88e-10**, versus a minimum distance to the
threshold of **1.31e-4**. The largest Wigner grid refinement change was **1.60e-4**,
within its separately stated tolerance of 2e-4. The production sweep took about six
minutes here; runtime depends on the machine and BLAS thread settings.

These local checks do not constitute a recorded GitHub Actions run for the new patch.
Git metadata was unavailable to the sandboxed production process; the report records
null Git fields and hashes of the source files and configuration instead.

## Before announcing the release

1. Land this patch and verify both existing CI configurations (core and Piquasso).
   The depth smoke run is now part of that workflow.
2. Tag the reviewed version and archive it together with the numerical reports. Add
   the final tag/DOI to the citation when one exists. The current citation has no DOI.
3. Keep the public description scoped to the tested single-mode family and the best
   Gaussian baseline found. The depth target varies with D; the fixed-total control
   uses one common target. The README and project page now make those limits explicit.

There is no need to finish the entire research programme before sharing this version.

## Research additions, in priority order

1. **Resource placement, k*.** At fixed D, total Kerr strength and noise budget, move a
   single Kerr insertion from early to late. Compare that with several weaker Kerr
   insertions of the same total strength. Use a common intended target when comparing
   implementations and keep the Gaussian baseline matched. This directly answers
   where the non-Gaussian resource should be placed.
2. **A stronger Gaussian baseline.** Add mixed Gaussian inputs and controlled
   intermediate Gaussian gates. Define the energy budget throughout the circuit so
   late pumping cannot receive an unaccounted resource advantage. Use a global-search
   or independent-grid cross-check near the positive-advantage boundary. This would
   strengthen the current optimistic estimates most directly.
3. **Depth on the second task.** Run homodyne phase-estimation Fisher information on
   these same Kerr trajectories. Comparing preparation D* with metrology D* would
   connect the project's two central findings without introducing another resource
   family.
4. **Strength and robustness maps.** Refine eta and sigma near transitions, then vary
   Kerr strength and input squeezing. Compare ideal-performance optima with
   noise-robust optima under stated resource constraints. The present map has three
   transmission values and two diffusion strengths; it is not a dense surface.
5. **A separate simulation-resource diagnostic.** Implement a tractable Gaussian
   extent/rank proxy or a measured simulation-cost benchmark, documenting its limits.
   Neither Wigner negativity nor the task advantage currently establishes classical
   simulation difficulty.
6. **A two-mode pilot, then thermal noise.** Validate a small interferometric circuit
   before attempting 3–4 modes. Thermal loss needs its own closed-form and cutoff
   checks before adding it to the depth maps.

For a paper, I would prioritize placement, the stronger baseline, and the second-task
depth sweep, then write a methods/results manuscript with a focused bibliography.
For repository maintenance, the next useful additions are a tested dependency lock,
short contribution instructions, and moving the clearly labeled V1 prototypes into
`legacy/` with their links updated.
