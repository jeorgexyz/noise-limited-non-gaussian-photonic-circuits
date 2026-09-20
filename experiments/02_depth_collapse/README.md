# Kerr depth collapse

This experiment measures the last integer depth with target-preparation advantage
above a stated tolerance. The tracked `config.json`, `report.json` and
`controls_report.json` specify the run and retain its numerical evidence.

```bash
python experiments/02_depth_collapse/run.py
python experiments/02_depth_collapse/controls.py
```

`run.py --smoke` exercises the pipeline on a short sweep. `run.py --publish` also
refreshes the tracked report and documentation figures after validation passes;
it does not deploy a website. A smoke run cannot publish results. Runtime data and
checkpoints go to `results/02_depth_collapse/`.

## Comparison

The input is squeezed vacuum with `r = 0.6` and mean photon number
`sinh(r)^2 = 0.4053277837`. Each layer applies

```
Kerr(xi = 0.2) -> pure loss(eta) -> phase diffusion(sigma)
```

The Kerr convention is `exp(i xi n^2)`, consistent with
[Piquasso's gate definition](https://docs.piquasso.com/instructions/gates.html#piquasso.instructions.gates.Kerr).
At depth `D`, both arms are scored by fidelity to the **same** noiseless target
`Kerr(D xi) S(r)|0>`. This target changes with depth, so the curves compare fidelity
to each circuit's own intended output, rather than increasingly long implementations
of one fixed target. The separate fixed-total control uses one target throughout.

The Gaussian baseline optimizes a pure displaced squeezed input at exactly the same
analytic input energy, then subjects it to `D` identical noise exposures. It receives
no Kerr gates. Active Gaussian preparation occurs at the input; intermediate passive
rotations can be absorbed into it. Intermediate squeezing/displacement, thermal
Gaussian inputs, and adaptive controls are outside this family. Phase diffusion is
not a Gaussian channel: the baseline is Gaussian-prepared, and its noisy output need
not remain Gaussian.

The baseline noise collapses exactly to `(eta^D, sigma sqrt(D))`. Its objective uses
the adjoint channel to evaluate the post-noise fidelity without repeating channel
evolution inside the optimizer. The NG arm is propagated through every individual
layer. Tests compare the adjoint objective with direct forward evolution on complex
mixed states and compare interleaved Kerr/loss evolution with Piquasso.

The optimizer finds a **lower bound** on the optimum in its comparison family. Its
reported advantage and D* are consequently optimistic estimates. Agreement between
starts is evidence, not a proof of global optimality. A negative advantage is robust
to improving the baseline.

## Why the sweep does not stop at the first crossing

Kerr evolution can produce revivals. The report records every contiguous interval
where `A > epsilon`, using a strict inequality, and takes the last viable depth.
An arbitrary finite search cannot exclude a later revival.

Here a bound supplies a finite search horizon. Both the NG state and an explicit
feasible Gaussian witness (the same initial squeezed vacuum, without Kerr) have
mean photon number at most `n0 eta^D`. For any state, its trace distance from vacuum
is bounded by `sqrt(1 - rho_00) <= sqrt(<n>)`. The triangle inequality therefore gives

```
A(D) <= S_NG(D) - S_witness(D) <= 2 sqrt(n0 eta^D).
```

This holds for every pure target, including the changing target in this experiment.
The optimizer always retains this feasible witness. Since the bound decreases with
depth, the first depth at which it is at most epsilon excludes **all** later revivals.
At `epsilon = 0.01`, every integer depth through 44, 93 and 189 is evaluated for
`eta = 0.80, 0.90, 0.95`, respectively. The certificate concerns the tail; it does not
certify the earlier numerical Gaussian optima.

Sensitivity summaries at epsilon 0.005 and 0.02 reuse the measured curves. When the
tail bound does not support a smaller epsilon, the report explicitly uses
`horizon_limited` and leaves `d_star` null. A run with no viable depth is recorded
separately from a run whose last viable depth is zero.

## Numerical checks

- Evolve every depth again at cutoff 44, up from 32, and compare NG fidelity.
- Reoptimize boundary candidates and representative points with 16 starts and a
  different seed, both at cutoff 32 and at cutoff 44. Record score changes separately
  for optimizer starts and cutoff.
- At the listed validation depths, refine the Wigner grid from 161 to 241 points,
  then extend it from ±6 to ±7 while preserving spacing. Check normalization as well
  as negativity. Wigner negativity is not used to define operational D*.
- Check trace, positivity, actual input energy of the baseline, and the distance to
  the advantage threshold. A failed check prevents publication artifacts from being
  refreshed and makes the process return a failure code.

The report includes all start scores, chosen parameters, tails, comparison depths,
cutoff/grid errors, package versions, source hashes and runtime. The main sweep uses
eight independent starts at each point and does not carry an optimum between depths.
The numerical tolerances are in `config.json`.

## Controls

The main report compares each output with a circuit that puts the same total Kerr
operation before a single accumulated noise channel. At depth 1 the outputs agree;
at larger depth their trace distance measures the ordering effect.

`controls.py` holds total transmissivity 0.5, diffusion standard deviation 0.2 and
Kerr strength 1.2 fixed while subdividing them into 1, 2, 4, 8, 16 and 32 layers.
The Gaussian baseline, target, and input energy remain the same. It rechecks every
point at the higher cutoff and with additional starts. Thus a depth effect cannot be
attributed solely to a larger total noise or Kerr dose.

Unit tests additionally cover Kerr-free reduction, complete/no loss, Kerr strengths
that reduce to phase rotations, and the Fock input that hides Kerr dynamics entirely.
