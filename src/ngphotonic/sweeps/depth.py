"""Kerr depth sweeps with a matched Gaussian preparation baseline.

NG arm: S(r)|0>, then [dephasing . loss . Kerr(xi)]^D.
Target: Kerr(D xi) S(r)|0>, the noiseless output at that depth.
Baseline: optimized displaced squeezed input, then D identical noise layers, no Kerr.
All active Gaussian preparation is at the input. Intermediate passive rotations can
be absorbed into that input; intermediate pumping/squeezing and mixed Gaussian inputs
are outside the comparison family. Both arms have input energy sinh(r)^2 and the same
number of noise exposures. Dephasing may make the Gaussian-prepared output non-Gaussian.

The baseline channel collapses exactly, so its fidelity can be evaluated using a
single adjoint-channel effect. The NG evolution is always evaluated layer by layer.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from ..analysis.thresholds import last_viable_depth
from ..backends.reference import mean_photon_number, purity, squeezed_ket, tail_weight, to_dm
from ..baselines.gaussian import gaussian_ket
from ..circuits.templates import LayerSpec, apply_layer, is_reducible
from ..metrics.negativity import grid_diagnostics, wigner_log_negativity
from ..metrics.operational import target_state_fidelity
from ..metrics.wigner import phase_space_grid, wigner
from ..noise.loss import loss_kraus
from ..noise.phase_diffusion import apply_phase_diffusion
from ..optimization.gaussian_baseline import default_shell_starts, optimize_gaussian_baseline


@dataclass(frozen=True)
class DepthConfig:
    cutoff: int = 32
    squeezing: float = 0.6
    kerr_xi: float = 0.2
    epsilon: float = 0.01
    grid_limit: float = 6.0
    grid_points: int = 161
    extra_starts: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("cutoff", "grid_points", "extra_starts", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                # ValueError, not TypeError: this dataclass validates every field
                # through one channel, and callers match on ValueError.
                raise ValueError(f"{name} must be an integer.")
        if self.cutoff < 4 or self.grid_points < 3 or self.extra_starts < 0 or self.seed < 0:
            raise ValueError("Invalid cutoff, grid_points, extra_starts or seed.")
        if not np.isfinite(self.squeezing) or self.squeezing <= 0:
            raise ValueError("squeezing must be finite and positive.")
        if not np.isfinite(self.epsilon) or not 0 < self.epsilon < 1:
            raise ValueError("epsilon must lie in (0, 1).")
        if not np.isfinite(self.grid_limit) or self.grid_limit <= 0:
            raise ValueError("grid_limit must be finite and positive.")
        LayerSpec(kerr_xi=self.kerr_xi)

    @property
    def budget(self) -> float:
        return float(np.sinh(self.squeezing) ** 2)


def fidelity_effect(target: np.ndarray, eta: float, sigma_phi: float) -> np.ndarray:
    """E_loss^dag E_phase^dag(|target><target|), independent of the candidate.

    Tr[E(rho) P] = Tr[rho E^dag(P)]. Diffusion is self-adjoint; loss is not.
    This avoids applying a full channel inside every optimizer evaluation.
    """
    effect = apply_phase_diffusion(to_dm(target), sigma_phi)
    return sum(k.conj().T @ effect @ k for k in loss_kraus(eta, len(target)))


def vacuum_advantage_bound(n_budget: float, eta: float, depth: int) -> float:
    """Upper bound on advantage, using the unoptimized squeezed input as a witness.

    Kerr and dephasing preserve photon number; loss multiplies it by eta. Each arm
    has trace distance to vacuum <= sqrt(<n>), so their fidelity difference to ANY
    pure target is <= 2 sqrt(n_budget eta^D). An optimized baseline can only improve
    on this explicit feasible witness. The bound decreases for every future D and
    therefore rules out later revivals once it is <= epsilon. It certifies absence
    of advantage, not global optimality of the baseline at earlier depths.
    """
    LayerSpec(eta=eta)
    if not np.isfinite(n_budget) or n_budget < 0:
        raise ValueError("n_budget must be finite and non-negative.")
    if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 0:
        raise ValueError("depth must be a non-negative integer.")
    return float(min(1.0, 2 * np.sqrt(n_budget * eta**depth)))


def certified_horizon(n_budget: float, eta: float, epsilon: float) -> int:
    """First depth where the vacuum bound excludes all subsequent viable depths."""
    vacuum_advantage_bound(n_budget, eta, 0)
    if not np.isfinite(epsilon) or not 0 < epsilon < 1:
        raise ValueError("epsilon must lie in (0, 1).")
    if vacuum_advantage_bound(n_budget, eta, 0) <= epsilon:
        return 0
    if eta == 1:
        raise ValueError("Lossless circuits have no finite vacuum-bound horizon.")
    if eta == 0:
        return 1
    depth = max(0, int(np.ceil(np.log(epsilon**2 / (4 * n_budget)) / np.log(eta))))
    while vacuum_advantage_bound(n_budget, eta, depth) > epsilon:
        depth += 1
    return depth


def depth_summary(values: list[float], epsilon: float, *, tail_bound: float | None = None) -> dict:
    """Last viable depth, every viable interval, and explicit finite-horizon status.

    No monotonicity is assumed. ``tail_bound`` must bound all depths after the last
    sample, not just the next one. Without it a final dip cannot certify collapse.
    """
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("values must be a nonempty finite sequence sampled at every depth.")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    if tail_bound is not None and (not np.isfinite(tail_bound) or tail_bound < 0):
        raise ValueError("tail_bound must be finite and non-negative.")
    last = last_viable_depth(values, epsilon)
    intervals = []
    for d, value in enumerate(values):
        if value > epsilon:
            if not intervals or intervals[-1][1] != d - 1:
                intervals.append([d, d])
            else:
                intervals[-1][1] = d
    certified = tail_bound is not None and tail_bound <= epsilon
    return {
        "epsilon": epsilon,
        "last_viable_depth_observed": last if last >= 0 else None,
        "d_star": last if certified and last >= 0 else None,
        "status": ("resolved" if last >= 0 else "never_viable") if certified else "horizon_limited",
        "searched_through": len(values) - 1,
        "viable_intervals": intervals,
        "revivals": max(0, len(intervals) - 1),
        "tail_bound": tail_bound,
        "boundary_margin": min(abs(value - epsilon) for value in values),
    }


def evaluate_depth(
    rho: np.ndarray, target: np.ndarray, spec: LayerSpec, depth: int, config: DepthConfig,
    *, measure_wigner: bool = True,
) -> dict:
    """Score one depth, retaining a feasible squeezed-input baseline witness."""
    eta_total = spec.eta**depth
    sigma_total = spec.sigma_phi * np.sqrt(depth)
    effect = fidelity_effect(target, eta_total, sigma_total)

    def score(candidate: np.ndarray) -> float:
        return float(np.einsum("ij,ji->", candidate, effect).real)

    with warnings.catch_warnings():
        # The experiment records tails and checks cutoff convergence explicitly.
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = optimize_gaussian_baseline(
            score, config.cutoff, config.budget,
            starts=default_shell_starts(extra=config.extra_starts, seed=config.seed),
        )
        source = to_dm(squeezed_ket(config.squeezing, 0, config.cutoff))
        candidate = to_dm(gaussian_ket(baseline.params, config.cutoff))
    witness_score = score(source)
    # Retaining this candidate makes the future-depth certificate valid even if
    # every local optimization terminates unsuccessfully.
    witness_used = witness_score > baseline.score
    baseline_score = max(baseline.score, witness_score)
    if witness_used:
        candidate = source
    s_ng = target_state_fidelity(rho, target)
    lumped = apply_layer(to_dm(target), LayerSpec(eta=eta_total, sigma_phi=sigma_total))
    row = {
        "depth": depth,
        "s_ng": s_ng,
        "s_gaussian": baseline_score,
        "advantage": s_ng - baseline_score,
        "s_lumped": target_state_fidelity(lumped, target),
        "lumped_trace_distance": float(np.abs(np.linalg.eigvalsh(rho - lumped)).sum() / 2),
        "n_bar": mean_photon_number(rho),
        "purity": purity(rho),
        "trace_error": float(abs(np.trace(rho) - 1)),
        "minimum_eigenvalue": float(np.linalg.eigvalsh(rho).min()),
        "baseline_n_bar_input": mean_photon_number(candidate),
        "baseline_energy_error": abs(mean_photon_number(candidate) - config.budget),
        "baseline_tail": tail_weight(candidate),
        "baseline_params": asdict(baseline.params) if not witness_used else
            {"alpha_re": 0.0, "alpha_im": 0.0, "r": config.squeezing, "phi": 0.0},
        "baseline_n_starts": baseline.n_starts,
        "baseline_scores_by_start": baseline.scores_by_start,
        "baseline_converged_fraction": baseline.converged_fraction,
        "baseline_score_spread": baseline.score_spread,
        "baseline_witness_score": witness_score,
        "baseline_witness_used": witness_used,
        "vacuum_advantage_bound": vacuum_advantage_bound(config.budget, spec.eta, depth),
    }
    if measure_wigner:
        x, p, X, P = phase_space_grid(limit=config.grid_limit, points=config.grid_points)
        W = wigner(rho, X, P)
        row["w_log"] = wigner_log_negativity(W, x, p)
        row["wigner_diagnostics"] = grid_diagnostics(W, x, p)
    return row


def sweep_depth(
    config: DepthConfig, eta: float, sigma_phi: float, max_depth: int | None = None,
    *, progress: Callable[[dict], None] | None = None, allow_reducible: bool = False,
) -> dict:
    """Sweep EVERY integer depth through a specified or automatically certified horizon."""
    spec = LayerSpec(eta=eta, sigma_phi=sigma_phi, kerr_xi=config.kerr_xi)
    if is_reducible(spec) and not allow_reducible:
        raise ValueError("A depth result requires an irreducible Kerr/loss layer.")
    if max_depth is None:
        max_depth = certified_horizon(config.budget, eta, config.epsilon)
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        source = squeezed_ket(config.squeezing, 0, config.cutoff)
    rho = to_dm(source)
    indices_squared = np.arange(config.cutoff) ** 2
    rows = []
    for depth in range(max_depth + 1):
        if depth:
            rho = apply_layer(rho, spec)
        target = np.exp(1j * depth * spec.kerr_xi * indices_squared) * source
        row = evaluate_depth(rho, target, spec, depth, config)
        rows.append(row)
        if progress is not None:
            progress(row)
    return {
        "layer": asdict(spec),
        "input_n_budget": config.budget,
        "input_tail": tail_weight(source),
        "baseline_is_gaussian_channel": sigma_phi == 0,
        "rows": rows,
        "collapse": depth_summary(
            [r["advantage"] for r in rows], config.epsilon,
            tail_bound=vacuum_advantage_bound(config.budget, eta, max_depth + 1),
        ),
        "epsilon_sensitivity": {
            str(eps): depth_summary(
                [r["advantage"] for r in rows], eps,
                tail_bound=vacuum_advantage_bound(config.budget, eta, max_depth + 1),
            ) for eps in (0.005, 0.01, 0.02)
        },
    }
