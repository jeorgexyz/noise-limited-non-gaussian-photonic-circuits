"""Optimized, resource-matched Gaussian baseline.

RESEARCH_PLAN.md section 8. This turns the claim

    "the non-Gaussian circuit beats a Gaussian example"

into

    "the non-Gaussian circuit beats the best Gaussian circuit found under stated
     physical constraints"

which is the difference between a result and an artefact of baseline choice.

The optimization maximises an operational score over the four-parameter displaced
squeezed vacuum family, subject to an energy constraint

    ``|n_bar(G) - n_bar_budget| <= tolerance``

with ``n_bar = |alpha|^2 + sinh^2(r)`` evaluated analytically, so feasibility costs
nothing. The candidate then passes through the *same* noise channels as the circuit it
is compared against: an advantage measured against a noiseless baseline would not be an
advantage.

Two properties are enforced rather than assumed:

**Multi-start.** The objective is not concave and has a symmetry in ``phi``, so a
single local optimization can and does land in local maxima. The default is a
deterministic Sobol-like spread of starts plus a fixed set of physically-motivated
ones (vacuum, pure coherent, pure squeezed). :func:`optimize_gaussian_baseline` reports
the spread across starts, so a poorly-converged run is visible rather than silent.

**Energy matching is exact by construction.** The default ``parameterization="shell"``
works in three coordinates -- the fraction of the budget held as squeezing, the
displacement angle and the squeeze phase -- so ``n_bar`` equals the budget identically
rather than approximately. The earlier soft-penalty mode is retained as
``parameterization="free"``, but it is unsafe for any score that grows monotonically
with energy: on phase estimation the optimizer found it worthwhile to pay the penalty
and overspend the budget by about 15%, which quietly breaks the matching the comparison
depends on.

**Reported as a lower bound.** The result is the best Gaussian score *found*. It is a
lower bound on the true Gaussian optimum, which means the advantage ``A`` derived from
it is an *upper* bound. That direction is the conservative one for a negative result
("no advantage") and the optimistic one for a positive result, so positive advantages
are the ones that need the convergence evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import minimize

from ..baselines.gaussian import (
    PARAM_BOUNDS,
    GaussianParams,
    gaussian_ket,
    mean_photon_number_analytic,
)

__all__ = [
    "BaselineResult",
    "optimize_gaussian_baseline",
    "default_starts",
    "shell_to_params",
]


@dataclass
class BaselineResult:
    """Outcome of a baseline optimization."""

    score: float
    """Best score found. A lower bound on the true Gaussian optimum."""

    params: GaussianParams
    mean_photon_number: float
    n_starts: int
    scores_by_start: list[float] = field(default_factory=list)

    @property
    def score_spread(self) -> float:
        """Difference between the best and second-best distinct start.

        A large spread means the landscape is multi-modal and the multi-start was
        doing real work; a spread near zero means the starts agreed.
        """
        if len(self.scores_by_start) < 2:
            return 0.0
        ordered = sorted(self.scores_by_start, reverse=True)
        return float(ordered[0] - ordered[1])

    @property
    def converged_fraction(self) -> float:
        """Fraction of starts landing within 1e-6 of the best score."""
        if not self.scores_by_start:
            return 0.0
        best = max(self.scores_by_start)
        return float(np.mean([abs(s - best) < 1e-6 for s in self.scores_by_start]))


def shell_to_params(vector, n_budget: float) -> GaussianParams:
    """Map shell coordinates ``(s, psi, phi)`` to a state with ``n_bar == n_budget``.

    ``s`` is the fraction of the budget held as squeezing, clipped to ``[0, 1]``; ``psi``
    is the displacement angle and ``phi`` the squeeze phase. Since
    ``n_bar = |alpha|^2 + sinh^2(r)``, setting ``sinh^2(r) = s n`` and
    ``|alpha|^2 = (1 - s) n`` satisfies the constraint exactly.
    """
    s = float(np.clip(vector[0], 0.0, 1.0))
    psi, phi = float(vector[1]), float(vector[2])
    squeezed_energy = s * n_budget
    displaced_magnitude = np.sqrt(max((1.0 - s) * n_budget, 0.0))
    return GaussianParams(
        alpha_re=displaced_magnitude * np.cos(psi),
        alpha_im=displaced_magnitude * np.sin(psi),
        r=float(np.arcsinh(np.sqrt(max(squeezed_energy, 0.0)))),
        phi=phi,
    )


def default_shell_starts(extra: int = 8, seed: int = 0) -> list[np.ndarray]:
    """Starts in shell coordinates: the pure-displacement and pure-squeezing corners,
    an even split, and a deterministic spread of orientations."""
    starts = [
        np.array([0.0, 0.0, 0.0]),          # all displacement
        np.array([1.0, 0.0, 0.0]),          # all squeezing
        np.array([1.0, 0.0, np.pi / 2]),    # all squeezing, rotated
        np.array([0.5, 0.0, 0.0]),          # even split
        np.array([0.5, np.pi / 2, np.pi / 2]),
    ]
    rng = np.random.default_rng(seed)
    for _ in range(extra):
        starts.append(
            np.array([rng.uniform(0.0, 1.0),
                      rng.uniform(0.0, 2.0 * np.pi),
                      rng.uniform(0.0, 2.0 * np.pi)])
        )
    return starts


def default_starts(n_budget: float, extra: int = 12, seed: int = 0) -> list[np.ndarray]:
    """Starting points: physically-motivated ones, then a deterministic spread.

    The fixed starts put a candidate at each corner of the trade-off the energy
    constraint allows -- all energy in displacement, all in squeezing, and an even
    split -- which are the configurations that tend to be optimal for, respectively,
    coherent-like, squeezed-like and intermediate targets.
    """
    alpha_max = float(np.sqrt(max(n_budget, 0.0)))
    r_max = float(np.arcsinh(np.sqrt(max(n_budget, 0.0))))
    half = float(np.sqrt(max(n_budget, 0.0) / 2.0))
    r_half = float(np.arcsinh(np.sqrt(max(n_budget, 0.0) / 2.0)))

    starts = [
        np.array([0.0, 0.0, 0.0, 0.0]),                 # vacuum
        np.array([alpha_max, 0.0, 0.0, 0.0]),           # all displacement
        np.array([0.0, 0.0, r_max, 0.0]),               # all squeezing, phi = 0
        np.array([0.0, 0.0, r_max, np.pi / 2]),         # all squeezing, rotated
        np.array([half, 0.0, r_half, 0.0]),             # even split
        np.array([half, 0.0, r_half, np.pi / 2]),
        np.array([0.0, alpha_max, 0.0, 0.0]),           # displacement along p
    ]

    rng = np.random.default_rng(seed)
    for _ in range(extra):
        starts.append(
            np.array(
                [
                    rng.uniform(-alpha_max - 0.5, alpha_max + 0.5),
                    rng.uniform(-alpha_max - 0.5, alpha_max + 0.5),
                    rng.uniform(0.0, max(r_max, 0.1)),
                    rng.uniform(0.0, 2.0 * np.pi),
                ]
            )
        )
    return starts


def optimize_gaussian_baseline(
    score: Callable[[np.ndarray], float],
    cutoff: int,
    n_budget: float,
    energy_tolerance: float = 0.02,
    starts: Sequence[np.ndarray] | None = None,
    penalty_weight: float = 100.0,
    seed: int = 0,
    parameterization: str = "shell",
) -> BaselineResult:
    """Maximise ``score`` over energy-matched Gaussian states.

    ``score`` receives a **density matrix** -- the Gaussian candidate already passed
    through whatever noise the comparison requires -- and returns a value to maximise.
    Callers supply that noise by closing over it, which keeps this function agnostic to
    the channel being studied.

    The energy constraint is imposed as a smooth quadratic penalty outside the
    tolerance band rather than as a hard constraint, so that gradient-free local search
    is not stopped at the boundary. The returned ``mean_photon_number`` is checked
    against the budget by the caller or by the tests.
    """
    if n_budget < 0:
        raise ValueError(f"n_budget must be non-negative, got {n_budget}.")
    if parameterization not in ("shell", "free"):
        raise ValueError(f"parameterization must be 'shell' or 'free', got {parameterization!r}.")

    on_shell = parameterization == "shell"
    if starts is None:
        starts = default_shell_starts(seed=seed) if on_shell else default_starts(n_budget, seed=seed)

    # Shell coordinates are (s, psi, phi); free coordinates are (Re a, Im a, r, phi).
    # Passing one to the other silently reinterprets the vector and explores a
    # meaningless region without erroring, so the dimension is checked.
    expected = 3 if on_shell else 4
    for index, start in enumerate(starts):
        if len(start) != expected:
            raise ValueError(
                f"parameterization={parameterization!r} expects {expected}-element "
                f"starts, but start {index} has {len(start)}. Use "
                f"{'default_shell_starts()' if on_shell else 'default_starts(n_budget)'}."
            )

    def to_params(vector: np.ndarray) -> GaussianParams:
        return shell_to_params(vector, n_budget) if on_shell else GaussianParams.from_vector(vector)

    def objective(vector: np.ndarray) -> float:
        params = to_params(vector)
        try:
            rho = _pure_dm(gaussian_ket(params, cutoff))
        except ValueError:
            return 1e6
        value = -(score(rho))
        if not on_shell:
            n_bar = mean_photon_number_analytic(params)
            excess = max(0.0, abs(n_bar - n_budget) - energy_tolerance)
            value += penalty_weight * excess**2
        return value

    best_value = np.inf
    best_vector = None
    per_start = []

    for start in starts:
        outcome = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-10},
        )
        clipped = outcome.x if on_shell else _clip_to_bounds(outcome.x)
        value = objective(clipped)
        per_start.append(-value)
        if value < best_value:
            best_value, best_vector = value, clipped

    best_params = to_params(best_vector)
    best_rho = _pure_dm(gaussian_ket(best_params, cutoff))

    return BaselineResult(
        score=float(score(best_rho)),
        params=best_params,
        mean_photon_number=mean_photon_number_analytic(best_params),
        n_starts=len(starts),
        scores_by_start=[float(v) for v in per_start],
    )


def _pure_dm(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def _clip_to_bounds(vector: np.ndarray) -> np.ndarray:
    """Nelder-Mead is unbounded, so candidates are clipped back into the family."""
    clipped = np.array(vector, dtype=float)
    for index, (low, high) in enumerate(PARAM_BOUNDS):
        clipped[index] = float(np.clip(clipped[index], low, high))
    return clipped
