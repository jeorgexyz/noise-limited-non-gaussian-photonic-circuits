"""Phase diffusion (dephasing in the Fock basis).

A random phase rotation ``U(phi) = exp(-i phi n)`` with ``phi ~ N(0, sigma^2)``,
averaged over the ensemble:

    ``E_sigma(rho) = int dphi p(phi) U(phi) rho U^dag(phi)``

Because ``(U rho U^dag)_{mn} = e^{-i phi (m - n)} rho_{mn}`` and the Gaussian
characteristic function is ``E[e^{-i phi k}] = exp(-sigma^2 k^2 / 2)``, the channel has
a closed form:

    ``rho_{mn} -> rho_{mn} exp(-sigma^2 (m - n)^2 / 2)``

so the production path needs **no Monte Carlo**. :func:`apply_phase_diffusion_sampled`
exists only to validate that closed form independently, and to give the Monte Carlo
convergence check RESEARCH_PLAN.md section 11 asks for.

Two consequences follow, and they bound what the depth sweeps can measure.

**It leaves diagonal states completely untouched.** Populations ``rho_{nn}`` are
multiplied by ``exp(0) = 1``, so the photon-number distribution, the mean photon number
and the purity of any diagonal state are exactly preserved. A Fock state is diagonal, so
**phase diffusion cannot reduce a Fock state's Wigner negativity at all**, at any
``sigma``. What it destroys is coherence between different photon numbers -- the
interference fringes of a cat state, for instance. Which noise axis matters therefore
depends on which resource is in use, which is a statement about resource choice rather
than a limitation of the model.

**It commutes with pure loss.** Every loss Kraus operator ``K_j`` has support only on
``K_j[n, n+j]``, so ``(K_j rho K_j^dag)_{nm}`` draws on ``rho_{n+j, m+j}``: the offset
``n - m`` is invariant. Phase diffusion depends on nothing but that offset. So the two
channels commute exactly, and a layered circuit built from them alone collapses to a
single ``(eta^D, D sigma^2)`` pair. Depth stops being an independent axis unless
something non-commuting -- a Kerr or cubic phase gate, or mode mixing -- sits between
the layers.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "phase_diffusion_factors",
    "apply_phase_diffusion",
    "apply_phase_diffusion_sampled",
    "compose_sigma",
]


def phase_diffusion_factors(sigma: float, cutoff: int) -> np.ndarray:
    """The ``exp(-sigma^2 (m - n)^2 / 2)`` damping matrix.

    Returned separately from :func:`apply_phase_diffusion` because the same factors
    reappear when reasoning about which coherences survive a given ``sigma``.
    """
    _check_sigma(sigma)
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}.")
    offsets = np.arange(cutoff)[:, None] - np.arange(cutoff)[None, :]
    return np.exp(-0.5 * sigma**2 * offsets.astype(float) ** 2)


def apply_phase_diffusion(rho: np.ndarray, sigma: float) -> np.ndarray:
    """Apply phase diffusion of standard deviation ``sigma`` (radians).

    Exact: this is the closed-form ensemble average, not a sample of it.
    """
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {rho.shape}.")
    return rho * phase_diffusion_factors(sigma, rho.shape[0])


def apply_phase_diffusion_sampled(
    rho: np.ndarray, sigma: float, samples: int, seed: int | None = None
) -> np.ndarray:
    """Monte Carlo estimate of the same channel, for validation only.

    Draws ``phi ~ N(0, sigma^2)`` and averages ``U(phi) rho U^dag(phi)``. Converges to
    :func:`apply_phase_diffusion` as ``1/sqrt(samples)``. Use the closed form in
    production; this exists so the closed form has something independent to be checked
    against.
    """
    rho = np.asarray(rho, dtype=complex)
    _check_sigma(sigma)
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}.")

    cutoff = rho.shape[0]
    n = np.arange(cutoff)
    rng = np.random.default_rng(seed)
    phases = rng.normal(0.0, sigma, size=samples)

    accumulated = np.zeros_like(rho)
    for phi in phases:
        rotation = np.exp(-1j * phi * n)
        accumulated += rotation[:, None] * rho * rotation.conj()[None, :]
    return accumulated / samples


def compose_sigma(*sigmas: float) -> float:
    """Standard deviation of several phase-diffusion steps applied in sequence.

    Variances add, so ``sigma_total = sqrt(sum sigma_i^2)``. This is why a depth-``D``
    chain of identical phase-diffusion layers is a single channel at ``sigma sqrt(D)``.
    """
    total = 0.0
    for sigma in sigmas:
        _check_sigma(sigma)
        total += sigma**2
    return float(np.sqrt(total))


def _check_sigma(sigma: float) -> None:
    if sigma < 0:
        raise ValueError(f"Phase diffusion sigma must be non-negative, got {sigma}.")
