"""Homodyne detection and phase-estimation Fisher information.

The second operational task of RESEARCH_PLAN.md section 9. Photon-number measurement is
useless for phase estimation on several of the states this project cares about, so the
measurement here is homodyne: a quadrature ``x`` is measured and the phase is inferred
from the shape of ``P(x | theta)``.

Position wavefunctions in the project's ``hbar = 1`` convention, where
``x = (a + a^dag)/sqrt(2)``:

    ``<x|n> = pi^{-1/4} (2^n n!)^{-1/2} H_n(x) exp(-x^2 / 2)``

so ``P(x) = <x|rho|x> = sum_{mn} psi_m(x) rho_{mn} psi_n(x)`` with real ``psi``.

A fact that governs the whole comparison: **a Fock state is invariant under phase
rotation.** ``exp(-i theta n) |n><n| exp(i theta n) = |n><n|`` exactly, so a Fock probe
carries *no* phase information under any measurement at all, and its Fisher information
is identically zero. This is not a limitation of homodyne detection; it is a property of
the state. Any phase-estimation comparison involving Fock probes therefore has a
degenerate non-Gaussian arm, which is precisely why phase-sensitive non-Gaussian probes
such as cat states have to be included for the comparison to say anything.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.special import gammaln

from .operational import photon_number_distribution  # noqa: F401  (re-export context)

__all__ = [
    "fock_wavefunctions",
    "homodyne_distribution",
    "homodyne_fisher_information",
    "phase_rotation",
    "quadrature_grid",
    "quantum_fisher_information_pure",
]


def quadrature_grid(limit: float = 8.0, points: int = 1201) -> np.ndarray:
    """Quadrature sample points for the homodyne integral.

    The grid must contain the state's support; ``homodyne_distribution`` exposes the
    normalisation so a truncated grid is detectable rather than silent.
    """
    return np.linspace(-limit, limit, points)


def fock_wavefunctions(cutoff: int, x: np.ndarray) -> np.ndarray:
    """``psi_n(x)`` for ``n = 0 .. cutoff-1``, shape ``(cutoff, len(x))``.

    Built by the Hermite recurrence on the *normalised* functions rather than by
    evaluating ``H_n`` and dividing by ``sqrt(2^n n!)`` separately: the latter overflows
    for moderate ``n`` while the ratio stays order one.
    """
    x = np.asarray(x, dtype=float)
    psi = np.empty((cutoff, x.size), dtype=float)

    psi[0] = np.pi ** (-0.25) * np.exp(-0.5 * x**2)
    if cutoff > 1:
        psi[1] = np.sqrt(2.0) * x * psi[0]
    for n in range(2, cutoff):
        # psi_n = sqrt(2/n) x psi_{n-1} - sqrt((n-1)/n) psi_{n-2}
        psi[n] = np.sqrt(2.0 / n) * x * psi[n - 1] - np.sqrt((n - 1) / n) * psi[n - 2]
    return psi


def homodyne_distribution(
    rho: np.ndarray, x: np.ndarray, psi: np.ndarray | None = None
) -> np.ndarray:
    """``P(x) = <x|rho|x>``, clipped at zero.

    Passing a precomputed ``psi`` avoids rebuilding the wavefunctions inside a sweep,
    which dominates the cost otherwise.
    """
    rho = np.asarray(rho, dtype=complex)
    if psi is None:
        psi = fock_wavefunctions(rho.shape[0], x)
    # sum_{mn} psi_m rho_mn psi_n, evaluated as a quadratic form per grid point.
    probability = np.einsum("mi,mn,ni->i", psi, np.real(rho), psi, optimize=True)
    return np.clip(probability, 0.0, None)


def phase_rotation(theta: float, cutoff: int) -> np.ndarray:
    """``U(theta) = exp(-i theta n)`` as a diagonal unitary."""
    return np.diag(np.exp(-1j * theta * np.arange(cutoff)))


def homodyne_fisher_information(
    rho_at: Callable[[float], np.ndarray],
    theta: float,
    x: np.ndarray,
    delta: float = 1e-3,
    floor: float = 1e-10,
) -> float:
    """Classical Fisher information for ``theta`` under homodyne detection.

        ``F(theta) = int dx (dP(x|theta)/dtheta)^2 / P(x|theta)``

    ``rho_at`` maps a phase to the output state, so the caller controls what the phase
    acts on and what noise follows it.

    Sample points where ``P`` falls below ``floor`` are dropped. They contribute nothing
    in exact arithmetic but dominate the sum numerically, since a finite-difference
    derivative divided by a near-zero probability is unstable; the tails of a Gaussian
    wavefunction reach ``1e-30`` well inside the grid.
    """
    cutoff = rho_at(theta).shape[0]
    psi = fock_wavefunctions(cutoff, x)

    probability = homodyne_distribution(rho_at(theta), x, psi)
    plus = homodyne_distribution(rho_at(theta + delta), x, psi)
    minus = homodyne_distribution(rho_at(theta - delta), x, psi)

    derivative = (plus - minus) / (2.0 * delta)
    significant = probability > floor
    integrand = np.zeros_like(probability)
    integrand[significant] = derivative[significant] ** 2 / probability[significant]
    return float(np.trapezoid(integrand, x))


def quantum_fisher_information_pure(ket: np.ndarray) -> float:
    """``QFI = 4 Var(n)`` for a pure probe under phase rotation.

    The measurement-independent upper bound on any classical Fisher information, so it
    serves as a consistency check: ``F_homodyne <= QFI`` must hold. It also makes the
    Fock result immediate, since ``Var(n) = 0`` for a Fock state.
    """
    ket = np.asarray(ket, dtype=complex)
    populations = np.abs(ket) ** 2
    n = np.arange(ket.size)
    mean = float(np.sum(populations * n))
    mean_square = float(np.sum(populations * n**2))
    return 4.0 * (mean_square - mean**2)


def _log_factorial(n: np.ndarray) -> np.ndarray:
    return gammaln(n + 1)
