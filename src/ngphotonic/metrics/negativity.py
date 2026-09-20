"""Wigner-negativity resource measures.

The headline quantity is the Wigner logarithmic negativity

    ``W_log(rho) = log \\int |W_rho(q, p)| dq dp``

an established non-Gaussian resource monotone, which vanishes exactly on states with a
non-negative Wigner function. Also provided is the negative volume
``\\int (|W| - W)/2``, i.e. the mass sitting below zero, which some of the literature
reports instead.

Both are grid integrals, so both inherit two error sources that must be controlled
rather than assumed away: the grid must extend far enough to contain the state's
support, and it must be fine enough to resolve the oscillations. ``grid_diagnostics``
reports on the first; the convergence tests handle the second.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "analytic_lossy_fock1_abs_integral",
    "analytic_lossy_fock1_log_negativity",
    "grid_diagnostics",
    "integrate_abs_wigner",
    "negative_volume",
    "wigner_log_negativity",
]


def integrate_abs_wigner(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> float:
    """``\\int |W| dx dp`` by two-dimensional trapezoidal quadrature."""
    return float(np.trapezoid(np.trapezoid(np.abs(W), p, axis=1), x))


def wigner_log_negativity(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> float:
    """``W_log = log \\int |W| dx dp``.

    Clamped at zero: ``\\int |W| >= \\int W = 1`` exactly, so a negative result can only
    come from quadrature error on a state that is in fact Wigner-positive.
    """
    return float(max(np.log(integrate_abs_wigner(W, x, p)), 0.0))


def negative_volume(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> float:
    """``\\int (|W| - W)/2 dx dp`` -- the Wigner mass below zero."""
    negative_part = np.where(W < 0.0, -W, 0.0)
    return float(np.trapezoid(np.trapezoid(negative_part, p, axis=1), x))


def grid_diagnostics(W: np.ndarray, x: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Sanity numbers for a computed Wigner grid.

    ``norm`` should be 1 (it is ``\\int W``, which is exact for any valid state, so a
    deviation indicts the grid rather than the state). ``edge_mass`` is the largest
    ``|W|`` anywhere on the boundary: if it is not tiny, the grid is too small and both
    negativity measures are underestimates.
    """
    norm = float(np.trapezoid(np.trapezoid(W, p, axis=1), x))
    edge = np.concatenate([np.abs(W[0]), np.abs(W[-1]), np.abs(W[:, 0]), np.abs(W[:, -1])])
    return {
        "norm": norm,
        "norm_error": abs(norm - 1.0),
        "edge_mass": float(edge.max()),
        "min_value": float(W.min()),
    }


# --------------------------------------------------------------------------------------
# Closed forms for validation
# --------------------------------------------------------------------------------------


def analytic_lossy_fock1_abs_integral(eta: float) -> float:
    """``\\int |W| dx dp`` for a single photon after pure loss of transmissivity ``eta``.

    The state is ``eta |1><1| + (1 - eta) |0><0|``, whose Wigner function is

        ``W(x, p) = (1/pi) exp(-r^2) [2 eta r^2 - 2 eta + 1]``

    negative exactly where ``r^2 < (2 eta - 1) / (2 eta)``. Integrating the negative
    region and using ``\\int W = 1`` gives

        ``\\int |W| = 4 eta exp(-(1 - 1/(2 eta))) - 1``   for ``eta >= 1/2``
        ``\\int |W| = 1``                                 for ``eta <= 1/2``

    so Wigner negativity of a lossy single photon vanishes exactly at ``eta = 1/2``.
    This is the sharpest closed-form check available to the project and it exercises
    the loss channel, the Wigner transform, and the negativity integral at once.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"Transmissivity eta must lie in [0, 1], got {eta}.")
    if eta <= 0.5:
        return 1.0
    return 4.0 * eta * np.exp(-(1.0 - 1.0 / (2.0 * eta))) - 1.0


def analytic_lossy_fock1_log_negativity(eta: float) -> float:
    """``W_log`` for a single photon after pure loss -- log of the closed form above."""
    return float(np.log(analytic_lossy_fock1_abs_integral(eta)))
