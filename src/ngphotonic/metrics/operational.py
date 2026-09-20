"""Operational task scores.

RESEARCH_PLAN.md section 9. These are what make the advantage
``A(D, nu) = S_NG - S_G*`` meaningful: the score has to be something *both* circuit
classes can attempt.

Wigner negativity cannot serve as ``S``. Gaussian states have ``W_log = 0`` by
construction, so the difference would be positive for any non-Gaussian circuit
whatsoever, including one that has been destroyed by noise. That is the failure the
matched-baseline requirement exists to prevent, so the scores here are operational:
they ask what the state is good *for*.

Implemented:

``target_state_fidelity``
    Fidelity to a specified target state. Against a non-Gaussian target this has a
    hard Gaussian ceiling, which is what the baseline optimizer locates.

``phase_estimation_fisher_information``
    Classical Fisher information for estimating a phase rotation under photon-number
    measurement. An operational quantity rather than another state-space measure.
"""

from __future__ import annotations

import numpy as np

from ..backends.reference import fidelity, number

__all__ = [
    "target_state_fidelity",
    "phase_estimation_fisher_information",
    "photon_number_distribution",
]


def target_state_fidelity(rho: np.ndarray, target: np.ndarray) -> float:
    """Fidelity between an output state and a target.

    ``target`` may be a ket or a density matrix. For a pure target this reduces to
    ``<psi|rho|psi>``, which is computed directly since it is both cheaper and better
    conditioned than the general Uhlmann form.
    """
    rho = np.asarray(rho, dtype=complex)
    target = np.asarray(target, dtype=complex)

    if target.ndim == 1:
        if target.shape[0] != rho.shape[0]:
            raise ValueError(
                f"Dimension mismatch: state has cutoff {rho.shape[0]}, "
                f"target has {target.shape[0]}."
            )
        return float(np.real(target.conj() @ rho @ target))

    if target.shape != rho.shape:
        raise ValueError(f"Shape mismatch: {rho.shape} vs {target.shape}.")
    return fidelity(rho, target)


def photon_number_distribution(rho: np.ndarray) -> np.ndarray:
    """Photon-number probabilities ``P(n) = rho_nn``, clipped at zero."""
    return np.clip(np.real(np.diag(np.asarray(rho))), 0.0, None)


def phase_estimation_fisher_information(
    rho_at: callable, theta: float, delta: float = 1e-3
) -> float:
    """Classical Fisher information for a phase parameter under PNR measurement.

        ``F(theta) = sum_n (dP(n)/dtheta)^2 / P(n)``

    ``rho_at`` maps a phase to the output density matrix, so the caller decides what
    the phase does and what noise follows it. The derivative is a central difference,
    which is adequate here because ``P(n)`` is smooth in ``theta``.

    Terms with negligible ``P(n)`` are dropped: they contribute nothing in exact
    arithmetic but dominate the sum numerically, since the derivative of a
    near-zero probability divided by that probability is unstable.
    """
    probabilities_plus = photon_number_distribution(rho_at(theta + delta))
    probabilities_minus = photon_number_distribution(rho_at(theta - delta))
    probabilities = photon_number_distribution(rho_at(theta))

    derivative = (probabilities_plus - probabilities_minus) / (2.0 * delta)
    significant = probabilities > 1e-12
    return float(np.sum(derivative[significant] ** 2 / probabilities[significant]))


def mean_photon_number_of(rho: np.ndarray) -> float:
    """``<n>``, re-exported here so task code need not reach into the backend."""
    return float(np.real(np.trace(rho @ number(rho.shape[0]))))
