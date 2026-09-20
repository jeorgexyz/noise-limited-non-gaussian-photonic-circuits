"""Pure-loss (attenuator) channel.

Loss is the first noise source in this project because it dominates most photonic
systems and, unusually among the imperfections here, it has closed-form checks that
make an implementation bug obvious rather than plausible.

The channel is ``E_eta(rho) = sum_k K_k rho K_k^dag`` with Kraus operators

    K_k = sqrt((1 - eta)^k / k!) eta^(n/2) a^k

whose matrix elements work out to

    <n| K_k |n + k> = sqrt( C(n + k, k) (1 - eta)^k eta^n )

and zero elsewhere. ``eta`` is the transmissivity: ``eta = 1`` is lossless, ``eta = 0``
maps everything to vacuum.

Truncation note
---------------
Every ``K_k`` only lowers Fock index, so the channel maps a cutoff-``N`` space into
itself with **no leakage**, and ``sum_k K_k^dag K_k = I`` holds *exactly* on the
truncated space (the identity ``sum_k C(n, k) (1-eta)^k eta^(n-k) = 1`` is a binomial
expansion that terminates at ``k = n``). Pure loss therefore introduces no truncation
error of its own, so a trace defect observed in a longer circuit originates elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, xlogy

__all__ = ["analytic_lossy_fock1", "apply_loss", "loss_kraus"]


def loss_kraus(eta: float, cutoff: int) -> list[np.ndarray]:
    """Kraus operators for the pure-loss channel at transmissivity ``eta``.

    Returns ``cutoff`` operators (``k = 0 .. cutoff-1``); higher ``k`` would annihilate
    every state in the truncated space.
    """
    _check_eta(eta)
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}.")

    n = np.arange(cutoff)
    kraus = []
    for k in range(cutoff):
        op = np.zeros((cutoff, cutoff), dtype=complex)
        rows = n[: cutoff - k]
        # log C(n+k, k) = lgamma(n+k+1) - lgamma(k+1) - lgamma(n+1)
        log_binom = gammaln(rows + k + 1) - gammaln(k + 1) - gammaln(rows + 1)
        # xlogy(c, v) == 0 when c == 0 even for v == 0, which is what makes the
        # eta = 0 and eta = 1 endpoints work: a plain k * log(0) gives 0 * -inf = nan.
        log_amp = 0.5 * (log_binom + xlogy(k, 1.0 - eta) + xlogy(rows, eta))
        op[rows, rows + k] = np.exp(log_amp)
        kraus.append(op)
    return kraus


def apply_loss(rho: np.ndarray, eta: float) -> np.ndarray:
    """Apply the pure-loss channel to a density matrix."""
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {rho.shape}.")
    kraus = loss_kraus(eta, rho.shape[0])
    return sum(k @ rho @ k.conj().T for k in kraus)


def analytic_lossy_fock1(eta: float) -> np.ndarray:
    """Closed form for ``E_eta(|1><1|)`` as a 2x2 block: ``eta|1><1| + (1-eta)|0><0|``.

    The reference the loss implementation is validated against.
    """
    _check_eta(eta)
    return np.diag([1.0 - eta, eta]).astype(complex)


def _check_eta(eta: float) -> None:
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"Transmissivity eta must lie in [0, 1], got {eta}.")

