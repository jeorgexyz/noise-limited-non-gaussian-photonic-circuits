"""Wigner function of a truncated Fock-space density matrix.

Two independent implementations are provided on purpose:

``method="laguerre"``
    The closed-form Fock-basis expansion. Fast enough for production grids
    (cost is ``O(N^2)`` grid-sized arrays for cutoff ``N``).

``method="parity"``
    The definition itself, ``W(alpha) = (2/pi) Tr[rho D(alpha) Pi D(alpha)^dag]``,
    evaluated by building each displacement operator with a matrix exponential. Slow
    -- one ``expm`` per grid point -- but it has essentially no room for an algebra
    error, so it is the arbiter when the fast path is in doubt.

    It carries one sharp caveat: it displaces by the *full grid extent*, so it needs a
    far larger cutoff than the state itself does. A truncated ``D(alpha)`` stops being
    unitary once ``|alpha|^2`` approaches the cutoff, and the method then returns a
    smooth but inaccurate result.

    Accuracy is governed by ``ratio = cutoff / |alpha|^2_max``, and is insensitive to
    how that ratio is reached. Measured worst-case disagreement with the Laguerre path
    over squeezed, cat, and thermal states:

        ratio    3      5      6.5     10      15
        error    1e-2   2e-4   1e-7    2e-9    1e-14

    So ``cutoff >= 10 |alpha|^2_max`` for a trustworthy cross-check, and for a square
    grid ``|alpha|^2_max = limit^2``. The guard below warns below ratio 8.

The validation suite asserts that the two agree. A Wigner routine carrying an incorrect
factor or conjugate still produces plausible-looking negativity plots, so an independent
implementation is used as a check.

Convention
----------
``x = (a + a^dag)/sqrt(2)``, ``p = (a - a^dag)/(i sqrt(2))``, ``hbar = 1``,
``alpha = (x + i p)/sqrt(2)``, normalised so ``\\int W(x, p) dx dp = 1``.
Vacuum is ``W = exp(-(x^2 + p^2))/pi``; Fock ``|1>`` has ``W(0, 0) = -1/pi``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_laguerre, gammaln

from ..backends.reference import annihilation, creation

__all__ = ["phase_space_grid", "wigner", "wigner_fock"]


def phase_space_grid(
    limit: float = 5.0, points: int = 201
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Square grid ``(x, p, X, P)`` spanning ``[-limit, limit]`` in both quadratures.

    ``limit`` must comfortably enclose the state's support or the negativity integral
    will be silently truncated; ``points`` controls quadrature error. Both are swept in
    the convergence tests rather than assumed.
    """
    x = np.linspace(-limit, limit, points)
    p = np.linspace(-limit, limit, points)
    X, P = np.meshgrid(x, p, indexing="ij")
    return x, p, X, P


def wigner(
    rho: np.ndarray, X: np.ndarray, P: np.ndarray, method: str = "laguerre"
) -> np.ndarray:
    """Wigner function of ``rho`` evaluated on the meshgrid ``(X, P)``."""
    rho = np.asarray(rho, dtype=complex)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"rho must be a square matrix, got shape {rho.shape}.")
    if method == "laguerre":
        return _wigner_laguerre(rho, X, P)
    if method == "parity":
        return _wigner_parity(rho, X, P)
    raise ValueError(f"Unknown method {method!r}; expected 'laguerre' or 'parity'.")


def wigner_fock(n: int, X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Analytic Wigner function of Fock state ``|n>``.

    ``W_n(x, p) = ((-1)^n / pi) L_n(2 r^2) exp(-r^2)`` with ``r^2 = x^2 + p^2``.
    Used as an independent closed form in the tests.
    """
    r2 = X**2 + P**2
    return ((-1.0) ** n / np.pi) * eval_laguerre(n, 2.0 * r2) * np.exp(-r2)


# --------------------------------------------------------------------------------------
# Implementations
# --------------------------------------------------------------------------------------


def _wigner_laguerre(rho: np.ndarray, X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Closed-form Fock-basis expansion.

    ``W = (1/pi) sum_{m,n} rho[m, n] T_{nm}`` where, for ``m >= n`` and ``k = m - n``,

        ``T_{nm} = (-1)^n sqrt(n!/m!) (2 alpha*)^k L_n^{(k)}(2 r^2) exp(-r^2)``

    and ``T_{mn} = conj(T_{nm})``. Hermiticity of ``rho`` collapses each off-diagonal
    pair to ``2 Re[rho[m, n] T_{nm}]``.
    """
    cutoff = rho.shape[0]
    r2 = X**2 + P**2
    laguerre_arg = 2.0 * r2
    envelope = np.exp(-r2)
    two_alpha_conj = np.sqrt(2.0) * (X - 1j * P)  # == 2 alpha*

    total = np.zeros(X.shape, dtype=float)
    # Outer loop over the offset k = m - n so each power of (2 alpha*) is formed once
    # by one multiply, instead of being recomputed for every (n, m) pair sharing it.
    power = np.ones(X.shape, dtype=complex)
    for k in range(cutoff):
        if k:
            power = power * two_alpha_conj
        column = np.abs(np.diagonal(rho, offset=-k))
        if column.max(initial=0.0) < 1e-15:
            continue  # this whole diagonal of rho is empty
        # L_n^{(k)} is advanced by its three-term recurrence in n rather than called
        # from scipy once per (n, k): same values, without the per-call overhead that
        # dominated the O(N^2) sweep.
        lag_prev = np.zeros(X.shape)          # L_{-1}, unused at n = 0
        lag_curr = np.ones(X.shape)           # L_0^{(k)} = 1
        for n in range(cutoff - k):
            m = n + k
            amplitude = rho[m, n]
            if abs(amplitude) >= 1e-15:
                coeff = ((-1.0) ** n) * np.exp(0.5 * (gammaln(n + 1) - gammaln(m + 1)))
                contribution = amplitude * (coeff * power * lag_curr * envelope)
                total += np.real(contribution) if k == 0 else 2.0 * np.real(contribution)

            # L_{n+1}^{(k)} = ((2n + 1 + k - x) L_n^{(k)} - (n + k) L_{n-1}^{(k)}) / (n+1)
            lag_prev, lag_curr = lag_curr, (
                (2 * n + 1 + k - laguerre_arg) * lag_curr - (n + k) * lag_prev
            ) / (n + 1)

    return total / np.pi


def _wigner_parity(rho: np.ndarray, X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Displaced-parity definition, evaluated pointwise. Slow but hard to get wrong."""
    cutoff = rho.shape[0]
    max_alpha_sq = float((X**2 + P**2).max()) / 2.0
    if cutoff < 8.0 * max_alpha_sq:
        warnings.warn(
            f"Parity-method Wigner with cutoff {cutoff} on a grid reaching "
            f"|alpha|^2 = {max_alpha_sq:.1f}: the truncated displacement operator is "
            f"not unitary here and the result will be wrong near the grid edges. "
            f"Use cutoff >= {int(np.ceil(10.0 * max_alpha_sq))} or shrink the grid.",
            RuntimeWarning,
            stacklevel=3,
        )

    a = annihilation(cutoff)
    adag = creation(cutoff)
    parity = np.diag((-1.0) ** np.arange(cutoff)).astype(complex)

    out = np.empty(X.shape, dtype=float)
    for index in np.ndindex(X.shape):
        alpha = (X[index] + 1j * P[index]) / np.sqrt(2.0)
        disp = expm(alpha * adag - np.conj(alpha) * a)
        displaced_parity = disp @ parity @ disp.conj().T
        out[index] = np.real(np.trace(rho @ displaced_parity))
    return out / np.pi
