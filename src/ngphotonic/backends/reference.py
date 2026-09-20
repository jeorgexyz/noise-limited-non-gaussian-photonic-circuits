"""Pure-NumPy mixed-state Fock backend.

This is the *reference* backend: explicit, unoptimised, and written to be obviously
correct rather than fast. Its jobs are to (a) provide closed-form-checkable behaviour
for the validation suite and (b) serve as the independent cross-check against the
Piquasso backend once that lands. Production sweeps should not run on it.

Conventions
-----------
Single mode with Fock cutoff ``N``: states span ``{|0>, ..., |N-1>}``.

Quadratures are ``x = (a + a^dag) / sqrt(2)`` and ``p = (a - a^dag) / (i sqrt(2))``
with ``hbar = 1``, so the coherent-state parameter is ``alpha = (x + i p) / sqrt(2)``
and the vacuum Wigner function is ``W(x, p) = exp(-(x^2 + p^2)) / pi``, normalised as
``\\int W dx dp = 1``.

Density matrices are ``(N, N)`` complex arrays with ``rho[m, n] = <m|rho|n>``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import expm

__all__ = [
    "annihilation",
    "creation",
    "number",
    "fock_ket",
    "fock_dm",
    "vacuum_ket",
    "vacuum_dm",
    "coherent_ket",
    "squeezed_ket",
    "cat_ket",
    "thermal_dm",
    "to_dm",
    "purity",
    "mean_photon_number",
    "fidelity",
    "tail_weight",
]


# --------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------


def annihilation(cutoff: int) -> np.ndarray:
    """Annihilation operator ``a`` truncated to ``cutoff`` Fock levels."""
    _check_cutoff(cutoff)
    return np.diag(np.sqrt(np.arange(1, cutoff, dtype=float)), k=1)


def creation(cutoff: int) -> np.ndarray:
    """Creation operator ``a^dag`` truncated to ``cutoff`` Fock levels."""
    return annihilation(cutoff).conj().T


def number(cutoff: int) -> np.ndarray:
    """Number operator ``n = a^dag a``."""
    _check_cutoff(cutoff)
    return np.diag(np.arange(cutoff, dtype=float))


# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------


def fock_ket(n: int, cutoff: int) -> np.ndarray:
    """Fock state ``|n>`` as a ket vector."""
    _check_cutoff(cutoff)
    if not 0 <= n < cutoff:
        raise ValueError(f"Fock level {n} does not fit in a cutoff-{cutoff} space.")
    ket = np.zeros(cutoff, dtype=complex)
    ket[n] = 1.0
    return ket


def fock_dm(n: int, cutoff: int) -> np.ndarray:
    """Fock state ``|n><n|`` as a density matrix."""
    return to_dm(fock_ket(n, cutoff))


def vacuum_ket(cutoff: int) -> np.ndarray:
    """Vacuum ``|0>``."""
    return fock_ket(0, cutoff)


def vacuum_dm(cutoff: int) -> np.ndarray:
    """Vacuum ``|0><0|``."""
    return fock_dm(0, cutoff)


def coherent_ket(alpha: complex, cutoff: int) -> np.ndarray:
    """Coherent state ``|alpha>``.

    Built by the stable recurrence ``c_n = c_{n-1} alpha / sqrt(n)`` rather than by
    exponentiating a displacement generator, so it stays accurate for small ``|alpha|``
    without a matrix exponential. Check :func:`tail_weight` before trusting a result at
    large ``|alpha|``: the truncated state is renormalised, which silently hides
    truncation error otherwise.
    """
    _check_cutoff(cutoff)
    ket = np.zeros(cutoff, dtype=complex)
    ket[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for n in range(1, cutoff):
        ket[n] = ket[n - 1] * alpha / np.sqrt(n)
    return _normalise(ket)


def squeezed_ket(r: float, phi: float = 0.0, cutoff: int = 30) -> np.ndarray:
    """Squeezed vacuum ``S(r e^{i phi}) |0>``.

    Uses the matrix exponential of the squeeze generator, which is more robust against
    my own algebra errors than the closed-form even-Fock series. The generator is
    truncated before exponentiation, so ``S`` is only approximately unitary on the
    truncated space.

    That truncation has a consequence worth stating plainly, because it can fabricate
    exactly the resource this project measures. By Hudson's theorem a *pure* state has
    a non-negative Wigner function if and only if it is Gaussian. A truncated,
    renormalised squeezed ket is not Gaussian, so it necessarily carries some Wigner
    negativity -- negativity that is a numerical artefact, not physics. Measured
    spurious ``W_log`` for squeezed vacuum:

        r = 0.6:  cutoff 20 -> 2.0e-3,  40 -> 1.0e-6,  60 -> 0
        r = 1.0:  cutoff 20 -> 9.4e-2,  40 -> 4.8e-3,  80 -> 0

    At ``r = 1.0`` and cutoff 20 that artefact is 26% of the genuine ``W_log = 0.355``
    of a single photon. Any negativity claimed by this project must sit well above the
    floor set by :func:`tail_weight`; the warning below fires when it might not.
    """
    _check_cutoff(cutoff)
    a = annihilation(cutoff)
    adag = creation(cutoff)
    z = r * np.exp(1j * phi)
    generator = 0.5 * (np.conj(z) * (a @ a) - z * (adag @ adag))
    ket = _normalise(expm(generator) @ vacuum_ket(cutoff))

    tail = tail_weight(ket)
    if tail > 1e-10:
        warnings.warn(
            f"Squeezed state r={r} at cutoff {cutoff} leaves {tail:.2e} population in "
            f"the top Fock levels. The truncated state is non-Gaussian and will show "
            f"spurious Wigner negativity. Raise the cutoff before trusting any "
            f"negativity computed from it.",
            RuntimeWarning,
            stacklevel=2,
        )
    return ket


def cat_ket(alpha: complex, parity: str = "even", cutoff: int = 30) -> np.ndarray:
    """Cat state ``(|alpha> +/- |-alpha>)`` normalised.

    ``parity='even'`` gives the ``+`` superposition, ``'odd'`` the ``-``.
    """
    if parity not in ("even", "odd"):
        raise ValueError("parity must be 'even' or 'odd'.")
    sign = 1.0 if parity == "even" else -1.0
    ket = coherent_ket(alpha, cutoff) + sign * coherent_ket(-alpha, cutoff)
    norm = np.linalg.norm(ket)
    if norm < 1e-12:
        raise ValueError(
            f"Cat state with alpha={alpha} and parity={parity!r} is numerically null; "
            "the two coherent components cancel."
        )
    return ket / norm


def thermal_dm(n_bar: float, cutoff: int) -> np.ndarray:
    """Thermal state with mean occupation ``n_bar``, truncated and renormalised."""
    _check_cutoff(cutoff)
    if n_bar < 0:
        raise ValueError("n_bar must be non-negative.")
    if n_bar == 0:
        return vacuum_dm(cutoff)
    n = np.arange(cutoff)
    weights = (n_bar / (1.0 + n_bar)) ** n / (1.0 + n_bar)
    return np.diag(weights / weights.sum()).astype(complex)


# --------------------------------------------------------------------------------------
# State utilities
# --------------------------------------------------------------------------------------


def to_dm(ket: np.ndarray) -> np.ndarray:
    """Density matrix ``|psi><psi|`` from a ket."""
    ket = np.asarray(ket, dtype=complex)
    return np.outer(ket, ket.conj())


def purity(rho: np.ndarray) -> float:
    """``Tr(rho^2)``."""
    return float(np.real(np.trace(rho @ rho)))


def mean_photon_number(rho: np.ndarray) -> float:
    """``<n> = Tr(rho n)``.

    Unreliable if the state has appreciable population in the top few Fock levels;
    that is a cutoff problem, not a physics result.
    """
    return float(np.real(np.trace(rho @ number(rho.shape[0]))))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity ``(Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2``.

    Uses the eigen-decomposition of the Hermitian ``rho`` rather than a general matrix
    square root, clipping small negative eigenvalues that arise from round-off.
    """
    evals, evecs = np.linalg.eigh(rho)
    evals = np.clip(evals, 0.0, None)
    sqrt_rho = (evecs * np.sqrt(evals)) @ evecs.conj().T
    inner = sqrt_rho @ sigma @ sqrt_rho
    inner_evals = np.clip(np.linalg.eigvalsh(inner), 0.0, None)
    return float(np.sum(np.sqrt(inner_evals)) ** 2)


def tail_weight(state: np.ndarray, levels: int = 3) -> float:
    """Population in the top ``levels`` Fock states — a truncation-error proxy.

    Accepts a ket or a density matrix. A value that is not tiny means the cutoff is too
    low and any metric computed from the state is suspect.
    """
    state = np.asarray(state)
    populations = np.abs(state) ** 2 if state.ndim == 1 else np.real(np.diag(state))
    return float(np.sum(populations[-levels:]))


# --------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------


def _check_cutoff(cutoff: int) -> None:
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}.")


def _normalise(ket: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(ket)
    if norm < 1e-15:
        raise ValueError("Cannot normalise a null ket.")
    return ket / norm
