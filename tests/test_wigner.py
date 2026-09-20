"""Validation of the Wigner transform.

Three independent references are used, so an error has nowhere to hide:

1. Closed forms for vacuum, Fock states, and coherent states.
2. The displaced-parity definition, evaluated by matrix exponential.
3. Normalisation and reality, which hold for any valid density matrix.

The cross-check between the fast Laguerre path and the slow parity path is the primary
test here, since a Wigner routine carrying an incorrect factor or a missing conjugate
still produces smooth, plausible-looking output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    cat_ket,
    coherent_ket,
    fock_dm,
    squeezed_ket,
    thermal_dm,
    to_dm,
    vacuum_dm,
)
from ngphotonic.metrics.wigner import phase_space_grid, wigner, wigner_fock

# The parity cross-check displaces by the full grid extent and needs
# cutoff >= 10 * limit^2. At limit 2.5 that is 63, so 60 lands at ratio 9.6
# (measured agreement ~3e-13), comfortably inside the 1e-9 tolerances below.
CUTOFF = 60


def _small_grid(limit: float = 2.5, points: int = 21):
    """Coarse grid for the expensive parity method, sized to keep D(alpha) unitary."""
    return phase_space_grid(limit=limit, points=points)


# --------------------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------------------


def test_vacuum_matches_gaussian() -> None:
    """``W_vac(x, p) = exp(-(x^2 + p^2)) / pi``."""
    x, p, X, P = phase_space_grid(limit=5.0, points=81)
    W = wigner(vacuum_dm(CUTOFF), X, P)
    np.testing.assert_allclose(W, np.exp(-(X**2 + P**2)) / np.pi, atol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_fock_states_match_laguerre_closed_form(n: int) -> None:
    """``W_n = ((-1)^n / pi) L_n(2 r^2) exp(-r^2)``."""
    x, p, X, P = phase_space_grid(limit=5.0, points=81)
    np.testing.assert_allclose(wigner(fock_dm(n, CUTOFF), X, P), wigner_fock(n, X, P), atol=1e-12)


def test_single_photon_has_known_value_at_origin() -> None:
    """``W(0, 0) = -1/pi`` for ``|1>`` -- the canonical Wigner-negative state."""
    _, _, X, P = phase_space_grid(limit=1.0, points=3)
    W = wigner(fock_dm(1, CUTOFF), X, P)
    assert np.isclose(W[1, 1], -1.0 / np.pi, atol=1e-12)


@pytest.mark.parametrize("alpha", [0.0, 0.7, 1.2j, 0.8 - 0.5j])
def test_coherent_state_is_displaced_gaussian(alpha: complex) -> None:
    """A coherent state is a vacuum Gaussian centred at ``(sqrt(2) Re a, sqrt(2) Im a)``."""
    x, p, X, P = phase_space_grid(limit=6.0, points=81)
    W = wigner(to_dm(coherent_ket(alpha, CUTOFF)), X, P)

    x0 = np.sqrt(2.0) * np.real(alpha)
    p0 = np.sqrt(2.0) * np.imag(alpha)
    expected = np.exp(-((X - x0) ** 2 + (P - p0) ** 2)) / np.pi
    np.testing.assert_allclose(W, expected, atol=1e-9)


def test_thermal_state_is_wider_gaussian() -> None:
    """``W_th = exp(-r^2/(2 n_bar + 1)) / (pi (2 n_bar + 1))``."""
    n_bar = 0.8
    x, p, X, P = phase_space_grid(limit=8.0, points=101)
    W = wigner(thermal_dm(n_bar, CUTOFF), X, P)
    spread = 2.0 * n_bar + 1.0
    expected = np.exp(-(X**2 + P**2) / spread) / (np.pi * spread)
    np.testing.assert_allclose(W, expected, atol=1e-6)


# --------------------------------------------------------------------------------------
# Cross-check: fast path vs. definition
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, rho",
    [
        ("vacuum", vacuum_dm(CUTOFF)),
        ("fock1", fock_dm(1, CUTOFF)),
        ("fock3", fock_dm(3, CUTOFF)),
        ("coherent", to_dm(coherent_ket(0.9, CUTOFF))),
        ("coherent_complex", to_dm(coherent_ket(0.6 - 0.7j, CUTOFF))),
        ("squeezed", to_dm(squeezed_ket(0.5, 0.0, CUTOFF))),
        ("squeezed_rotated", to_dm(squeezed_ket(0.4, 0.9, CUTOFF))),
        ("cat_even", to_dm(cat_ket(1.1, "even", CUTOFF))),
        ("cat_odd", to_dm(cat_ket(1.1, "odd", CUTOFF))),
        ("thermal", thermal_dm(0.5, CUTOFF)),
    ],
)
def test_laguerre_agrees_with_displaced_parity(name: str, rho: np.ndarray) -> None:
    """The fast closed form must reproduce the definition on off-diagonal states.

    Cat and rotated-squeezed states are included specifically because they carry large
    off-diagonal coherences in the Fock basis; the diagonal states alone would not
    catch a conjugation or phase error.
    """
    _, _, X, P = _small_grid()
    fast = wigner(rho, X, P, method="laguerre")
    exact = wigner(rho, X, P, method="parity")
    np.testing.assert_allclose(fast, exact, atol=1e-9)


def test_superposition_interference_fringes_agree() -> None:
    """``(|0> + |3>)/sqrt(2)`` has threefold interference structure; both paths must match."""
    ket = np.zeros(CUTOFF, dtype=complex)
    ket[0] = ket[3] = 1.0 / np.sqrt(2.0)
    _, _, X, P = _small_grid()
    np.testing.assert_allclose(
        wigner(to_dm(ket), X, P, method="laguerre"),
        wigner(to_dm(ket), X, P, method="parity"),
        atol=1e-9,
    )
    # A genuine superposition must produce negativity somewhere.
    assert wigner(to_dm(ket), X, P, method="laguerre").min() < -1e-3


# --------------------------------------------------------------------------------------
# General properties
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rho",
    [
        fock_dm(2, CUTOFF),
        to_dm(coherent_ket(1.0, CUTOFF)),
        to_dm(cat_ket(1.2, "even", CUTOFF)),
        thermal_dm(0.4, CUTOFF),
    ],
)
def test_wigner_is_real_and_normalised(rho: np.ndarray) -> None:
    """``\\int W dx dp = 1`` for any valid state, and ``W`` is real by construction."""
    x, p, X, P = phase_space_grid(limit=7.0, points=201)
    W = wigner(rho, X, P)
    assert np.isrealobj(W)
    norm = np.trapezoid(np.trapezoid(W, p, axis=1), x)
    assert np.isclose(norm, 1.0, atol=1e-6)


def test_gaussian_states_are_non_negative() -> None:
    """Vacuum, coherent, squeezed, and thermal states have no Wigner negativity.

    The Gaussian limit check from RESEARCH_PLAN.md: whatever the metric pipeline
    reports for these states must be zero, or it is measuring numerical noise.
    """
    _, _, X, P = phase_space_grid(limit=6.0, points=121)
    for rho in (
        vacuum_dm(CUTOFF),
        to_dm(coherent_ket(1.3, CUTOFF)),
        to_dm(squeezed_ket(0.6, 0.3, CUTOFF)),
        thermal_dm(1.0, CUTOFF),
    ):
        assert wigner(rho, X, P).min() > -1e-9


def test_parity_method_degrades_when_cutoff_too_low() -> None:
    """The parity path fails loudly, not silently, when the grid outruns the cutoff.

    Asserted here because the error is smooth and plausible rather than obvious. A
    cutoff-16 state on a grid reaching
    |alpha|^2 = 25 has a non-unitary displacement operator, so the two Wigner paths
    disagree at the percent level rather than at machine precision.
    """
    _, _, X, P = phase_space_grid(limit=2.0, points=13)  # |alpha|^2_max = 4

    # ratio 3: the displacement operator is badly non-unitary here. The starved
    # squeezed state raises its own (expected, separate) truncation warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        small = to_dm(squeezed_ket(0.5, 0.0, 12))
    with pytest.warns(RuntimeWarning, match="not unitary"):
        bad = wigner(small, X, P, method="parity")
    assert np.abs(bad - wigner(small, X, P, method="laguerre")).max() > 1e-3

    # ratio 15: agreement returns to machine precision, and no warning fires.
    big = to_dm(squeezed_ket(0.5, 0.0, 60))
    good = wigner(big, X, P, method="parity")
    np.testing.assert_allclose(good, wigner(big, X, P, method="laguerre"), atol=1e-11)


def test_unknown_method_rejected() -> None:
    _, _, X, P = _small_grid(points=5)
    with pytest.raises(ValueError, match="Unknown method"):
        wigner(vacuum_dm(CUTOFF), X, P, method="nope")
