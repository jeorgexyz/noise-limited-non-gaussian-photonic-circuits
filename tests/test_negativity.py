"""Validation of the Wigner-negativity measures.

The centrepiece is the lossy single photon. Its Wigner function is

    ``W(x, p) = (1/pi) exp(-r^2) [2 eta r^2 - 2 eta + 1]``

which is negative exactly where ``r^2 < (2 eta - 1)/(2 eta)``, giving

    ``\\int |W| = 4 eta exp(-(1 - 1/(2 eta))) - 1``  for ``eta >= 1/2``

and ``1`` otherwise. So negativity vanishes at exactly ``eta = 1/2``.

That single closed form exercises the loss channel, the Wigner transform, and the
negativity integral simultaneously, which is why RESEARCH_PLAN.md nominates it as the
first validation milestone. It requires an explicitly constructed state, which the
phenomenological V1 model does not produce.
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
from ngphotonic.metrics.negativity import (
    analytic_lossy_fock1_abs_integral,
    analytic_lossy_fock1_log_negativity,
    grid_diagnostics,
    integrate_abs_wigner,
    negative_volume,
    wigner_log_negativity,
)
from ngphotonic.metrics.wigner import phase_space_grid, wigner
from ngphotonic.noise.loss import apply_loss

CUTOFF = 30
# Squeezed states need far more headroom: a truncated pure squeezed ket is
# non-Gaussian (Hudson) and shows spurious negativity until the tail is negligible.
GAUSSIAN_CUTOFF = 60
# Grid chosen by the convergence test below, not by eye.
GRID = {"limit": 6.0, "points": 401}

# |W| has a kink at each zero crossing, so trapezoidal quadrature converges more slowly
# than its usual O(h^2). 1e-4 is the realistic tolerance on this grid.
ATOL = 1e-4


def _grid():
    return phase_space_grid(**GRID)


# --------------------------------------------------------------------------------------
# Closed forms
# --------------------------------------------------------------------------------------


def test_single_photon_abs_integral_matches_closed_form() -> None:
    """``\\int |W| = 4 e^{-1/2} - 1`` for ``|1>``."""
    x, p, X, P = _grid()
    W = wigner(fock_dm(1, CUTOFF), X, P)
    assert np.isclose(integrate_abs_wigner(W, x, p), 4.0 * np.exp(-0.5) - 1.0, atol=ATOL)


def test_single_photon_log_negativity_value() -> None:
    """``W_log(|1>) = log(4 e^{-1/2} - 1) ~ 0.3549``."""
    x, p, X, P = _grid()
    W = wigner(fock_dm(1, CUTOFF), X, P)
    assert np.isclose(wigner_log_negativity(W, x, p), np.log(4.0 * np.exp(-0.5) - 1.0), atol=ATOL)


@pytest.mark.parametrize("eta", [1.0, 0.95, 0.9, 0.8, 0.75, 0.6, 0.55, 0.5, 0.45, 0.3, 0.1, 0.0])
def test_lossy_single_photon_tracks_closed_form(eta: float) -> None:
    """The headline validation: simulated negativity matches the analytic curve.

    Exercises the loss channel, the Wigner transform, and the quadrature at once.
    """
    x, p, X, P = _grid()
    rho = apply_loss(fock_dm(1, CUTOFF), eta)
    W = wigner(rho, X, P)

    assert np.isclose(
        integrate_abs_wigner(W, x, p), analytic_lossy_fock1_abs_integral(eta), atol=ATOL
    )
    assert np.isclose(
        wigner_log_negativity(W, x, p), analytic_lossy_fock1_log_negativity(eta), atol=ATOL
    )


def test_negativity_vanishes_at_half_transmissivity() -> None:
    """The sharp threshold: a single photon stops being Wigner-negative at eta = 1/2.

    Not a fitted or thresholded number -- it is where ``2 eta r^2 - 2 eta + 1`` stops
    reaching below zero anywhere in phase space.
    """
    x, p, X, P = _grid()

    def w_log(eta: float) -> float:
        return wigner_log_negativity(wigner(apply_loss(fock_dm(1, CUTOFF), eta), X, P), x, p)

    assert w_log(0.55) > 1e-3
    assert w_log(0.51) > 0.0
    assert w_log(0.50) == pytest.approx(0.0, abs=1e-9)
    assert w_log(0.49) == pytest.approx(0.0, abs=1e-9)
    assert w_log(0.40) == pytest.approx(0.0, abs=1e-9)


def test_negativity_decreases_monotonically_with_loss() -> None:
    """Additional loss does not increase the resource."""
    x, p, X, P = _grid()
    etas = np.linspace(0.5, 1.0, 11)
    values = [
        wigner_log_negativity(wigner(apply_loss(fock_dm(1, CUTOFF), e), X, P), x, p) for e in etas
    ]
    assert np.all(np.diff(values) >= -1e-9)


def test_wigner_negative_volume_of_single_photon() -> None:
    """Negative volume ``\\int (|W| - W)/2 = (\\int |W| - 1)/2 ~ 0.2131``."""
    x, p, X, P = _grid()
    W = wigner(fock_dm(1, CUTOFF), X, P)
    assert np.isclose(negative_volume(W, x, p), (4.0 * np.exp(-0.5) - 2.0) / 2.0, atol=ATOL)


# --------------------------------------------------------------------------------------
# Gaussian limit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, rho",
    [
        ("vacuum", vacuum_dm(CUTOFF)),
        ("coherent", to_dm(coherent_ket(1.2, CUTOFF))),
        ("squeezed", to_dm(squeezed_ket(0.6, 0.0, GAUSSIAN_CUTOFF))),
        ("squeezed_rotated", to_dm(squeezed_ket(0.5, 1.1, GAUSSIAN_CUTOFF))),
        ("thermal", thermal_dm(0.9, CUTOFF)),
    ],
)
def test_gaussian_states_have_zero_log_negativity(name: str, rho: np.ndarray) -> None:
    """Gaussian states carry no Wigner negativity, so the metric must report zero.

    The other half of the validation: a measure that fires on Gaussian states is
    measuring quadrature noise, and would make every later "advantage" spurious.
    """
    x, p, X, P = _grid()
    W = wigner(rho, X, P)
    assert wigner_log_negativity(W, x, p) < 1e-6
    assert negative_volume(W, x, p) < 1e-6


def test_truncation_induces_negativity_in_gaussian_states() -> None:
    """Too small a cutoff produces negativity in a state that has none.

    A squeezed vacuum is Gaussian and must have ``W_log = 0``. Truncate it and, by
    Hudson's theorem, the renormalised pure state is no longer Gaussian and acquires
    Wigner negativity that is pure numerics. At ``r = 1.0`` and cutoff 20 the artefact
    reaches ``W_log ~ 0.063``, about 18% of the ``0.355`` carried by a single photon, so
    it is comparable in magnitude to a physical resource.

    This is the concrete form of the hazard RESEARCH_PLAN.md section 11 exists to
    guard against, so it is asserted rather than left as a comment.
    """
    x, p, X, P = _grid()
    reference_value = wigner_log_negativity(wigner(fock_dm(1, CUTOFF), X, P), x, p)

    with pytest.warns(RuntimeWarning, match="spurious Wigner negativity"):
        starved = to_dm(squeezed_ket(1.0, 0.0, 20))
    artefact = wigner_log_negativity(wigner(starved, X, P), x, p)

    assert artefact > 0.05
    assert artefact > 0.15 * reference_value  # comparable to a physical resource

    # Enough headroom and the artefact disappears.
    converged = to_dm(squeezed_ket(1.0, 0.0, 100))
    assert wigner_log_negativity(wigner(converged, X, P), x, p) < 1e-9


def test_truncation_artefact_shrinks_monotonically_with_cutoff() -> None:
    """The artefact must be controllable by cutoff alone, or the metric is unusable."""
    x, p, X, P = _grid()
    values = []
    for cutoff in (20, 40, 60, 100):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ket = squeezed_ket(0.8, 0.0, cutoff)
        values.append(wigner_log_negativity(wigner(to_dm(ket), X, P), x, p))
    assert np.all(np.diff(values) <= 1e-12), values
    assert values[-1] < 1e-9


# --------------------------------------------------------------------------------------
# Non-Gaussian resources
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_fock_states_are_negative(n: int) -> None:
    x, p, X, P = _grid()
    assert wigner_log_negativity(wigner(fock_dm(n, CUTOFF), X, P), x, p) > 0.3


@pytest.mark.parametrize("parity", ["even", "odd"])
def test_cat_states_are_negative(parity: str) -> None:
    x, p, X, P = _grid()
    W = wigner(to_dm(cat_ket(1.4, parity, CUTOFF)), X, P)
    assert wigner_log_negativity(W, x, p) > 0.1


def test_cat_negativity_survives_less_loss_than_fock() -> None:
    """A larger cat is more fragile than a single photon under the same loss.

    Resource strength and resource robustness are different axes -- one of the things
    the V2 sweeps exist to quantify.
    """
    x, p, X, P = _grid()
    eta = 0.8

    def w_log(rho):
        return wigner_log_negativity(wigner(apply_loss(rho, eta), X, P), x, p)

    assert w_log(to_dm(cat_ket(2.0, "even", CUTOFF))) < w_log(fock_dm(1, CUTOFF))


# --------------------------------------------------------------------------------------
# Numerical hygiene
# --------------------------------------------------------------------------------------


def test_grid_diagnostics_report_a_sane_grid() -> None:
    x, p, X, P = _grid()
    diag = grid_diagnostics(wigner(fock_dm(1, CUTOFF), X, P), x, p)
    assert diag["norm_error"] < 1e-6
    assert diag["edge_mass"] < 1e-12
    assert diag["min_value"] < 0.0


def test_grid_diagnostics_flag_a_grid_that_is_too_small() -> None:
    """A truncated grid loses Wigner mass, and the diagnostic has to notice."""
    x, p, X, P = phase_space_grid(limit=1.0, points=51)
    diag = grid_diagnostics(wigner(fock_dm(1, CUTOFF), X, P), x, p)
    assert diag["norm_error"] > 0.1
    assert diag["edge_mass"] > 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("points", [201, 301, 401, 501])
def test_quadrature_converges_with_grid_refinement(points: int) -> None:
    """Grid convergence for the headline number, per RESEARCH_PLAN.md section 11."""
    x, p, X, P = phase_space_grid(limit=6.0, points=points)
    value = integrate_abs_wigner(wigner(fock_dm(1, CUTOFF), X, P), x, p)
    assert abs(value - (4.0 * np.exp(-0.5) - 1.0)) < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("cutoff", [10, 15, 20, 30, 40])
def test_result_is_independent_of_fock_cutoff(cutoff: int) -> None:
    """Cutoff convergence ``|M_{c+dc} - M_c| < eps``.

    A single photon needs only two Fock levels, so every cutoff above 2 must agree.
    Inexpensive here, but the same check bounds truncation effects for the cat and
    squeezed states.
    """
    x, p, X, P = _grid()
    value = integrate_abs_wigner(wigner(apply_loss(fock_dm(1, cutoff), 0.8), X, P), x, p)
    assert abs(value - analytic_lossy_fock1_abs_integral(0.8)) < ATOL
