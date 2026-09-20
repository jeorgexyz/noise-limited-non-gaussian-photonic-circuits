"""Validation of the phase-diffusion channel.

The closed form ``rho_{mn} -> rho_{mn} exp(-sigma^2 (m-n)^2 / 2)`` is checked against an
independent Monte Carlo average over sampled phase rotations, and against the structural
properties it must have (trace, positivity, population preservation, composition).

Two of these tests carry results rather than just checks: phase diffusion leaves Fock
states entirely alone, and it commutes exactly with pure loss. Both constrain what the
depth sweeps can show.
"""

from __future__ import annotations

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    cat_ket,
    coherent_ket,
    fock_dm,
    mean_photon_number,
    purity,
    thermal_dm,
    to_dm,
)
from ngphotonic.metrics.negativity import wigner_log_negativity
from ngphotonic.metrics.wigner import phase_space_grid, wigner
from ngphotonic.noise.loss import apply_loss
from ngphotonic.noise.phase_diffusion import (
    apply_phase_diffusion,
    apply_phase_diffusion_sampled,
    compose_sigma,
    phase_diffusion_factors,
)

CUTOFF = 25
SIGMAS = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]


# --------------------------------------------------------------------------------------
# Closed form vs. independent Monte Carlo
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.2, 0.5, 1.0])
def test_closed_form_matches_monte_carlo(sigma: float) -> None:
    """The analytic ensemble average agrees with sampling the ensemble.

    Genuinely independent: one evaluates a Gaussian characteristic function, the other
    draws phases and rotates. Tolerance is set by 1/sqrt(samples).
    """
    rho = to_dm(cat_ket(1.2, "even", CUTOFF))
    exact = apply_phase_diffusion(rho, sigma)
    sampled = apply_phase_diffusion_sampled(rho, sigma, samples=200_000, seed=11)
    assert np.abs(exact - sampled).max() < 5e-3


@pytest.mark.slow
def test_monte_carlo_converges_as_inverse_sqrt_samples() -> None:
    """Monte Carlo convergence check, per RESEARCH_PLAN.md section 11."""
    rho = to_dm(cat_ket(1.2, "even", CUTOFF))
    exact = apply_phase_diffusion(rho, 0.5)

    errors = []
    for samples in (1_000, 10_000, 100_000):
        sampled = apply_phase_diffusion_sampled(rho, 0.5, samples=samples, seed=5)
        errors.append(np.abs(exact - sampled).max())

    assert errors[0] > errors[-1]
    # Ten times the samples should buy roughly sqrt(10) ~ 3.2x accuracy; allow slack.
    assert errors[0] / errors[-1] > 3.0


# --------------------------------------------------------------------------------------
# Structural properties
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", SIGMAS)
def test_trace_preserved_and_state_stays_valid(sigma: float) -> None:
    out = apply_phase_diffusion(to_dm(cat_ket(1.3, "even", CUTOFF)), sigma)
    assert np.isclose(np.trace(out).real, 1.0, atol=1e-12)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() > -1e-10


@pytest.mark.parametrize("sigma", SIGMAS)
def test_populations_are_untouched(sigma: float) -> None:
    """Diagonal elements are multiplied by exp(0) = 1, so photon statistics survive.

    Phase diffusion is pure dephasing: it carries away phase information and no energy.
    """
    rho = to_dm(cat_ket(1.4, "odd", CUTOFF))
    out = apply_phase_diffusion(rho, sigma)
    np.testing.assert_allclose(np.diag(out), np.diag(rho), atol=1e-12)
    assert np.isclose(mean_photon_number(out), mean_photon_number(rho), atol=1e-10)


def test_zero_sigma_is_the_identity() -> None:
    rho = to_dm(cat_ket(1.1, "even", CUTOFF))
    np.testing.assert_allclose(apply_phase_diffusion(rho, 0.0), rho, atol=1e-14)


def test_large_sigma_leaves_only_the_diagonal() -> None:
    """As sigma grows the state is dephased into its photon-number distribution."""
    rho = to_dm(coherent_ket(1.2, CUTOFF))
    out = apply_phase_diffusion(rho, 25.0)
    np.testing.assert_allclose(out, np.diag(np.diag(rho)), atol=1e-12)


@pytest.mark.parametrize("sigma_a, sigma_b", [(0.3, 0.4), (0.5, 0.5), (0.1, 0.9)])
def test_channel_composes_by_adding_variances(sigma_a: float, sigma_b: float) -> None:
    """Two steps equal one step at ``sqrt(sigma_a^2 + sigma_b^2)``.

    This is what makes a depth-D chain of phase-diffusion layers collapse to a single
    channel at ``sigma sqrt(D)``.
    """
    rho = to_dm(cat_ket(1.2, "even", CUTOFF))
    chained = apply_phase_diffusion(apply_phase_diffusion(rho, sigma_a), sigma_b)
    direct = apply_phase_diffusion(rho, compose_sigma(sigma_a, sigma_b))
    np.testing.assert_allclose(chained, direct, atol=1e-12)


def test_depth_chain_matches_single_equivalent_channel() -> None:
    rho = to_dm(cat_ket(1.3, "even", CUTOFF))
    sigma, depth = 0.15, 12

    chained = rho
    for _ in range(depth):
        chained = apply_phase_diffusion(chained, sigma)

    np.testing.assert_allclose(
        chained, apply_phase_diffusion(rho, sigma * np.sqrt(depth)), atol=1e-12
    )


def test_factors_are_symmetric_and_unit_on_the_diagonal() -> None:
    factors = phase_diffusion_factors(0.4, 8)
    np.testing.assert_allclose(np.diag(factors), 1.0, atol=1e-15)
    np.testing.assert_allclose(factors, factors.T, atol=1e-15)
    assert factors.max() <= 1.0 + 1e-15


def test_negative_sigma_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_phase_diffusion(fock_dm(1, CUTOFF), -0.1)


# --------------------------------------------------------------------------------------
# Results, not just checks
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.0, 0.5, 2.0, 50.0])
@pytest.mark.parametrize("n", [1, 2, 3])
def test_fock_states_are_completely_immune(sigma: float, n: int) -> None:
    """Phase diffusion cannot touch a Fock state, at any sigma.

    ``|n><n|`` is already diagonal, so every damping factor it meets is ``exp(0) = 1``.
    Its Wigner negativity is therefore exactly preserved however violent the dephasing.

    This is a statement about which noise axis threatens which resource, and it means a
    loss-plus-dephasing sweep on Fock resources has only one live axis.
    """
    rho = fock_dm(n, CUTOFF)
    np.testing.assert_allclose(apply_phase_diffusion(rho, sigma), rho, atol=1e-14)


def test_fock_negativity_survives_dephasing_but_cat_negativity_does_not() -> None:
    """The same noise destroys one resource and leaves the other untouched."""
    x, p, X, P = phase_space_grid(limit=6.0, points=301)
    sigma = 1.0

    fock = fock_dm(1, CUTOFF)
    fock_before = wigner_log_negativity(wigner(fock, X, P), x, p)
    fock_after = wigner_log_negativity(wigner(apply_phase_diffusion(fock, sigma), X, P), x, p)
    assert np.isclose(fock_before, fock_after, atol=1e-12)
    assert fock_after > 0.3

    cat = to_dm(cat_ket(1.6, "even", CUTOFF))
    cat_before = wigner_log_negativity(wigner(cat, X, P), x, p)
    cat_after = wigner_log_negativity(wigner(apply_phase_diffusion(cat, sigma), X, P), x, p)
    assert cat_before > 0.1
    assert cat_after < 0.5 * cat_before


@pytest.mark.parametrize("eta", [0.3, 0.7, 0.95])
@pytest.mark.parametrize("sigma", [0.2, 0.8])
def test_phase_diffusion_commutes_with_loss(eta: float, sigma: float) -> None:
    """Exactly, not approximately.

    Loss Kraus operators preserve the Fock-index offset ``m - n``, and phase diffusion
    depends on nothing else. Consequence: a layered circuit built only from these two
    collapses to a single ``(eta^D, sigma sqrt(D))`` channel, so depth is not an
    independent axis until something non-commuting is inserted between layers.
    """
    rho = to_dm(cat_ket(1.2, "even", CUTOFF))
    loss_first = apply_phase_diffusion(apply_loss(rho, eta), sigma)
    phase_first = apply_loss(apply_phase_diffusion(rho, sigma), eta)
    np.testing.assert_allclose(loss_first, phase_first, atol=1e-12)


def test_layered_loss_and_dephasing_collapses_to_one_channel() -> None:
    """The reducibility that the depth experiment has to work around."""
    rho = to_dm(cat_ket(1.3, "even", CUTOFF))
    eta, sigma, depth = 0.95, 0.1, 8

    layered = rho
    for _ in range(depth):
        layered = apply_phase_diffusion(apply_loss(layered, eta), sigma)

    collapsed = apply_phase_diffusion(apply_loss(rho, eta**depth), sigma * np.sqrt(depth))
    np.testing.assert_allclose(layered, collapsed, atol=1e-12)


@pytest.mark.parametrize("thermal_nbar", [0.0, 0.5])
def test_diagonal_states_are_fixed_points(thermal_nbar: float) -> None:
    rho = thermal_dm(thermal_nbar, CUTOFF)
    out = apply_phase_diffusion(rho, 1.5)
    np.testing.assert_allclose(out, rho, atol=1e-14)
    assert np.isclose(purity(out), purity(rho), atol=1e-12)
