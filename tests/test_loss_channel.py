"""Validation of the pure-loss channel against closed forms.

This is the project's primary unit test. The V1 prototype asserted an exponential decay
law; the point of these tests is that nothing here is asserted -- every expected value
is derived analytically and the implementation has to reproduce it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    coherent_ket,
    fock_dm,
    mean_photon_number,
    purity,
    thermal_dm,
    to_dm,
    vacuum_dm,
)
from ngphotonic.noise.loss import analytic_lossy_fock1, apply_loss, loss_kraus

CUTOFF = 20
ETAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]


@pytest.mark.parametrize("eta", ETAS)
def test_kraus_operators_are_trace_preserving(eta: float) -> None:
    """``sum_k K_k^dag K_k = I`` exactly, with no truncation defect.

    Exact rather than approximate because each ``K_k`` only lowers Fock index, so the
    binomial identity terminates inside the truncated space.
    """
    kraus = loss_kraus(eta, CUTOFF)
    total = sum(k.conj().T @ k for k in kraus)
    np.testing.assert_allclose(total, np.eye(CUTOFF), atol=1e-12)


@pytest.mark.parametrize("eta", ETAS)
def test_single_photon_matches_closed_form(eta: float) -> None:
    """``E_eta(|1><1|) = eta |1><1| + (1 - eta) |0><0|``.

    The reference check named in RESEARCH_PLAN.md as the first validation milestone.
    """
    out = apply_loss(fock_dm(1, CUTOFF), eta)

    expected = np.zeros((CUTOFF, CUTOFF), dtype=complex)
    expected[:2, :2] = analytic_lossy_fock1(eta)
    np.testing.assert_allclose(out, expected, atol=1e-12)


@pytest.mark.parametrize("eta", ETAS)
@pytest.mark.parametrize("n", [0, 1, 2, 3, 5])
def test_fock_state_gives_binomial_distribution(eta: float, n: int) -> None:
    """``E_eta(|n><n|)`` is diagonal with ``P(k) = C(n, k) eta^k (1-eta)^(n-k)``.

    The general case behind the single-photon check: a Fock state under pure loss
    undergoes binomial thinning.
    """
    out = apply_loss(fock_dm(n, CUTOFF), eta)

    from scipy.stats import binom

    expected = np.zeros(CUTOFF)
    expected[: n + 1] = binom.pmf(np.arange(n + 1), n, eta)

    np.testing.assert_allclose(np.real(np.diag(out)), expected, atol=1e-12)
    # No off-diagonal coherence is generated from a diagonal input.
    np.testing.assert_allclose(out - np.diag(np.diag(out)), 0.0, atol=1e-12)


@pytest.mark.parametrize("eta", ETAS)
def test_trace_is_preserved_on_states(eta: float) -> None:
    for rho in (fock_dm(3, CUTOFF), thermal_dm(0.7, CUTOFF), vacuum_dm(CUTOFF)):
        assert np.isclose(np.trace(apply_loss(rho, eta)).real, 1.0, atol=1e-12)


@pytest.mark.parametrize("eta", ETAS)
def test_output_is_a_valid_density_matrix(eta: float) -> None:
    """Hermitian and positive semi-definite."""
    out = apply_loss(fock_dm(4, CUTOFF), eta)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(out).min() > -1e-12


@pytest.mark.parametrize("eta", ETAS)
def test_mean_photon_number_scales_linearly(eta: float) -> None:
    """``<n>`` after loss is ``eta <n>`` before -- the defining attenuator property."""
    rho = fock_dm(5, CUTOFF)
    assert np.isclose(mean_photon_number(apply_loss(rho, eta)), eta * 5.0, atol=1e-10)


def test_lossless_channel_is_identity() -> None:
    rho = to_dm(coherent_ket(1.0, CUTOFF))
    np.testing.assert_allclose(apply_loss(rho, 1.0), rho, atol=1e-12)


def test_total_loss_gives_vacuum() -> None:
    rho = to_dm(coherent_ket(1.2, CUTOFF))
    np.testing.assert_allclose(apply_loss(rho, 0.0), vacuum_dm(CUTOFF), atol=1e-12)


@pytest.mark.parametrize("eta_a, eta_b", [(0.9, 0.8), (0.5, 0.5), (0.99, 0.3)])
def test_channel_composes(eta_a: float, eta_b: float) -> None:
    """Loss is multiplicative: ``E_a . E_b = E_{a b}``.

    This is what makes a depth-``D`` chain of identical lossy layers equivalent to a
    single channel at ``eta^D``, which the depth sweeps will rely on.
    """
    rho = to_dm(coherent_ket(0.8, CUTOFF))
    composed = apply_loss(apply_loss(rho, eta_b), eta_a)
    direct = apply_loss(rho, eta_a * eta_b)
    np.testing.assert_allclose(composed, direct, atol=1e-12)


def test_coherent_state_stays_pure_and_shrinks() -> None:
    """A coherent state under loss stays pure and maps ``alpha -> sqrt(eta) alpha``."""
    alpha, eta = 1.1, 0.64
    out = apply_loss(to_dm(coherent_ket(alpha, CUTOFF)), eta)
    expected = to_dm(coherent_ket(np.sqrt(eta) * alpha, CUTOFF))
    np.testing.assert_allclose(out, expected, atol=1e-10)
    assert np.isclose(purity(out), 1.0, atol=1e-10)


def test_depth_chain_matches_single_equivalent_channel() -> None:
    """20 layers at eta=0.98 equals one layer at 0.98^20, to machine precision."""
    rho = fock_dm(2, CUTOFF)
    eta, depth = 0.98, 20

    chained = rho
    for _ in range(depth):
        chained = apply_loss(chained, eta)

    np.testing.assert_allclose(chained, apply_loss(rho, eta**depth), atol=1e-12)


@pytest.mark.parametrize("eta", [-0.01, 1.01])
def test_invalid_transmissivity_rejected(eta: float) -> None:
    with pytest.raises(ValueError, match="eta"):
        loss_kraus(eta, CUTOFF)
