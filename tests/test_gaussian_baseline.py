"""Validation of the Gaussian state family, task scores and baseline optimizer.

The baseline is the component that decides whether a measured advantage is real, so it
is checked against independent constructions at every level:

- the state family against separately-built coherent and squeezed states,
- the mean photon number against its closed form,
- the optimizer's answer against a brute-force grid search,
- and the optimizer's answer against the best *coherent* state, which is analytically
  ``1/e`` for a single-photon target and which any correct optimizer must beat.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    coherent_ket,
    fock_dm,
    fock_ket,
    mean_photon_number,
    squeezed_ket,
    to_dm,
)
from ngphotonic.baselines.gaussian import (
    GaussianParams,
    displace_unitary,
    gaussian_ket,
    is_gaussian_channel,
    mean_photon_number_analytic,
)
from ngphotonic.metrics.operational import (
    phase_estimation_fisher_information,
    photon_number_distribution,
    target_state_fidelity,
)
from ngphotonic.noise.loss import apply_loss
from ngphotonic.optimization.gaussian_baseline import (
    default_shell_starts,
    default_starts,
    optimize_gaussian_baseline,
)

CUTOFF = 30

# Maximum fidelity between a single photon and any energy-matched Gaussian state.
# Confirmed twice independently of the optimizer: a 2-D brute-force grid on the energy
# shell, using rotational symmetry to fix the displacement on the real axis, agrees to
# 2.4e-09; a full 4-parameter scan that assumes no symmetry agrees to 1.4e-06, limited
# by its coarser grid. The second run also confirms the symmetry reduction used by the
# first.
MAX_GAUSSIAN_FIDELITY_TO_FOCK1 = 0.47788941

# Best *coherent* state only: |<1|alpha>|^2 = |alpha|^2 exp(-|alpha|^2), max 1/e.
BEST_COHERENT_FIDELITY_TO_FOCK1 = 1.0 / np.e


@pytest.fixture(autouse=True)
def _quiet_truncation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def _fast_starts(budget: float = 0.0):
    """Smaller start set to keep the suite quick; the full set is used in slow tests.

    Shell coordinates, matching the optimizer's default parameterization. ``budget`` is
    accepted and ignored so callers read the same as before.
    """
    return default_shell_starts(extra=3)


# --------------------------------------------------------------------------------------
# State family
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.5, 1.2, 1.0 + 0.8j, -0.7 + 0.3j])
def test_displacement_reproduces_coherent_state(alpha: complex) -> None:
    """``D(alpha)|0>`` against the independently-built coherent state.

    The displacement operator is a matrix exponential on the truncated space while
    ``coherent_ket`` uses a stable recurrence, so this is a genuine cross-check.
    """
    displaced = displace_unitary(alpha, CUTOFF) @ np.eye(CUTOFF)[0]
    np.testing.assert_allclose(displaced, coherent_ket(alpha, CUTOFF), atol=1e-12)


def test_zero_squeezing_reduces_to_coherent() -> None:
    ket = gaussian_ket(GaussianParams(alpha_re=1.0), CUTOFF)
    np.testing.assert_allclose(ket, coherent_ket(1.0, CUTOFF), atol=1e-12)


@pytest.mark.parametrize("r, phi", [(0.6, 0.0), (0.4, 1.1)])
def test_zero_displacement_reduces_to_squeezed(r: float, phi: float) -> None:
    ket = gaussian_ket(GaussianParams(r=r, phi=phi), CUTOFF)
    np.testing.assert_allclose(ket, squeezed_ket(r, phi, CUTOFF), atol=1e-14)


@pytest.mark.parametrize(
    "params",
    [
        GaussianParams(alpha_re=1.0),
        GaussianParams(r=0.6),
        GaussianParams(alpha_re=0.8, alpha_im=0.5, r=0.4, phi=1.1),
    ],
)
def test_mean_photon_number_matches_closed_form(params: GaussianParams) -> None:
    """``n_bar = |alpha|^2 + sinh^2(r)``, which the energy constraint relies on."""
    ket = gaussian_ket(params, 50)
    assert np.isclose(mean_photon_number(to_dm(ket)), mean_photon_number_analytic(params),
                      atol=1e-8)


def test_vacuum_has_zero_energy() -> None:
    assert mean_photon_number_analytic(GaussianParams()) == 0.0


def test_phase_diffusion_is_not_a_gaussian_channel() -> None:
    """Recorded explicitly: only sigma = 0 leaves a Gaussian input Gaussian."""
    assert is_gaussian_channel(0.0)
    assert not is_gaussian_channel(0.2)


# --------------------------------------------------------------------------------------
# Task scores
# --------------------------------------------------------------------------------------


def test_fidelity_to_self_is_one() -> None:
    assert np.isclose(target_state_fidelity(fock_dm(1, CUTOFF), fock_ket(1, CUTOFF)), 1.0)


def test_fidelity_of_orthogonal_states_is_zero() -> None:
    assert np.isclose(target_state_fidelity(fock_dm(0, CUTOFF), fock_ket(1, CUTOFF)), 0.0,
                      atol=1e-14)


@pytest.mark.parametrize("eta", [0.0, 0.3, 0.5, 0.8, 1.0])
def test_lossy_photon_fidelity_equals_transmissivity(eta: float) -> None:
    """``E_eta(|1><1|) = eta|1><1| + (1-eta)|0><0|``, so its fidelity to |1> is exactly eta.

    A closed form for the non-Gaussian arm of the comparison.
    """
    score = target_state_fidelity(apply_loss(fock_dm(1, CUTOFF), eta), fock_ket(1, CUTOFF))
    assert np.isclose(score, eta, atol=1e-12)


def test_best_coherent_fidelity_to_single_photon_is_one_over_e() -> None:
    """``|<1|alpha>|^2 = |alpha|^2 e^{-|alpha|^2}``, maximised at ``|alpha|^2 = 1``."""
    best = max(
        target_state_fidelity(to_dm(coherent_ket(a, CUTOFF)), fock_ket(1, CUTOFF))
        for a in np.linspace(0.1, 2.5, 400)
    )
    assert np.isclose(best, BEST_COHERENT_FIDELITY_TO_FOCK1, atol=1e-6)


def test_photon_number_distribution_is_a_distribution() -> None:
    probabilities = photon_number_distribution(apply_loss(fock_dm(2, CUTOFF), 0.6))
    assert np.all(probabilities >= 0.0)
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-12)


def test_fisher_information_is_zero_for_phase_insensitive_state() -> None:
    """A Fock state's photon statistics do not depend on phase, so it carries no
    phase information under photon-number measurement."""
    from ngphotonic.backends.reference import kerr_unitary, apply_unitary

    def rho_at(theta: float) -> np.ndarray:
        rotation = np.diag(np.exp(-1j * theta * np.arange(CUTOFF)))
        return apply_unitary(fock_dm(1, CUTOFF), rotation)

    assert phase_estimation_fisher_information(rho_at, 0.3) < 1e-8
    _ = kerr_unitary  # imported to assert availability for later tasks


def test_fidelity_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        target_state_fidelity(fock_dm(1, CUTOFF), fock_ket(1, CUTOFF + 2))


# --------------------------------------------------------------------------------------
# Baseline optimizer
# --------------------------------------------------------------------------------------


def test_optimizer_matches_brute_force_ceiling() -> None:
    """The headline baseline number, against an independent grid search.

    The maximum fidelity between a single photon and an energy-matched Gaussian state is
    0.47788941, located by brute-force scans that do not use the optimizer at all -- once
    with a symmetry-reduced 2-D grid and once with a full 4-parameter grid that assumes
    no symmetry. The optimizer must reproduce it.
    """
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target), cutoff=CUTOFF, n_budget=1.0
    )
    assert np.isclose(result.score, MAX_GAUSSIAN_FIDELITY_TO_FOCK1, atol=1e-6)


def test_optimum_splits_energy_two_thirds_displacement_one_third_squeezing() -> None:
    """The optimal baseline for a single-photon target is not an arbitrary point.

    It places exactly 2/3 of the energy budget in displacement and 1/3 in squeezing,
    confirmed independently by the brute-force scan.
    """
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target), cutoff=CUTOFF, n_budget=1.0
    )
    assert np.isclose(abs(result.params.alpha) ** 2, 2.0 / 3.0, atol=5e-3)
    assert np.isclose(np.sinh(result.params.r) ** 2, 1.0 / 3.0, atol=5e-3)


def test_optimizer_beats_the_best_coherent_state() -> None:
    """Squeezing has to be used. An optimizer that ignored it would stop at 1/e."""
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target),
        cutoff=CUTOFF, n_budget=1.0, starts=_fast_starts(),
    )
    assert result.score > BEST_COHERENT_FIDELITY_TO_FOCK1 + 0.05


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimizer_is_reproducible_across_seeds(seed: int) -> None:
    """Different random starts must reach the same optimum, or the reported ceiling
    would depend on the seed rather than on the physics."""
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target),
        cutoff=CUTOFF, n_budget=1.0, seed=seed,
    )
    assert np.isclose(result.score, MAX_GAUSSIAN_FIDELITY_TO_FOCK1, atol=1e-6)


@pytest.mark.parametrize("budget", [0.5, 1.0, 2.0])
def test_shell_parameterization_matches_energy_identically(budget: float) -> None:
    """Exact, not within a tolerance band: the constraint holds by construction."""
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, fock_ket(1, CUTOFF)),
        cutoff=CUTOFF, n_budget=budget, starts=_fast_starts(),
    )
    assert abs(result.mean_photon_number - budget) < 1e-9


@pytest.mark.parametrize("budget", [0.5, 1.0, 2.0])
def test_free_parameterization_stays_within_its_tolerance_here(budget: float) -> None:
    """On the preparation task the soft penalty is adequate, since fidelity does not
    grow monotonically with energy. It is not adequate for phase estimation; see
    test_homodyne.test_free_parameterization_overspends_on_a_monotone_score."""
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, fock_ket(1, CUTOFF)),
        cutoff=CUTOFF, n_budget=budget, energy_tolerance=0.02,
        starts=default_starts(budget, extra=3), parameterization="free",
    )
    assert abs(result.mean_photon_number - budget) < 0.05


def test_multi_start_is_doing_work() -> None:
    """Not every start reaches the optimum, so a single-start optimizer would be wrong.

    Asserted rather than assumed: if the landscape were unimodal the multi-start would
    be wasted effort and this test would flag it as removable.
    """
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target), cutoff=CUTOFF, n_budget=1.0
    )
    assert result.converged_fraction < 1.0, "landscape appears unimodal; multi-start unneeded"
    assert result.converged_fraction > 0.1, "most starts failed; optimizer unreliable"


def test_baseline_is_a_lower_bound_on_the_gaussian_optimum() -> None:
    """The reported score must be at least as good as any explicit candidate."""
    target = fock_ket(1, CUTOFF)
    result = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target),
        cutoff=CUTOFF, n_budget=1.0, starts=_fast_starts(),
    )
    for candidate in (
        GaussianParams(alpha_re=1.0),
        GaussianParams(r=np.arcsinh(1.0)),
        GaussianParams(alpha_re=0.8, r=0.5),
    ):
        explicit = target_state_fidelity(to_dm(gaussian_ket(candidate, CUTOFF)), target)
        assert result.score >= explicit - 1e-6


def test_negative_energy_budget_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        optimize_gaussian_baseline(score=lambda rho: 0.0, cutoff=CUTOFF, n_budget=-1.0)


# --------------------------------------------------------------------------------------
# Advantage
# --------------------------------------------------------------------------------------


def test_advantage_is_positive_without_noise() -> None:
    """A non-Gaussian circuit holding the target exactly beats the Gaussian ceiling."""
    target = fock_ket(1, CUTOFF)
    baseline = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(rho, target),
        cutoff=CUTOFF, n_budget=1.0, starts=_fast_starts(),
    )
    assert 1.0 - baseline.score > 0.5


@pytest.mark.slow
def test_advantage_survives_below_the_negativity_threshold() -> None:
    """The decoupling result: ``W_log = 0`` and ``A > 0`` at the same transmissivity.

    A lossy single photon loses all Wigner negativity at eta = 1/2, yet at eta = 0.35 it
    still scores measurably above the best energy-matched Gaussian state on the
    preparation task. Resource survival and task usefulness therefore do not collapse
    together, which is the question RESEARCH_PLAN.md section 2 poses.
    """
    from ngphotonic.metrics.negativity import wigner_log_negativity
    from ngphotonic.metrics.wigner import phase_space_grid, wigner

    eta = 0.35
    target = fock_ket(1, CUTOFF)
    noisy = apply_loss(fock_dm(1, CUTOFF), eta)

    x, p, X, P = phase_space_grid(limit=6.0, points=201)
    assert wigner_log_negativity(wigner(noisy, X, P), x, p) == pytest.approx(0.0, abs=1e-9)

    baseline = optimize_gaussian_baseline(
        score=lambda rho: target_state_fidelity(apply_loss(rho, eta), target),
        cutoff=CUTOFF, n_budget=1.0, starts=_fast_starts(),
    )
    advantage = target_state_fidelity(noisy, target) - baseline.score
    assert advantage > 0.01, f"expected surviving advantage, got {advantage}"
