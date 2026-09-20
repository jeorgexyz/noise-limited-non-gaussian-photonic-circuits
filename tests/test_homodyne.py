"""Validation of homodyne detection, phase-estimation Fisher information, and the
exact energy-shell parameterization of the Gaussian baseline.

Three closed-form anchors are used:

- position wavefunctions against orthonormality and against the analytic vacuum,
- the Fisher information against the quantum Fisher information ``4 Var(n)``, which
  upper-bounds it for any measurement,
- and the coherent state, for which homodyne is known to saturate that bound.

Two structural facts are asserted because the phase-estimation comparison depends on
them: a Fock state is invariant under phase rotation, and the Gaussian family is closed
under phase rotation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    apply_unitary,
    cat_ket,
    coherent_ket,
    fock_dm,
    fock_ket,
    squeezed_ket,
    to_dm,
)
from ngphotonic.baselines.gaussian import GaussianParams, gaussian_ket
from ngphotonic.metrics.homodyne import (
    fock_wavefunctions,
    homodyne_distribution,
    homodyne_fisher_information,
    phase_rotation,
    quadrature_grid,
    quantum_fisher_information_pure,
)
from ngphotonic.metrics.operational import target_state_fidelity
from ngphotonic.noise.loss import apply_loss
from ngphotonic.optimization.gaussian_baseline import (
    optimize_gaussian_baseline,
    shell_to_params,
)

CUTOFF = 30


def _quiet(build):
    """Construct a state at module scope without tripping the truncation warning.

    Parametrize decorators evaluate at import time, outside the autouse fixture, so the
    suppression has to be explicit here. The warning itself is exercised by its own
    tests in test_negativity.py.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build()



@pytest.fixture(autouse=True)
def _quiet_truncation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


@pytest.fixture(scope="module")
def x():
    return quadrature_grid()


def _fisher(ket, theta, x, cutoff=CUTOFF):
    rho = to_dm(ket)
    return homodyne_fisher_information(
        lambda th: apply_unitary(rho, phase_rotation(th, cutoff)), theta, x
    )


# --------------------------------------------------------------------------------------
# Wavefunctions and distributions
# --------------------------------------------------------------------------------------


def test_wavefunctions_are_orthonormal(x) -> None:
    psi = fock_wavefunctions(8, x)
    overlap = np.trapezoid(psi[:, None, :] * psi[None, :, :], x, axis=2)
    np.testing.assert_allclose(overlap, np.eye(8), atol=1e-10)


def test_vacuum_distribution_matches_closed_form(x) -> None:
    """``P(x) = exp(-x^2)/sqrt(pi)`` for the vacuum at hbar = 1."""
    probability = homodyne_distribution(fock_dm(0, CUTOFF), x)
    np.testing.assert_allclose(probability, np.exp(-(x**2)) / np.sqrt(np.pi), atol=1e-12)


@pytest.mark.parametrize(
    "state",
    [
        fock_dm(0, CUTOFF),
        fock_dm(3, CUTOFF),
        to_dm(coherent_ket(1.2, CUTOFF)),
        _quiet(lambda: to_dm(squeezed_ket(0.6, 0.0, CUTOFF))),
        to_dm(cat_ket(1.5, "even", CUTOFF)),
    ],
)
def test_distribution_is_normalised_and_non_negative(state, x) -> None:
    probability = homodyne_distribution(state, x)
    assert np.all(probability >= 0.0)
    assert np.isclose(np.trapezoid(probability, x), 1.0, atol=1e-9)


# --------------------------------------------------------------------------------------
# Structural facts the comparison rests on
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
@pytest.mark.parametrize("theta", [0.0, 0.4, 1.3, 2.7])
def test_fock_states_are_invariant_under_phase_rotation(n: int, theta: float) -> None:
    """``exp(-i theta n)|n><n|exp(i theta n) = |n><n|`` exactly.

    The reason a Fock probe carries no phase information under *any* measurement.
    """
    rho = fock_dm(n, CUTOFF)
    rotated = apply_unitary(rho, phase_rotation(theta, CUTOFF))
    np.testing.assert_allclose(rotated, rho, atol=1e-14)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("theta", [0.0, 0.6, 1.9])
def test_fock_probes_have_exactly_zero_fisher_information(n: int, theta: float, x) -> None:
    """Follows from invariance: the strongest resource in the preparation task is the
    weakest possible probe here."""
    assert _fisher(fock_ket(n, CUTOFF), theta, x) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("theta", [0.3, 1.1])
def test_gaussian_family_is_closed_under_phase_rotation(theta: float) -> None:
    """``U(theta) D(alpha) S(r, phi) = D(alpha e^{-i theta}) S(r, phi - 2 theta)``.

    This is what lets the baseline be optimized at fixed ``theta`` without loss of
    generality: the optimizer reaches every orientation through alpha and phi.
    """
    params = GaussianParams(0.9, 0.4, 0.5, 1.2)
    rotated = phase_rotation(theta, CUTOFF) @ gaussian_ket(params, CUTOFF)

    alpha = params.alpha * np.exp(-1j * theta)
    equivalent = GaussianParams(
        alpha.real, alpha.imag, params.r, (params.phi - 2 * theta) % (2 * np.pi)
    )
    np.testing.assert_allclose(rotated, gaussian_ket(equivalent, CUTOFF), atol=1e-12)


@pytest.mark.parametrize("eta, theta", [(0.7, 0.4), (0.3, 1.2)])
def test_phase_rotation_commutes_with_loss(eta: float, theta: float) -> None:
    """So the order of phase imprinting and loss is immaterial."""
    rho = to_dm(coherent_ket(1.0, CUTOFF))
    rotate_first = apply_loss(apply_unitary(rho, phase_rotation(theta, CUTOFF)), eta)
    lose_first = apply_unitary(apply_loss(rho, eta), phase_rotation(theta, CUTOFF))
    np.testing.assert_allclose(rotate_first, lose_first, atol=1e-14)


# --------------------------------------------------------------------------------------
# Fisher information against the quantum bound
# --------------------------------------------------------------------------------------


def test_quantum_fisher_information_is_four_times_variance() -> None:
    ket = coherent_ket(1.2, CUTOFF)
    assert np.isclose(quantum_fisher_information_pure(ket), 4.0 * 1.44, atol=1e-6)


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_fock_quantum_fisher_information_is_zero(n: int) -> None:
    assert quantum_fisher_information_pure(fock_ket(n, CUTOFF)) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "ket",
    [
        coherent_ket(1.2, CUTOFF),
        _quiet(lambda: squeezed_ket(0.6, 0.0, CUTOFF)),
        cat_ket(1.5, "even", CUTOFF),
    ],
)
def test_classical_fisher_information_respects_the_quantum_bound(ket, x) -> None:
    """``F_homodyne <= QFI`` for every phase, by the Cramer-Rao hierarchy."""
    bound = quantum_fisher_information_pure(ket)
    best = max(_fisher(ket, float(theta), x) for theta in np.linspace(0, np.pi, 17))
    assert best <= bound * (1 + 1e-6) + 1e-9


def test_homodyne_saturates_the_bound_for_a_coherent_state(x) -> None:
    """A known result: homodyne is optimal for coherent-state phase estimation, so the
    classical and quantum Fisher informations coincide at the best quadrature."""
    ket = coherent_ket(1.2, CUTOFF)
    best = max(_fisher(ket, float(theta), x) for theta in np.linspace(0, np.pi, 65))
    assert np.isclose(best, quantum_fisher_information_pure(ket), rtol=2e-3)


def test_squeezing_beats_displacement_at_equal_energy(x) -> None:
    """Why the baseline wins this task: at fixed energy a squeezed probe carries far
    more phase information than a coherent one."""
    budget = 1.0
    coherent = coherent_ket(np.sqrt(budget), CUTOFF)
    squeezed = squeezed_ket(float(np.arcsinh(np.sqrt(budget))), 0.0, CUTOFF)
    assert quantum_fisher_information_pure(squeezed) > 3 * quantum_fisher_information_pure(coherent)


# --------------------------------------------------------------------------------------
# Exact energy matching
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget", [0.5, 1.0, 2.2])
@pytest.mark.parametrize("s", [0.0, 0.25, 0.5, 1.0])
def test_shell_parameterization_matches_energy_exactly(budget: float, s: float) -> None:
    params = shell_to_params([s, 0.7, 1.3], budget)
    assert np.isclose(params.mean_photon_number, budget, atol=1e-12)


def test_shell_and_free_agree_on_the_preparation_task() -> None:
    """Where the free parameterization already respects the budget, the two must agree.

    Cross-validates the shell coordinates against the earlier results.
    """
    target = fock_ket(1, CUTOFF)
    scores = {}
    for mode in ("shell", "free"):
        result = optimize_gaussian_baseline(
            score=lambda rho: target_state_fidelity(rho, target),
            cutoff=CUTOFF,
            n_budget=1.0,
            parameterization=mode,
        )
        scores[mode] = result.score
        assert np.isclose(result.mean_photon_number, 1.0, atol=0.03)
    assert np.isclose(scores["shell"], scores["free"], atol=1e-6)


@pytest.mark.slow
def test_free_parameterization_overspends_on_a_monotone_score(x) -> None:
    """Pinned flaw: a soft energy penalty is unsafe when the score grows with energy.

    Fisher information increases monotonically with energy, so the free-parameterization
    optimizer finds it worthwhile to pay the quadratic penalty and exceed the budget --
    measured at about 15% on this task, which breaks the resource matching the whole
    comparison depends on. The shell parameterization cannot do this by construction.
    """
    def fisher_score(rho):
        return homodyne_fisher_information(
            lambda th: apply_unitary(rho, phase_rotation(th, CUTOFF)), 0.0, x
        )

    free = optimize_gaussian_baseline(
        score=fisher_score, cutoff=CUTOFF, n_budget=1.0, parameterization="free"
    )
    shell = optimize_gaussian_baseline(
        score=fisher_score, cutoff=CUTOFF, n_budget=1.0, parameterization="shell"
    )

    assert free.mean_photon_number > 1.05, "expected the free mode to overspend"
    assert np.isclose(shell.mean_photon_number, 1.0, atol=1e-9)
    # Overspending buys a higher score, which is exactly why it is not a fair baseline.
    assert free.score > shell.score


@pytest.mark.slow
@pytest.mark.parametrize("budget", [1.0, 2.0])
def test_optimized_baseline_matches_the_squeezed_vacuum_closed_form(budget: float, x) -> None:
    """The phase-estimation baseline has an analytic optimum, and the optimizer finds it.

    For squeezed vacuum ``Var(n) = 2 n (1 + n)``, so ``QFI = 8 n (1 + n)``, and homodyne
    saturates it at the right quadrature. The energy-matched Gaussian optimum is
    therefore pure squeezing with no displacement, at exactly that value: 16 at
    ``n = 1`` and 48 at ``n = 2``.

    This is the check that caught the earlier soft-penalty baseline, which reported
    19.59 at ``n = 1`` -- above the closed form, because it was overspending the energy
    budget rather than outperforming it.
    """
    cutoff = 60

    def fisher_score(rho):
        return homodyne_fisher_information(
            lambda th: apply_unitary(rho, phase_rotation(th, cutoff)), 0.0, x
        )

    result = optimize_gaussian_baseline(
        score=fisher_score, cutoff=cutoff, n_budget=budget, parameterization="shell"
    )

    closed_form = 8.0 * budget * (1.0 + budget)
    assert result.score == pytest.approx(closed_form, rel=3e-3)
    assert np.isclose(result.mean_photon_number, budget, atol=1e-9)
    # The optimum is pure squeezing: displacement does not help this task.
    assert abs(result.params.alpha) < 0.05


def test_invalid_parameterization_rejected() -> None:
    with pytest.raises(ValueError, match="parameterization"):
        optimize_gaussian_baseline(
            score=lambda rho: 0.0, cutoff=CUTOFF, n_budget=1.0, parameterization="nope"
        )
