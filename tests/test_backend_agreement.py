"""Cross-validation of the Piquasso backend against the reference backend.

Two backends are maintained so that each provides an independent check on the other.
The comparison has identified issues on both sides: the reference squeezing
implementation was measured as less accurate (it exponentiated a truncated generator and
now uses the exact even-Fock series), and an Attenuator overflow was found in Piquasso
8.0.1 as ``eta -> 0``.

One condition is asserted throughout this module: **agreement between two backends is
evidence only when they compute differently.** Where both exponentiate the same truncated
operator, as both do for the cubic phase gate, they incur the same error and agree to
machine precision while both deviating from the converged result. Those cases are
validated by cutoff convergence instead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ngphotonic.backends import reference as ref
from ngphotonic.backends import piquasso as pqb
from ngphotonic.noise.loss import apply_loss

pytestmark = pytest.mark.skipif(
    not pqb.PIQUASSO_AVAILABLE, reason="Piquasso not installed (pip install -e '.[sim]')"
)

CUTOFF = 30


@pytest.fixture(autouse=True)
def _quiet_truncation_warnings():
    """Truncation warnings are the subject of their own tests, not noise here."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


# --------------------------------------------------------------------------------------
# Conventions
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_transmissivity_maps_to_attenuator_angle(eta: float) -> None:
    """``eta = cos^2(theta)``, the mapping the whole loss comparison rests on."""
    assert np.isclose(np.cos(pqb.transmissivity_to_theta(eta)) ** 2, eta, atol=1e-12)


def test_piquasso_uses_project_hbar() -> None:
    """hbar = 1 throughout. Piquasso's Config defaults to 2.0, which would redefine x
    and therefore silently change the cubic phase gate."""
    assert pqb.HBAR == 1.0


# --------------------------------------------------------------------------------------
# Independent agreement: different algorithms, so agreement means something
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_fock_state_preparation_agrees(n: int) -> None:
    np.testing.assert_allclose(pqb.fock_dm(n, CUTOFF), ref.fock_dm(n, CUTOFF), atol=1e-12)


@pytest.mark.parametrize("eta", [1e-6, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
def test_loss_channel_agrees_on_single_photon(eta: float) -> None:
    """Piquasso's Attenuator against the project's own Kraus implementation.

    Genuinely independent: Piquasso builds the channel from a beamsplitter-angle
    recursion, the reference from explicit Kraus operators.
    """
    simulated = pqb.run([pqb.vacuum(), pqb.create(), pqb.loss(eta)], cutoff=CUTOFF)
    np.testing.assert_allclose(simulated, apply_loss(ref.fock_dm(1, CUTOFF), eta), atol=1e-10)


@pytest.mark.parametrize("eta", [0.3, 0.7, 0.95])
@pytest.mark.parametrize("n", [2, 3])
def test_loss_agrees_on_higher_fock_states(eta: float, n: int) -> None:
    simulated = pqb.run([pqb.vacuum()] + [pqb.create() for _ in range(n)] + [pqb.loss(eta)],
                        cutoff=CUTOFF)
    np.testing.assert_allclose(simulated, apply_loss(ref.fock_dm(n, CUTOFF), eta), atol=1e-10)


@pytest.mark.parametrize("r, phi", [(0.3, 0.0), (0.5, 0.0), (0.4, 0.9), (0.7, 2.1)])
def test_squeezing_agrees(r: float, phi: float) -> None:
    """Independent: the reference uses the closed-form even-Fock series, Piquasso its
    own internal construction. This comparison measured the reference's previous
    ``expm``-based squeezing as 200x less accurate at cutoff 12."""
    simulated = pqb.run([pqb.vacuum(), pqb.squeezing(r, phi)], cutoff=60)
    np.testing.assert_allclose(simulated, ref.to_dm(ref.squeezed_ket(r, phi, 60)), atol=1e-8)


@pytest.mark.parametrize("xi", [0.1, 0.3, 0.7])
def test_kerr_gate_agrees(xi: float) -> None:
    """Kerr is diagonal in the Fock basis, hence exact at any cutoff in both backends."""
    circuit = [pqb.vacuum(), pqb.squeezing(0.4), pqb.kerr(xi)]
    simulated = pqb.run(circuit, cutoff=60)
    expected = ref.apply_unitary(
        ref.to_dm(ref.squeezed_ket(0.4, 0.0, 60)), ref.kerr_unitary(xi, 60)
    )
    np.testing.assert_allclose(simulated, expected, atol=1e-8)


def test_loss_composes_across_backends() -> None:
    """A depth-10 lossy chain in Piquasso equals one reference channel at eta^10."""
    eta, depth = 0.9, 10
    circuit = [pqb.vacuum(), pqb.create()] + [pqb.loss(eta) for _ in range(depth)]
    simulated = pqb.run(circuit, cutoff=CUTOFF)
    np.testing.assert_allclose(
        simulated, apply_loss(ref.fock_dm(1, CUTOFF), eta**depth), atol=1e-10
    )


# --------------------------------------------------------------------------------------
# Correlated error: agreement here proves nothing
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.1, 0.3])
def test_cubic_phase_agreement_is_correlated_not_independent(gamma: float) -> None:
    """Both backends exponentiate the same truncated ``x^3``, so both are wrong alike.

    At gamma = 0.3 and cutoff 20 the two agree to ~1e-17 while each differs from the
    cutoff-converged answer by ~1e-04, and those two deviations are equal to every
    digit. Asserted here because backend agreement is not evidence of correctness unless
    the backends compute differently.
    """
    small = 20
    simulated = pqb.run([pqb.vacuum(), pqb.cubic_phase(gamma)], cutoff=small)
    reference = ref.apply_unitary(ref.vacuum_dm(small), ref.cubic_phase_unitary(gamma, small))

    # They agree with each other far better than either agrees with the truth.
    agreement = np.abs(simulated - reference).max()
    assert agreement < 1e-14

    converged = ref.apply_unitary(ref.vacuum_dm(260), ref.cubic_phase_unitary(gamma, 260))
    error_ref = np.abs(reference - converged[:small, :small]).max()
    error_pq = np.abs(simulated - converged[:small, :small]).max()

    assert error_ref > 100 * agreement, "expected a shared truncation error"
    assert np.isclose(error_ref, error_pq, rtol=1e-6), "errors should be the same error"


@pytest.mark.slow
@pytest.mark.parametrize("gamma", [0.1, 0.3])
def test_cubic_phase_converges_with_cutoff(gamma: float) -> None:
    """The only meaningful check for cubic phase: convergence, not cross-agreement.

    Required cutoff grows with gamma. At gamma = 0.3, cutoff 20 gives ~1e-04 while
    cutoff 60 reaches ~1e-13.
    """
    converged = ref.apply_unitary(ref.vacuum_dm(260), ref.cubic_phase_unitary(gamma, 260))

    errors = []
    for cutoff in (20, 30, 40, 60):
        simulated = pqb.run([pqb.vacuum(), pqb.cubic_phase(gamma)], cutoff=cutoff)
        errors.append(np.abs(simulated[:20, :20] - converged[:20, :20]).max())

    assert np.all(np.diff(errors) < 0), f"not monotone: {errors}"
    assert errors[-1] < 1e-10


# --------------------------------------------------------------------------------------
# Upstream edge cases
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.0, 1e-12, 1e-9])
def test_attenuator_refuses_transmissivity_that_overflows(eta: float) -> None:
    """Piquasso 8.0.1 returns NaN as eta -> 0; the backend refuses instead.

    Cutoff-dependent upstream: NaN at eta = 0 from cutoff 12, and at eta = 1e-12 by
    cutoff 30, so a coarse sweep can pass while a finer one returns NaN at some points.
    """
    with pytest.raises(ValueError, match="overflows"):
        pqb.loss(eta)


def test_reference_backend_handles_total_loss_that_piquasso_cannot() -> None:
    """eta = 0 is a legitimate endpoint, and the reference is exact there."""
    np.testing.assert_allclose(
        apply_loss(ref.fock_dm(1, CUTOFF), 0.0), ref.vacuum_dm(CUTOFF), atol=1e-12
    )


def test_non_finite_output_is_rejected() -> None:
    """`run` raises on non-finite output rather than returning it.

    Constructed by bypassing the `loss` guard and passing Piquasso the raw Attenuator
    that overflows.
    """
    import piquasso as raw_pq

    bad = [pqb.vacuum(), pqb.create(), ((0,), raw_pq.Attenuator(theta=np.pi / 2))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(FloatingPointError, match="non-finite"):
            pqb.run(bad, cutoff=20)
