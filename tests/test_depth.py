"""Depth physics, matched-noise scoring, revivals, and finite-search certificates."""

import subprocess
import sys
import warnings

import numpy as np
import pytest

from ngphotonic.backends.reference import (
    apply_unitary, fock_dm, kerr_unitary, mean_photon_number, squeezed_ket, to_dm,
)
from ngphotonic.circuits.templates import (
    LayerSpec, apply_layer, collapsed_equivalent, is_reducible, run_layered,
)
from ngphotonic.metrics.operational import target_state_fidelity
from ngphotonic.noise.loss import apply_loss
from ngphotonic.noise.phase_diffusion import apply_phase_diffusion
from ngphotonic.sweeps.depth import (
    DepthConfig, certified_horizon, depth_summary, fidelity_effect, sweep_depth,
    vacuum_advantage_bound,
)


@pytest.fixture(autouse=True)
def _quiet_tails():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def test_reference_import_does_not_initialize_optional_backend():
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import sys; import ngphotonic.backends.reference; "
         "assert 'piquasso' not in sys.modules"], capture_output=True, text=True,
    )
    # pytest's pythonpath is not inherited by subprocesses; pass it explicitly.
    if result.returncode and "No module named" in result.stderr:
        import os
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
        result = subprocess.run(result.args, capture_output=True, text=True, env=env)
    assert result.returncode == 0, result.stderr


def test_kerr_layer_order_matches_explicit_channel_construction():
    rho = to_dm(squeezed_ket(.6, 0, 24))
    spec = LayerSpec(eta=.86, sigma_phi=.13, kerr_xi=.2)
    expected = apply_phase_diffusion(
        apply_loss(apply_unitary(rho, kerr_unitary(.2, 24)), .86), .13,
    )
    np.testing.assert_allclose(apply_layer(rho, spec), expected, atol=1e-13)


def test_kerr_loss_chain_is_not_a_lumped_channel():
    rho = to_dm(squeezed_ket(.6, 0, 30))
    spec = LayerSpec(eta=.9, sigma_phi=.1, kerr_xi=.2)
    layered = run_layered(rho, spec, 6)
    lumped = apply_layer(rho, LayerSpec(eta=.9**6, sigma_phi=.1*np.sqrt(6), kerr_xi=1.2))
    distance = np.abs(np.linalg.eigvalsh(layered-lumped)).sum()/2
    assert distance > .01
    assert not is_reducible(spec)
    with pytest.raises(ValueError, match="irreducible"):
        collapsed_equivalent(spec, 6)


@pytest.mark.parametrize("spec", [
    LayerSpec(eta=.8, sigma_phi=.2),
    LayerSpec(eta=1, sigma_phi=.2, kerr_xi=.2),
    LayerSpec(eta=0, sigma_phi=.2, kerr_xi=.2),
    LayerSpec(eta=.8, sigma_phi=.2, kerr_xi=np.pi),
    LayerSpec(eta=.8, sigma_phi=.2, kerr_xi=2*np.pi),
])
def test_reducibility_including_loss_endpoints_and_kerr_rotations(spec):
    ket = np.array([1, 1j, .3, -.2j, .1], dtype=complex)
    rho = to_dm(ket/np.linalg.norm(ket))
    assert is_reducible(spec)
    for d in (0, 1, 5):
        np.testing.assert_allclose(run_layered(rho, spec, d),
                                   apply_layer(rho, collapsed_equivalent(spec, d)), atol=1e-12)


def test_fock_input_hides_kerr_and_is_only_a_control():
    rho = fock_dm(2, 12)
    np.testing.assert_allclose(run_layered(rho, LayerSpec(eta=.9, kerr_xi=.2), 6),
                               apply_loss(rho, .9**6), atol=1e-12)


def test_layer_trajectory_preserves_physicality_and_known_energy_decay():
    source = to_dm(squeezed_ket(.6, 0, 24))
    states = run_layered(source, LayerSpec(eta=.9, sigma_phi=.1, kerr_xi=.2), 12, record=True)
    assert len(states) == 13
    for d, state in enumerate(states):
        assert np.trace(state) == pytest.approx(1, abs=1e-12)
        assert np.linalg.eigvalsh(state).min() > -1e-12
        assert mean_photon_number(state) == pytest.approx(mean_photon_number(source)*.9**d)


@pytest.mark.parametrize("eta,sigma", [(0, .1), (1, 0), (.5, .2), (.9, 0)])
def test_adjoint_fidelity_matches_forward_channel_on_complex_mixed_state(eta, sigma):
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(9, 9)) + 1j*rng.normal(size=(9, 9))
    rho = matrix @ matrix.conj().T
    rho /= np.trace(rho)
    target = rng.normal(size=9) + 1j*rng.normal(size=9)
    target /= np.linalg.norm(target)
    forward = target_state_fidelity(apply_phase_diffusion(apply_loss(rho, eta), sigma), target)
    adjoint = np.einsum("ij,ji->", rho, fidelity_effect(target, eta, sigma)).real
    assert adjoint == pytest.approx(forward, abs=1e-12)


def test_last_viable_depth_handles_revival_and_strict_epsilon():
    result = depth_summary([0, .1, .01, -.1, .03, .01, 0], .01, tail_bound=.005)
    assert result["d_star"] == 4
    assert result["viable_intervals"] == [[1, 1], [4, 4]]
    assert result["revivals"] == 1
    assert result["status"] == "resolved"


@pytest.mark.parametrize("values", [[0, .1, 0], [0, .1, .2], [0, 0, 0]])
def test_finite_sweep_never_silently_certifies_future_collapse(values):
    result = depth_summary(values, .01)
    assert result["d_star"] is None
    assert result["status"] == "horizon_limited"


def test_no_viable_depth_is_distinct_from_depth_zero():
    result = depth_summary([0, .005, -.2], .01, tail_bound=.001)
    assert result["d_star"] is None
    assert result["last_viable_depth_observed"] is None
    assert result["status"] == "never_viable"


@pytest.mark.parametrize("eta", [.8, .9, .95])
def test_automatic_horizon_bounds_all_later_depths(eta):
    budget, epsilon = np.sinh(.6)**2, .01
    depth = certified_horizon(budget, eta, epsilon)
    assert vacuum_advantage_bound(budget, eta, depth - 1) > epsilon
    for later in (depth, depth + 1, depth + 100):
        assert vacuum_advantage_bound(budget, eta, later) <= epsilon


def test_vacuum_bound_covers_layered_vs_feasible_gaussian_witness():
    source = to_dm(squeezed_ket(.6, 0, 30))
    for d in (1, 20, 50):
        ng = run_layered(source, LayerSpec(eta=.8, kerr_xi=.2), d)
        gaussian = apply_loss(source, .8**d)
        # Largest eigenvalue bounds the score difference for ANY target projector.
        max_advantage = np.linalg.eigvalsh(ng - gaussian).max()
        assert max_advantage <= vacuum_advantage_bound(np.sinh(.6)**2, .8, d)


def test_sweep_compares_common_noiseless_target_and_excludes_kerr_from_baseline():
    config = DepthConfig(cutoff=24, grid_points=61, extra_starts=1)
    sweep = sweep_depth(config, .9, .1, 6)
    rows = sweep["rows"]
    assert rows[0]["s_ng"] == pytest.approx(1)
    assert rows[0]["s_gaussian"] == pytest.approx(1, abs=1e-6)
    assert rows[1]["advantage"] > .03
    assert rows[4]["advantage"] < 0
    assert rows[6]["advantage"] > config.epsilon
    assert rows[1]["lumped_trace_distance"] < 1e-12
    assert rows[6]["lumped_trace_distance"] > .01
    assert sweep["collapse"]["viable_intervals"] == [[1, 3], [6, 6]]
    for row in rows:
        assert row["s_gaussian"] >= row["baseline_witness_score"]
        assert row["baseline_energy_error"] < 2e-6


@pytest.mark.parametrize("depth", [-1, .5, True])
def test_invalid_depth_rejected(depth):
    for operation in (run_layered,):
        with pytest.raises(ValueError, match="integer"):
            operation(fock_dm(0, 4), LayerSpec(), depth)
    with pytest.raises(ValueError, match="integer"):
        collapsed_equivalent(LayerSpec(), depth)


@pytest.mark.parametrize("kwargs", [
    {"eta": np.nan}, {"sigma_phi": np.nan}, {"sigma_phi": np.inf}, {"kerr_xi": np.inf},
])
def test_nonfinite_layers_rejected(kwargs):
    with pytest.raises(ValueError):
        LayerSpec(**kwargs)


def test_reducible_depth_sweep_rejected_unless_explicit_control():
    config = DepthConfig(kerr_xi=0, cutoff=20, grid_points=41, extra_starts=0)
    with pytest.raises(ValueError, match="irreducible"):
        sweep_depth(config, .9, .1, 2)
    control = sweep_depth(config, .9, .1, 2, allow_reducible=True)
    assert max(r["advantage"] for r in control["rows"]) < 1e-8
    assert max(r["lumped_trace_distance"] for r in control["rows"]) < 1e-12


def test_lossless_horizon_requires_explicit_depth():
    with pytest.raises(ValueError, match="Lossless"):
        certified_horizon(1, 1, .01)
