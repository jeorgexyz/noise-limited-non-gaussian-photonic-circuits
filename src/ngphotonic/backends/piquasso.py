"""Piquasso-backed Fock simulation.

The production backend. Where :mod:`ngphotonic.backends.reference` is written to be
obviously correct, this one is written to be usable for sweeps; the validation suite
holds the two against each other on every circuit both can express.

Module naming: this file is ``ngphotonic.backends.piquasso`` and imports the
third-party ``piquasso``. Python 3 uses absolute imports, so there is no shadowing. The
name follows the layout in RESEARCH_PLAN.md.

Conventions pinned against Piquasso 8.0.1
-----------------------------------------
``hbar``
    Piquasso's ``Config`` defaults to ``hbar = 2.0``. This project uses ``hbar = 1``
    throughout, so :data:`HBAR` is passed explicitly on every simulator. It does not
    affect Fock-basis density matrices, but it does define ``x``, and therefore the
    cubic phase gate.

``Attenuator(theta)``
    Transmissivity is ``eta = cos^2(theta)``, verified numerically, so the project
    parameter maps as ``theta = arccos(sqrt(eta))``. ``FockSimulator`` does *not*
    accept Piquasso's ``Loss`` instruction -- ``Attenuator`` is the supported channel.
    It also takes ``mean_thermal_excitation``, which is the hook for the V3 thermal
    channel.

``Kerr(xi)``
    ``U = exp(i xi n^2)``, verified on an off-diagonal element.

``CubicPhase(gamma)``
    ``U = exp(i x^3 gamma / (3 hbar))``. At ``hbar = 1`` this is the reference
    backend's :func:`~ngphotonic.backends.reference.cubic_phase_unitary`.

``Squeezing(r, phi)``
    ``S(z) = exp((conj(z) a^2 - z a^dag^2)/2)``, identical to the reference.

Cutoff
------
For a single mode, ``Config(cutoff=N)`` gives an ``N x N`` density matrix indexed
``|0> .. |N-1>``, matching the reference backend. For several modes Piquasso's cutoff
is a bound on *total* particle number, not a per-mode dimension, so multimode cutoffs
are not directly comparable to a per-mode truncation. Single-mode work is unaffected.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "PIQUASSO_AVAILABLE",
    "HBAR",
    "run",
    "transmissivity_to_theta",
    "vacuum",
    "create",
    "loss",
    "kerr",
    "cubic_phase",
    "squeezing",
    "displacement",
    "fock_dm",
    "require_piquasso",
]

HBAR = 1.0

# Below this transmissivity Piquasso 8.0.1's Attenuator overflows (see `loss`).
MIN_PIQUASSO_TRANSMISSIVITY = 1e-6

try:
    import piquasso as pq

    PIQUASSO_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where the extra is absent
    pq = None
    PIQUASSO_AVAILABLE = False


def require_piquasso() -> None:
    """Raise a useful error if the optional backend is missing."""
    if not PIQUASSO_AVAILABLE:
        raise ImportError(
            "The Piquasso backend requires the 'sim' extra: pip install -e '.[sim]'. "
            "The reference backend in ngphotonic.backends.reference needs no extras."
        )


# --------------------------------------------------------------------------------------
# Parameter translation
# --------------------------------------------------------------------------------------


def transmissivity_to_theta(eta: float) -> float:
    """Project transmissivity ``eta`` -> Piquasso ``Attenuator`` angle.

    ``eta = cos^2(theta)``, so ``theta = arccos(sqrt(eta))``.
    """
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"Transmissivity eta must lie in [0, 1], got {eta}.")
    return float(np.arccos(np.sqrt(eta)))


# --------------------------------------------------------------------------------------
# Instruction helpers
# --------------------------------------------------------------------------------------
#
# Each returns a ``(modes, instruction)`` pair for :func:`run`, so a circuit reads as a
# plain list and the Piquasso parameter conventions stay in this module.


def vacuum(modes: tuple[int, ...] = (0,)) -> tuple[tuple[int, ...], Any]:
    require_piquasso()
    return modes, pq.Vacuum()


def create(mode: int = 0) -> tuple[tuple[int, ...], Any]:
    """Add one photon to ``mode`` (Piquasso renormalises afterwards)."""
    require_piquasso()
    return (mode,), pq.Create()


def loss(eta: float, mode: int = 0, mean_thermal_excitation: float = 0.0):
    """Pure-loss channel at transmissivity ``eta``.

    Refuses ``eta`` below :data:`MIN_PIQUASSO_TRANSMISSIVITY`. Piquasso 8.0.1 forms
    ``tan(theta)**(2k)`` internally, and as ``eta -> 0`` the angle approaches ``pi/2``,
    so that term overflows and the density matrix comes back as NaN. The failure is
    cutoff-dependent -- at ``eta = 0`` it appears from cutoff 12, at ``eta = 1e-12``
    by cutoff 30, while ``eta >= 1e-6`` was clean at every cutoff tested. A coarse sweep
    can therefore pass while a finer one returns NaN at a few grid points.

    The reference backend handles the whole closed interval including ``eta = 0``
    exactly, so total loss should be routed there.
    """
    require_piquasso()
    if eta < MIN_PIQUASSO_TRANSMISSIVITY:
        raise ValueError(
            f"Piquasso's Attenuator overflows for eta={eta} (theta -> pi/2 makes "
            f"tan(theta)**(2k) diverge), returning NaN at larger cutoffs. Use "
            f"eta >= {MIN_PIQUASSO_TRANSMISSIVITY}, or the reference backend, which "
            f"is exact down to eta = 0."
        )
    return (mode,), pq.Attenuator(
        theta=transmissivity_to_theta(eta),
        mean_thermal_excitation=mean_thermal_excitation,
    )


def kerr(xi: float, mode: int = 0):
    """Kerr gate ``exp(i xi n^2)``."""
    require_piquasso()
    return (mode,), pq.Kerr(xi=xi)


def cubic_phase(gamma: float, mode: int = 0):
    """Cubic phase gate ``exp(i gamma x^3 / 3)`` at ``hbar = 1``."""
    require_piquasso()
    return (mode,), pq.CubicPhase(gamma=gamma)


def squeezing(r: float, phi: float = 0.0, mode: int = 0):
    """Squeezing ``S(r e^{i phi})``."""
    require_piquasso()
    return (mode,), pq.Squeezing(r=r, phi=phi)


def displacement(r: float, phi: float = 0.0, mode: int = 0):
    """Displacement by ``alpha = r e^{i phi}``."""
    require_piquasso()
    return (mode,), pq.Displacement(r=r, phi=phi)


# --------------------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------------------


def run(
    instructions: Sequence[tuple[Iterable[int], Any]],
    cutoff: int,
    modes: int = 1,
) -> np.ndarray:
    """Execute a circuit and return its Fock-basis density matrix.

    ``instructions`` is a sequence of ``(modes, instruction)`` pairs as produced by the
    helpers above. The returned array uses the same indexing as the reference backend
    for a single mode, so the two can be compared elementwise.
    """
    require_piquasso()
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}.")

    with pq.Program() as program:
        for target, instruction in instructions:
            pq.Q(*target) | instruction

    simulator = pq.FockSimulator(d=modes, config=pq.Config(cutoff=cutoff, hbar=HBAR))
    state = simulator.execute(program).state
    density_matrix = np.asarray(state.density_matrix, dtype=complex)

    # Upstream numerical failures surface as NaN/inf rather than as exceptions. Raise
    # here so they cannot propagate into a sweep aggregate.
    if not np.all(np.isfinite(density_matrix)):
        raise FloatingPointError(
            f"Piquasso returned a non-finite density matrix at cutoff {cutoff}. This "
            f"usually means a parameter reached an unstable edge of an instruction -- "
            f"see `loss` for the Attenuator's behaviour as eta -> 0."
        )
    return density_matrix


def fock_dm(n: int, cutoff: int) -> np.ndarray:
    """``|n><n|`` prepared through Piquasso, for backend comparison."""
    require_piquasso()
    if not 0 <= n < cutoff:
        raise ValueError(f"Fock level {n} does not fit in a cutoff-{cutoff} space.")
    return run([vacuum()] + [create() for _ in range(n)], cutoff=cutoff)
