"""Layered single-mode circuit templates.

Implements the first circuit family of RESEARCH_PLAN.md section 5:

    ``|0> -> resource -> [N . G]^D``

with a resource state, then ``D`` repetitions of a layer combining a Gaussian or
non-Gaussian operation ``G`` with an imperfection channel ``N``.

One structural fact governs how these are used. Pure loss and phase diffusion **commute
exactly** with each other, and each composes with itself (``eta^D``, ``sigma sqrt(D)``),
so a layer built only from those two makes a depth-``D`` circuit indistinguishable from
a single channel. Depth is then not an independent axis, and a "collapse depth" measured
on such a circuit is just a relabelled transmissivity.

A Kerr layer breaks that. ``U = exp(i xi n^2)`` multiplies ``rho_{mn}`` by
``exp(i xi (m^2 - n^2))``, which depends on more than the offset ``m - n`` that loss
preserves, so Kerr and loss do not commute and the layered circuit stops being
reducible. :func:`is_reducible` reports which regime a given layer is in, and the depth
experiment uses it to avoid reporting a reducible sweep as a depth result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backends.reference import apply_unitary, kerr_unitary
from ..noise.loss import apply_loss
from ..noise.phase_diffusion import apply_phase_diffusion

__all__ = ["LayerSpec", "apply_layer", "run_layered", "is_reducible", "collapsed_equivalent"]


@dataclass(frozen=True)
class LayerSpec:
    """One circuit layer: a gate followed by imperfections.

    Ordering within a layer is gate, then loss, then dephasing. Loss and dephasing
    commute so their relative order is immaterial; the gate's position is not.
    """

    eta: float = 1.0
    """Transmissivity of the pure-loss channel. 1.0 is lossless."""

    sigma_phi: float = 0.0
    """Phase-diffusion standard deviation, in radians."""

    kerr_xi: float = 0.0
    """Kerr strength. Non-zero makes the layered circuit irreducible in depth."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.eta <= 1.0:
            raise ValueError(f"eta must lie in [0, 1], got {self.eta}.")
        if self.sigma_phi < 0.0:
            raise ValueError(f"sigma_phi must be non-negative, got {self.sigma_phi}.")


def is_reducible(spec: LayerSpec) -> bool:
    """True if a depth-``D`` chain of this layer collapses to a single channel.

    That happens exactly when there is no Kerr term, because loss and phase diffusion
    commute with each other and each composes with itself. A depth sweep over a
    reducible layer measures nothing that a transmissivity sweep would not.
    """
    return spec.kerr_xi == 0.0


def apply_layer(rho: np.ndarray, spec: LayerSpec) -> np.ndarray:
    """Apply one layer: Kerr, then loss, then phase diffusion."""
    if spec.kerr_xi != 0.0:
        rho = apply_unitary(rho, kerr_unitary(spec.kerr_xi, rho.shape[0]))
    if spec.eta < 1.0:
        rho = apply_loss(rho, spec.eta)
    if spec.sigma_phi > 0.0:
        rho = apply_phase_diffusion(rho, spec.sigma_phi)
    return rho


def run_layered(
    rho: np.ndarray, spec: LayerSpec, depth: int, record: bool = False
) -> np.ndarray | list[np.ndarray]:
    """Run ``depth`` identical layers.

    With ``record=True`` returns the state after every layer including the input, which
    is what the depth sweeps consume; the list has ``depth + 1`` entries.
    """
    if depth < 0:
        raise ValueError(f"depth must be non-negative, got {depth}.")

    trajectory = [np.asarray(rho, dtype=complex)]
    for _ in range(depth):
        trajectory.append(apply_layer(trajectory[-1], spec))
    return trajectory if record else trajectory[-1]


def collapsed_equivalent(spec: LayerSpec, depth: int) -> LayerSpec:
    """The single layer equivalent to ``depth`` repetitions of a reducible ``spec``.

    Transmissivities multiply and phase variances add, so the equivalent layer is
    ``(eta^D, sigma sqrt(D))``. Raises for an irreducible layer, where no single-layer
    equivalent exists.
    """
    if not is_reducible(spec):
        raise ValueError(
            f"Layer with kerr_xi={spec.kerr_xi} is irreducible: Kerr does not commute "
            f"with loss, so no single-layer equivalent exists. This is the regime where "
            f"circuit depth is a genuine axis."
        )
    return LayerSpec(
        eta=spec.eta**depth,
        sigma_phi=spec.sigma_phi * np.sqrt(depth),
        kerr_xi=0.0,
    )
