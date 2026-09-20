"""Single-mode Gaussian state family used as the comparison baseline.

The most general single-mode pure Gaussian state is a displaced squeezed vacuum

    ``|G(alpha, r, phi)> = D(alpha) S(r e^{i phi}) |0>``

parameterised by four real numbers. Its mean photon number has the closed form

    ``n_bar = |alpha|^2 + sinh^2(r)``

which is what makes the energy-matching constraint cheap to impose during
optimization: the constraint is evaluated analytically rather than by building the
state.

Mixed Gaussian states reachable by this project's channels are obtained by passing a
member of this family through pure loss and phase diffusion, both of which are applied
in Fock space by the existing channel code. Note that phase diffusion is *not* a
Gaussian channel; a dephased Gaussian state is generally non-Gaussian. It is included
here because the baseline must experience the same noise as the circuit it is compared
against, and :func:`is_gaussian_channel` records which configurations keep the baseline
strictly Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from ..backends.reference import (
    annihilation,
    creation,
    squeezed_ket,
    tail_weight,
)

__all__ = [
    "PARAM_BOUNDS",
    "GaussianParams",
    "displace_unitary",
    "gaussian_ket",
    "is_gaussian_channel",
    "mean_photon_number_analytic",
]


@dataclass(frozen=True)
class GaussianParams:
    """Parameters of a single-mode pure Gaussian state.

    ``alpha = alpha_re + i alpha_im`` is the displacement; ``r`` and ``phi`` the
    squeezing magnitude and angle.
    """

    alpha_re: float = 0.0
    alpha_im: float = 0.0
    r: float = 0.0
    phi: float = 0.0

    @property
    def alpha(self) -> complex:
        return complex(self.alpha_re, self.alpha_im)

    @classmethod
    def from_vector(cls, vector) -> GaussianParams:
        """Build from the flat array the optimizer works in."""
        return cls(*(float(v) for v in vector))

    def to_vector(self) -> np.ndarray:
        return np.array([self.alpha_re, self.alpha_im, self.r, self.phi], dtype=float)

    @property
    def mean_photon_number(self) -> float:
        return mean_photon_number_analytic(self)


# Bounds for the optimizer. The squeezing range is deliberate: r beyond ~1.5 needs a
# cutoff large enough that the truncation floor documented in backends.reference starts
# to matter, and the energy constraint makes large r unreachable in practice anyway.
PARAM_BOUNDS = [(-4.0, 4.0), (-4.0, 4.0), (0.0, 1.5), (0.0, 2.0 * np.pi)]


def mean_photon_number_analytic(params: GaussianParams) -> float:
    """``n_bar = |alpha|^2 + sinh^2(r)``, exact and independent of cutoff.

    Used for the energy-matching constraint so that the optimizer does not have to
    construct a state to evaluate feasibility.
    """
    return float(abs(params.alpha) ** 2 + np.sinh(params.r) ** 2)


def displace_unitary(alpha: complex, cutoff: int) -> np.ndarray:
    """Displacement operator ``D(alpha) = exp(alpha a^dag - conj(alpha) a)``.

    Built by matrix exponential on the truncated space, so it is only approximately
    unitary; accuracy is governed by ``cutoff`` against ``|alpha|^2`` in the same way as
    the parity Wigner method. The test suite checks ``D(alpha)|0>`` against the
    independently-constructed coherent state.
    """
    if cutoff < 1:
        raise ValueError(f"cutoff must be >= 1, got {cutoff}.")
    a = annihilation(cutoff)
    adag = creation(cutoff)
    return expm(alpha * adag - np.conj(alpha) * a)


def gaussian_ket(params: GaussianParams, cutoff: int) -> np.ndarray:
    """``D(alpha) S(r e^{i phi}) |0>`` as a normalised ket.

    Squeezing is applied first, then displacement, matching the convention in which
    ``n_bar = |alpha|^2 + sinh^2(r)``.

    The squeezed vacuum comes from the closed-form even-Fock series rather than from
    exponentiating a truncated generator; at cutoff 50 the two differ by 3e-08, and the
    series is the more accurate of the two.
    """
    squeezed = squeezed_ket(params.r, params.phi, cutoff)
    ket = displace_unitary(params.alpha, cutoff) @ squeezed
    norm = np.linalg.norm(ket)
    if norm < 1e-12:
        raise ValueError(f"Gaussian state for {params} is numerically null.")
    return ket / norm


def is_gaussian_channel(sigma_phi: float) -> bool:
    """Whether the noise configuration keeps a Gaussian input Gaussian.

    Pure loss is a Gaussian channel. Phase diffusion is not: it damps Fock coherences
    by ``exp(-sigma^2 (m-n)^2/2)``, which is not a Gaussian operation, so a dephased
    Gaussian state is generally non-Gaussian.

    The baseline still uses it, because a fair comparison subjects the baseline to the
    same physical imperfections as the circuit under test. This function exists so that
    an experiment can state which regime it is reporting rather than leave the
    distinction implicit.
    """
    return sigma_phi == 0.0


def truncation_report(params: GaussianParams, cutoff: int) -> dict[str, float]:
    """Truncation diagnostics for a candidate baseline state.

    ``tail`` should be negligible; if it is not, the optimizer is exploring parameters
    the cutoff cannot represent and the resulting ceiling would be a truncation
    artefact rather than a physical bound.
    """
    ket = gaussian_ket(params, cutoff)
    return {
        "tail": tail_weight(ket),
        "mean_photon_number": mean_photon_number_analytic(params),
        "norm_defect": float(abs(np.linalg.norm(ket) - 1.0)),
    }
