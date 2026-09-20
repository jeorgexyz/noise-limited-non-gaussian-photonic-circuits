"""Resource-matched baselines against which advantage is measured."""

from .gaussian import (
    GaussianParams,
    displace_unitary,
    gaussian_ket,
    is_gaussian_channel,
    mean_photon_number_analytic,
)

__all__ = [
    "GaussianParams",
    "gaussian_ket",
    "displace_unitary",
    "mean_photon_number_analytic",
    "is_gaussian_channel",
]
