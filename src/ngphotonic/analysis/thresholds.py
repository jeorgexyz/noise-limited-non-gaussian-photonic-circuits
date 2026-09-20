"""Threshold and collapse-boundary extraction.

Turns a sweep into the objects RESEARCH_PLAN.md section 3 defines: the last depth at
which a resource survives, and the critical noise parameter at which it stops.

Every threshold is reported as a **bracket** rather than a point estimate. A threshold
quoted more precisely than the underlying metric's noise floor describes that noise
rather than the physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


__all__ = ["Threshold", "last_viable_depth", "critical_parameter", "resolution_floor"]


@dataclass(frozen=True)
class Threshold:
    """A located threshold together with the interval that brackets it."""

    estimate: float
    lower: float
    upper: float
    epsilon: float
    """The metric value treated as the survival boundary."""

    resolved: bool
    """False if the threshold fell outside the search interval."""

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def __str__(self) -> str:
        if not self.resolved:
            return f"unresolved in [{self.lower:g}, {self.upper:g}]"
        return f"{self.estimate:.4f} +/- {self.half_width:.4f}"


def last_viable_depth(values: Sequence[float], epsilon: float) -> int:
    """``D* = max{D : metric(D) > epsilon}``, from a metric sampled at depths 0, 1, 2...

    Returns ``-1`` if depth 0 is already below ``epsilon``. Uses the last index above
    threshold rather than the first index below it, so that a non-monotone metric does
    not report an early dip as the collapse point.
    """
    above = [index for index, value in enumerate(values) if value > epsilon]
    return above[-1] if above else -1


def critical_parameter(
    metric: Callable[[float], float],
    lower: float,
    upper: float,
    epsilon: float,
    iterations: int = 20,
) -> Threshold:
    """Bisect for the parameter where ``metric`` crosses ``epsilon``.

    Assumes ``metric`` is below ``epsilon`` at ``lower`` and above at ``upper`` -- the
    orientation of a transmissivity sweep, where more transmission means more surviving
    resource. Returns an unresolved :class:`Threshold` if the bracket does not actually
    straddle the crossing, rather than silently returning a midpoint.

    ``iterations`` should stop once the bracket is narrower than the metric's noise
    floor; see :func:`resolution_floor`.
    """
    if lower >= upper:
        raise ValueError(f"lower ({lower}) must be below upper ({upper}).")

    low_value, high_value = metric(lower), metric(upper)
    if not (low_value <= epsilon < high_value):
        return Threshold(
            estimate=float("nan"),
            lower=lower,
            upper=upper,
            epsilon=epsilon,
            resolved=False,
        )

    lo, hi = lower, upper
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if metric(mid) > epsilon:
            hi = mid
        else:
            lo = mid

    return Threshold(
        estimate=0.5 * (lo + hi), lower=lo, upper=hi, epsilon=epsilon, resolved=True
    )


def resolution_floor(
    metric: Callable[[float], float], point: float, delta: float, noise: float
) -> float:
    """How finely a threshold near ``point`` can honestly be located.

    Estimates the local slope of ``metric`` and converts the metric's noise floor into
    a parameter uncertainty: ``d(param) ~ noise / |d(metric)/d(param)|``. Bisecting below
    this width resolves quadrature error rather than physics.
    """
    slope = (metric(point + delta) - metric(point - delta)) / (2.0 * delta)
    if abs(slope) < 1e-15:
        return float("inf")
    return float(noise / abs(slope))
