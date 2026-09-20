"""Optimization of baselines and of circuit parameters."""

from .gaussian_baseline import BaselineResult, default_starts, optimize_gaussian_baseline

__all__ = ["BaselineResult", "default_starts", "optimize_gaussian_baseline"]
