"""Simulation backends.

``reference`` is the pure-NumPy arbiter. A Piquasso backend joins it and is validated
against it.
"""

from . import reference

__all__ = ["reference"]
