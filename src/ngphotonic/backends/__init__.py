"""Simulation backends.

``reference``
    Pure NumPy, no optional dependencies. Written to be obviously correct rather than
    fast, and exact at parameter endpoints the production backend cannot reach.

``piquasso``
    The production backend. Importing this module is safe without Piquasso installed;
    it sets ``PIQUASSO_AVAILABLE = False`` and raises a useful error on use.

The two are held against each other in tests/test_backend_agreement.py. Note the
caveat recorded there: agreement is only evidence where the backends compute
differently, which is not true of every gate.
"""

from . import piquasso, reference

__all__ = ["reference", "piquasso"]
