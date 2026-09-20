"""Circuit families.

See RESEARCH_PLAN.md section 5. Only the single-mode layered family exists so far.
"""

from .templates import (
    LayerSpec,
    apply_layer,
    collapsed_equivalent,
    is_reducible,
    run_layered,
)

__all__ = [
    "LayerSpec",
    "apply_layer",
    "run_layered",
    "is_reducible",
    "collapsed_equivalent",
]
