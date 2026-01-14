"""
Unstructured Grid Operators (Finite Volume)

Finite volume operators for irregular unstructured grids (triangles, mixed meshes).
Uses Green-Gauss gradient reconstruction and edge-based flux calculations.

Includes boundary treatment for ocean grids with coastlines and islands:
- Ghost cell extrapolation for improved boundary accuracy
- Land mask support for internal boundaries
"""

from .divergence_fv import divergence_fv
from .gradient_fv import gradient_fv
from .laplacian_fv import laplacian_fv
from .boundary_treatment import (
    identify_boundary_cells,
    compute_ghost_values,
    create_extended_neighbors,
)

__all__ = [
    'divergence_fv',
    'gradient_fv',
    'laplacian_fv',
    'identify_boundary_cells',
    'compute_ghost_values',
    'create_extended_neighbors',
]
