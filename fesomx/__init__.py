"""FESOMx: JAX-based parallel grid computations with halo exchange.

A framework for parallel grid computations on structured and unstructured
grids with support for halo exchange, targeting ocean/climate modeling
applications.

Grid Types:
    - QuadGrid: Structured quadrilateral grids
    - TriGrid: Unstructured triangular grids
    - MixedGrid: Mixed quadrilateral/triangular grids

Operators:
    - divergence: Vector field divergence
    - gradient: Scalar field gradient
    - laplacian: Scalar field Laplacian

Staggering:
    - StaggerType.A: All variables at cell centers
    - StaggerType.B: Velocities at corners, scalars at centers
    - StaggerType.C: Velocities at faces, scalars at centers
"""

from fesomx.core import (
    Backend,
    get_backend,
    JAXShardingBackend,
    BaseGrid,
    QuadGrid,
    TriGrid,
    MixedGrid,
    StaggerType,
    StaggeredField,
    StaggerLocation,
)
from fesomx.operators import divergence, gradient, laplacian

__version__ = "0.1.0"
__all__ = [
    # Backend
    "Backend",
    "get_backend",
    "JAXShardingBackend",
    # Grids
    "BaseGrid",
    "QuadGrid",
    "TriGrid",
    "MixedGrid",
    # Staggering
    "StaggerType",
    "StaggeredField",
    "StaggerLocation",
    # Operators
    "divergence",
    "gradient",
    "laplacian",
]
