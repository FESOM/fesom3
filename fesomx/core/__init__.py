"""Core module for grid structures and parallel operations."""

from .backend import Backend, get_backend
from .jax_sharding_backend import JAXShardingBackend
from .base_grid import BaseGrid
from .staggering import StaggerType, StaggeredField, StaggerLocation
from .quad_grid import QuadGrid
from .tri_grid import TriGrid
from .mixed_grid import MixedGrid

__all__ = [
    "Backend",
    "get_backend",
    "JAXShardingBackend",
    "BaseGrid",
    "QuadGrid",
    "TriGrid",
    "MixedGrid",
    "StaggerType",
    "StaggeredField",
    "StaggerLocation",
]
