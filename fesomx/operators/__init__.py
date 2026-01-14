"""
Operators Module - Mathematical operators for grid computations

This module provides fundamental differential operators with support for
both structured grids (finite difference) and unstructured grids (finite volume).

Operators:
    - divergence: ∇·u (velocity divergence)
    - gradient: ∇φ (scalar gradient)
    - laplacian: ∇²φ (scalar Laplacian)

Grid Support:
    - Structured grids (QuadGrid): Finite difference with A/B/C staggering
    - Unstructured grids (TriGrid, MixedGrid): Finite volume method

All operators:
    - Automatically dispatch to appropriate implementation
    - Handle halo exchange automatically
    - Work with partitioned grids
    - JAX-native (JIT-compilable)
"""

from typing import Optional, Dict, Tuple
import jax.numpy as jnp

from fesomx.core import BaseGrid

# Import implementations
from .structured import divergence_fd, gradient_fd, laplacian_fd
from .unstructured import divergence_fv, gradient_fv, laplacian_fv


def divergence(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    **kwargs
) -> jnp.ndarray:
    """
    Compute divergence: ∇·u = ∂u/∂x + ∂v/∂y

    Automatically selects appropriate method based on grid type:
    - Structured grids → Finite difference
    - Unstructured grids → Finite volume

    Args:
        u: X-component of vector field
        v: Y-component of vector field
        grid: Grid object (QuadGrid, TriGrid, or MixedGrid)
        partition_info: Partitioning metadata (for halo exchange)
        **kwargs: Implementation-specific options

    Returns:
        div: Divergence field

    Examples:
        >>> # Works on any grid type
        >>> div = divergence(u, v, grid)

        >>> # With specific options for structured grids
        >>> div = divergence(u, v, quad_grid, stagger_type=StaggerType.C)
    """
    if grid.is_structured():
        return divergence_fd(u, v, grid, partition_info, **kwargs)
    else:
        return divergence_fv(u, v, grid, partition_info, **kwargs)


def gradient(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    **kwargs
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient: ∇φ = (∂φ/∂x, ∂φ/∂y)

    Automatically selects appropriate method based on grid type:
    - Structured grids → Finite difference
    - Unstructured grids → Green-Gauss (finite volume)

    Args:
        phi: Scalar field
        grid: Grid object
        partition_info: Partitioning metadata
        **kwargs: Implementation-specific options

    Returns:
        dphidx: X-component of gradient
        dphidy: Y-component of gradient

    Examples:
        >>> dpdx, dpdy = gradient(pressure, grid)
    """
    if grid.is_structured():
        return gradient_fd(phi, grid, partition_info, **kwargs)
    else:
        return gradient_fv(phi, grid, partition_info, **kwargs)


def laplacian(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    **kwargs
) -> jnp.ndarray:
    """
    Compute Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

    Automatically selects appropriate method based on grid type:
    - Structured grids → Finite difference (2nd or 4th order)
    - Unstructured grids → Finite volume (∇·∇φ)

    Args:
        phi: Scalar field
        grid: Grid object
        partition_info: Partitioning metadata
        **kwargs: Implementation-specific options (e.g., order=4 for structured)

    Returns:
        lapl: Laplacian field

    Examples:
        >>> # 2nd order on structured grid
        >>> lapl = laplacian(phi, quad_grid)

        >>> # 4th order on structured grid
        >>> lapl = laplacian(phi, quad_grid, order=4)

        >>> # Finite volume on unstructured grid
        >>> lapl = laplacian(phi, tri_grid)
    """
    if grid.is_structured():
        return laplacian_fd(phi, grid, partition_info, **kwargs)
    else:
        return laplacian_fv(phi, grid, partition_info, **kwargs)


__all__ = [
    'divergence',
    'gradient',
    'laplacian',
]

__version__ = '0.2.0'  # Updated for modular architecture
