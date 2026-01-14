"""
Laplacian Operator (Finite Volume): ∇²φ

Computes the Laplacian of a 2D scalar field using finite volume method
on unstructured grids.

Mathematical definition:
    ∇²φ = ∇·(∇φ)

Implementation:
    Two-step process:
    1. Compute gradient: ∇φ = (dφ/dx, dφ/dy)
    2. Compute divergence: ∇·(∇φ)
"""

from typing import Optional, Dict
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from fesomx.core import BaseGrid


def laplacian_fv(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    perform_halo_exchange: bool = True,
    **kwargs
) -> jnp.ndarray:
    """
    Compute Laplacian using finite volume method.

    Args:
        phi: Scalar field at cell centers (n_cells,)
        grid: Unstructured grid object
        partition_info: Partitioning metadata
        perform_halo_exchange: Whether to exchange halos

    Returns:
        lapl: Laplacian at cell centers (n_cells,)

    Method:
        ∇²φ = ∇·(∇φ)
        1. Compute gradient using Green-Gauss
        2. Compute divergence of gradient
    """
    # Step 1: Compute gradient
    from .gradient_fv import gradient_fv
    dphidx, dphidy = gradient_fv(
        phi, grid,
        partition_info=partition_info,
        perform_halo_exchange=perform_halo_exchange
    )

    # Step 2: Compute divergence of gradient
    # Note: No halo exchange needed here because gradient_fv already exchanged
    # halos for phi, which allowed it to compute complete dphidx, dphidy fields
    # for all cells (including those at partition boundaries)
    from .divergence_fv import divergence_fv
    lapl = divergence_fv(
        dphidx, dphidy, grid,
        partition_info=partition_info,
        perform_halo_exchange=False  # Already exchanged in gradient step
    )

    return lapl
