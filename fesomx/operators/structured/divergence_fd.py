"""
Divergence Operator (Finite Difference): ∇·u

Computes the divergence of a 2D vector field using finite differences
on structured grids with support for Arakawa A, B, and C grid staggering.

Mathematical definition:
    ∇·u = ∂u/∂x + ∂v/∂y

Grid-specific implementations:
    - A-grid: u,v at cell centers → centered differences
    - B-grid: u,v at corners → interpolate or compute at corners
    - C-grid: u at x-faces, v at y-faces → natural divergence form
"""

from typing import Optional, Dict
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from fesomx.core import BaseGrid, StaggerType
from fesomx.core.halo_exchange import StructuredHaloExchanger


def divergence_fd(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    stagger_type: Optional[StaggerType] = None,
    perform_halo_exchange: bool = True,
) -> jnp.ndarray:
    """
    Compute divergence of vector field (u, v).

    Args:
        u: X-component of vector field
        v: Y-component of vector field
        grid: Grid object (contains dx, dy, staggering info)
        partition_info: Partitioning metadata (for halo exchange)
        stagger_type: Override grid's stagger type (A, B, or C)
        perform_halo_exchange: Whether to exchange halos (default: True)

    Returns:
        div: Divergence field ∇·(u,v)

    Grid-specific stencils:
        A-grid (u,v at centers):
            div[j,i] = (u[j,i+1] - u[j,i-1])/(2*dx) +
                       (v[j+1,i] - v[j-1,i])/(2*dy)

        B-grid (u,v at corners):
            Option 1: Interpolate u,v to centers, then standard div
            Option 2: Compute div at corners

        C-grid (u at x-faces, v at y-faces):
            div[j,i] = (u[j,i+1] - u[j,i])/dx +
                       (v[j+1,i] - v[j,i])/dy
            Natural form for flux computations

    Examples:
        >>> # Simple A-grid divergence
        >>> div = divergence(u, v, grid)

        >>> # Without automatic halo exchange
        >>> div = divergence(u, v, grid, perform_halo_exchange=False)

        >>> # Override stagger type
        >>> div = divergence(u, v, grid, stagger_type=StaggerType.C)

    Raises:
        ValueError: If arrays incompatible with grid
        NotImplementedError: If grid type not supported
    """
    # Determine stagger type
    stagger = stagger_type if stagger_type is not None else grid.stagger_type

    # Perform halo exchange if needed
    if perform_halo_exchange and partition_info is not None:
        u_with_halos = _exchange_halos(u, partition_info, grid)
        v_with_halos = _exchange_halos(v, partition_info, grid)
    else:
        u_with_halos = u
        v_with_halos = v

    # Get grid spacing
    dx = grid.dx if hasattr(grid, 'dx') else _compute_dx(grid)
    dy = grid.dy if hasattr(grid, 'dy') else _compute_dy(grid)

    # Dispatch to stagger-specific implementation
    if stagger == StaggerType.A:
        return _divergence_a_grid(u_with_halos, v_with_halos, dx, dy)
    elif stagger == StaggerType.B:
        return _divergence_b_grid(u_with_halos, v_with_halos, dx, dy)
    elif stagger == StaggerType.C:
        return _divergence_c_grid(u_with_halos, v_with_halos, dx, dy)
    else:
        raise NotImplementedError(f"Divergence for {stagger} not implemented")


def _exchange_halos(array: jnp.ndarray, partition_info: Dict, grid: BaseGrid) -> jnp.ndarray:
    """Exchange halos for an array using grid's backend."""
    if not hasattr(grid, 'backend') or grid.backend is None:
        return array  # No backend, skip exchange

    from fesomx.core.backend import get_backend
    backend = grid.backend if hasattr(grid, 'backend') else get_backend('jax_sharding')

    # Create exchanger
    exchanger = StructuredHaloExchanger(backend)

    # Perform exchange with halo_width=1 (sufficient for 2nd order)
    result, _ = exchanger.exchange(array, partition_info, halo_width=1)

    return result


def _compute_dx(grid: BaseGrid) -> float:
    """Compute grid spacing in x-direction."""
    if hasattr(grid, 'vertices') and grid.vertices is not None:
        x = grid.vertices[:, 0]
        return float(np.median(np.diff(np.unique(x))))
    return 1.0  # Default if can't determine


def _compute_dy(grid: BaseGrid) -> float:
    """Compute grid spacing in y-direction."""
    if hasattr(grid, 'vertices') and grid.vertices is not None:
        y = grid.vertices[:, 1]
        return float(np.median(np.diff(np.unique(y))))
    return 1.0  # Default if can't determine


# ============================================================================
# A-Grid: All variables at cell centers
# ============================================================================

def _divergence_a_grid(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute divergence on A-grid (all variables at cell centers).

    Uses centered finite differences:
        ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2*dx)
        ∂v/∂y ≈ (v[j+1] - v[j-1]) / (2*dy)

    Args:
        u: X-velocity component at cell centers (ny, nx)
        v: Y-velocity component at cell centers (ny, nx)
        dx: Grid spacing in x-direction
        dy: Grid spacing in y-direction

    Returns:
        div: Divergence at cell centers (ny, nx)
    """
    if u.shape != v.shape:
        raise ValueError(f"u and v must have same shape: {u.shape} vs {v.shape}")

    ny, nx = u.shape
    div = jnp.zeros_like(u)

    # Interior cells using centered differences
    # Avoid boundaries (assume halos provide boundary data)
    div = div.at[1:-1, 1:-1].set(
        (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
        (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
    )

    return div


# ============================================================================
# B-Grid: Scalars at centers, velocities at corners
# ============================================================================

def _divergence_b_grid(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute divergence on B-grid (velocities at corners).

    B-grid layout:
        - Scalars (φ, h, etc.) at cell centers (i, j)
        - Velocities (u, v) at cell corners (i+½, j+½)

    For divergence at cell centers, we have two options:

    Option 1 (implemented): Average velocities from corners to centers, then div
        u_center[j,i] = (u[j,i] + u[j+1,i] + u[j,i+1] + u[j+1,i+1]) / 4
        v_center[j,i] = (v[j,i] + v[j+1,i] + v[j,i+1] + v[j+1,i+1]) / 4
        div = ∂u_center/∂x + ∂v_center/∂y

    Option 2: Compute divergence at corners (where u,v naturally live)
        div_corner[j+½,i+½] = (u[j+½,i+3/2] - u[j+½,i-½])/dx +
                               (v[j+3/2,i+½] - v[j-½,i+½])/dy

    Args:
        u: X-velocity at corners (ny+1, nx+1)
        v: Y-velocity at corners (ny+1, nx+1)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        div: Divergence at cell centers (ny, nx)
    """
    # Check dimensions (velocities should be on corners)
    ny_corner, nx_corner = u.shape

    # Average velocities from 4 surrounding corners to cell center
    # u[j,i] at corner (j,i) contributes to cells:
    #   (j-1,i-1), (j-1,i), (j,i-1), (j,i)

    ny_center = ny_corner - 1
    nx_center = nx_corner - 1

    # Average u from corners to centers
    u_center = 0.25 * (
        u[:-1, :-1] + u[1:, :-1] +  # Left two corners
        u[:-1, 1:] + u[1:, 1:]       # Right two corners
    )

    # Average v from corners to centers
    v_center = 0.25 * (
        v[:-1, :-1] + v[1:, :-1] +  # Bottom two corners
        v[:-1, 1:] + v[1:, 1:]       # Top two corners
    )

    # Now compute divergence using centered differences at centers
    div = jnp.zeros((ny_center, nx_center))

    # Interior cells
    if ny_center > 2 and nx_center > 2:
        div = div.at[1:-1, 1:-1].set(
            (u_center[1:-1, 2:] - u_center[1:-1, :-2]) / (2 * dx) +
            (v_center[2:, 1:-1] - v_center[:-2, 1:-1]) / (2 * dy)
        )

    return div


# ============================================================================
# C-Grid: Scalars at centers, u at x-faces, v at y-faces
# ============================================================================

def _divergence_c_grid(u: jnp.ndarray, v: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute divergence on C-grid (velocities at faces).

    C-grid layout:
        - Scalars (φ, h, etc.) at cell centers (i, j)
        - u at x-faces (east/west edges): (i+½, j)
        - v at y-faces (north/south edges): (i, j+½)

    This is the NATURAL form for divergence!
        div[j,i] = (u[j,i+1] - u[j,i])/dx + (v[j+1,i] - v[j,i])/dy

    where:
        u[j,i] is velocity at x-face between cells (i-1,j) and (i,j)
        v[j,i] is velocity at y-face between cells (i,j-1) and (i,j)

    Args:
        u: X-velocity at x-faces (ny, nx+1)
        v: Y-velocity at y-faces (ny+1, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        div: Divergence at cell centers (ny, nx)
    """
    # Check dimensions
    # u should be (ny, nx+1) - one extra in x-direction (faces)
    # v should be (ny+1, nx) - one extra in y-direction (faces)

    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"u and v must be 2D arrays")

    ny_u, nx_u = u.shape
    ny_v, nx_v = v.shape

    # Infer number of cell centers
    ny = ny_u  # Same as u's y-dimension
    nx = nx_v  # Same as v's x-dimension

    # Verify dimensions
    if nx_u != nx + 1:
        raise ValueError(f"u x-dimension should be {nx+1}, got {nx_u}")
    if ny_v != ny + 1:
        raise ValueError(f"v y-dimension should be {ny+1}, got {ny_v}")

    # Compute divergence (natural C-grid form)
    div = jnp.zeros((ny, nx))

    # All interior cells (no need to exclude boundaries if halos present)
    div = (u[:, 1:] - u[:, :-1]) / dx + (v[1:, :] - v[:-1, :]) / dy

    return div


# ============================================================================
# JIT-compiled versions (optional, for performance)
# ============================================================================

if JAX_AVAILABLE:
    # Create JIT-compiled versions for performance
    _divergence_a_grid_jit = jit(_divergence_a_grid, static_argnums=(2, 3))
    _divergence_b_grid_jit = jit(_divergence_b_grid, static_argnums=(2, 3))
    _divergence_c_grid_jit = jit(_divergence_c_grid, static_argnums=(2, 3))
else:
    _divergence_a_grid_jit = _divergence_a_grid
    _divergence_b_grid_jit = _divergence_b_grid
    _divergence_c_grid_jit = _divergence_c_grid


# ============================================================================
# Utility functions
# ============================================================================

def divergence_free_field_2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    mode: str = 'sine'
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate a divergence-free vector field for testing.

    ∇·(u,v) = 0 everywhere

    Args:
        x: X-coordinates (ny, nx)
        y: Y-coordinates (ny, nx)
        mode: Field type ('sine', 'rotation', 'double_gyre')

    Returns:
        u: X-component of velocity
        v: Y-component of velocity
    """
    if mode == 'sine':
        # u = sin(πx)cos(πy), v = -cos(πx)sin(πy)
        # ∇·(u,v) = π cos(πx)cos(πy) - π cos(πx)cos(πy) = 0
        u = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * y)
        v = -jnp.cos(jnp.pi * x) * jnp.sin(jnp.pi * y)

    elif mode == 'rotation':
        # Solid body rotation around center
        x_c, y_c = 0.5, 0.5
        u = -(y - y_c)
        v = (x - x_c)

    elif mode == 'double_gyre':
        # Double gyre pattern (oceanography)
        u = -jnp.pi * jnp.sin(jnp.pi * y) * jnp.cos(jnp.pi * x)
        v = jnp.pi * jnp.cos(jnp.pi * y) * jnp.sin(jnp.pi * x)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return u, v
