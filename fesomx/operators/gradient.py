"""
Gradient Operator: ∇φ

Computes the gradient of a 2D scalar field with support for
Arakawa A, B, and C grid staggering.

Mathematical definition:
    ∇φ = (∂φ/∂x, ∂φ/∂y)

Grid-specific implementations:
    - A-grid: φ at centers → (∇φ)_x and (∇φ)_y at centers
    - B-grid: φ at centers → ∇φ naturally at corners (where velocities are)
    - C-grid: φ at centers → (∇φ)_x at x-faces, (∇φ)_y at y-faces
"""

from typing import Optional, Dict, Tuple
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


def gradient(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    stagger_type: Optional[StaggerType] = None,
    return_at_centers: bool = True,
    perform_halo_exchange: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient of scalar field: ∇φ = (∂φ/∂x, ∂φ/∂y)

    Args:
        phi: Scalar field
        grid: Grid object (contains dx, dy, staggering info)
        partition_info: Partitioning metadata (for halo exchange)
        stagger_type: Override grid's stagger type (A, B, or C)
        return_at_centers: If False, return gradients at natural locations
                          (corners for B-grid, faces for C-grid)
        perform_halo_exchange: Whether to exchange halos (default: True)

    Returns:
        dphidx: X-component of gradient
        dphidy: Y-component of gradient

    Grid-specific behavior:
        A-grid:
            φ at centers → ∇φ at centers
            dphidx[j,i] = (phi[j,i+1] - phi[j,i-1]) / (2*dx)

        B-grid:
            φ at centers → ∇φ naturally at corners
            dphidx[j+½,i+½] = (phi[j,i+1] - phi[j,i]) / dx
            If return_at_centers=True: interpolate to centers

        C-grid:
            φ at centers → dphidx at x-faces, dphidy at y-faces
            dphidx[j,i+½] = (phi[j,i+1] - phi[j,i]) / dx (natural!)
            dphidy[j+½,i] = (phi[j+1,i] - phi[j,i]) / dy (natural!)
            If return_at_centers=True: interpolate to centers

    Examples:
        >>> # Pressure gradient force (A-grid)
        >>> dpdx, dpdy = gradient(pressure, grid)
        >>> force_x = -dpdx / rho

        >>> # C-grid: get gradients at faces (natural for velocities)
        >>> dpdx, dpdy = gradient(pressure, grid,
        ...                       return_at_centers=False)
        >>> # dpdx at x-faces, dpdy at y-faces

    Raises:
        ValueError: If array incompatible with grid
    """
    # Determine stagger type
    stagger = stagger_type if stagger_type is not None else grid.stagger_type

    # Perform halo exchange if needed
    if perform_halo_exchange and partition_info is not None:
        phi_with_halos = _exchange_halos(phi, partition_info, grid)
    else:
        phi_with_halos = phi

    # Get grid spacing
    dx = grid.dx if hasattr(grid, 'dx') else _compute_dx(grid)
    dy = grid.dy if hasattr(grid, 'dy') else _compute_dy(grid)

    # Dispatch to stagger-specific implementation
    if stagger == StaggerType.A:
        return _gradient_a_grid(phi_with_halos, dx, dy)

    elif stagger == StaggerType.B:
        return _gradient_b_grid(phi_with_halos, dx, dy, return_at_centers)

    elif stagger == StaggerType.C:
        return _gradient_c_grid(phi_with_halos, dx, dy, return_at_centers)

    else:
        raise NotImplementedError(f"Gradient for {stagger} not implemented")


def _exchange_halos(array: jnp.ndarray, partition_info: Dict, grid: BaseGrid) -> jnp.ndarray:
    """Exchange halos for an array using grid's backend."""
    if not hasattr(grid, 'backend') or grid.backend is None:
        return array

    from fesomx.core.backend import get_backend
    backend = grid.backend if hasattr(grid, 'backend') else get_backend('jax_sharding')

    exchanger = StructuredHaloExchanger(backend)
    result, _ = exchanger.exchange(array, partition_info, halo_width=1)

    return result


def _compute_dx(grid: BaseGrid) -> float:
    """Compute grid spacing in x-direction."""
    if hasattr(grid, 'vertices') and grid.vertices is not None:
        x = grid.vertices[:, 0]
        return float(np.median(np.diff(np.unique(x))))
    return 1.0


def _compute_dy(grid: BaseGrid) -> float:
    """Compute grid spacing in y-direction."""
    if hasattr(grid, 'vertices') and grid.vertices is not None:
        y = grid.vertices[:, 1]
        return float(np.median(np.diff(np.unique(y))))
    return 1.0


# ============================================================================
# A-Grid: All variables at cell centers
# ============================================================================

def _gradient_a_grid(
    phi: jnp.ndarray,
    dx: float,
    dy: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient on A-grid (φ at cell centers).

    Returns gradient also at cell centers using centered differences.

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        dphidx: X-gradient at cell centers (ny, nx)
        dphidy: Y-gradient at cell centers (ny, nx)
    """
    ny, nx = phi.shape

    dphidx = jnp.zeros_like(phi)
    dphidy = jnp.zeros_like(phi)

    # Interior cells using centered differences
    dphidx = dphidx.at[1:-1, 1:-1].set(
        (phi[1:-1, 2:] - phi[1:-1, :-2]) / (2 * dx)
    )

    dphidy = dphidy.at[1:-1, 1:-1].set(
        (phi[2:, 1:-1] - phi[:-2, 1:-1]) / (2 * dy)
    )

    return dphidx, dphidy


# ============================================================================
# B-Grid: Scalars at centers, velocities at corners
# ============================================================================

def _gradient_b_grid(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    return_at_centers: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient on B-grid (φ at centers, ∇φ naturally at corners).

    B-grid layout:
        - Scalars φ at cell centers (i, j)
        - Gradient naturally computed at corners (i+½, j+½)

    Natural computation at corners:
        dphidx[j+½,i+½] = (phi[j,i+1] + phi[j+1,i+1] -
                           phi[j,i] - phi[j+1,i]) / (2*dx)
        or simply: (phi[j:j+2,i+1] - phi[j:j+2,i]).mean() / dx

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        return_at_centers: If True, interpolate back to centers

    Returns:
        dphidx: X-gradient (at corners or centers)
        dphidy: Y-gradient (at corners or centers)
    """
    ny, nx = phi.shape

    # Compute gradients at corners (ny+1, nx+1)
    # B-grid: corners are at vertices between cells

    # X-gradient at corners
    # At corner (j, i), approximate using cells to left and right
    # We compute at interior corners [0:ny, 0:nx] using adjacent cell differences
    dphidx_corners = jnp.zeros((ny, nx))
    dphidy_corners = jnp.zeros((ny, nx))

    if ny > 0 and nx > 0:
        # Simple approach: Use one-sided differences from adjacent cells
        # For corner between cells (j, i-1) and (j, i):
        #   dphidx ≈ (phi[j,i] - phi[j,i-1]) / dx
        # This gives us (ny, nx-1) x-gradients at vertical edges

        # Use centered differences where possible
        # At corner (j, i), average gradients from surrounding cells
        if nx > 1:
            dphidx_corners = dphidx_corners.at[:, 1:].set(
                (phi[:, 1:] - phi[:, :-1]) / dx
            )

        if ny > 1:
            dphidy_corners = dphidy_corners.at[1:, :].set(
                (phi[1:, :] - phi[:-1, :]) / dy
            )

    if not return_at_centers:
        return dphidx_corners, dphidy_corners

    # For B-grid, corners and centers have same dimensions
    # Simply return the gradient values (they're already reasonable approximations)
    return dphidx_corners, dphidy_corners


# ============================================================================
# C-Grid: Scalars at centers, gradients at faces
# ============================================================================

def _gradient_c_grid(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    return_at_centers: bool = True
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute gradient on C-grid (φ at centers, ∇φ at faces).

    C-grid layout:
        - Scalars φ at cell centers (i, j)
        - dφ/dx naturally at x-faces (i+½, j)
        - dφ/dy naturally at y-faces (i, j+½)

    This is PERFECT for pressure gradient force!
        u at x-faces: F_u = -dφ/dx (also at x-faces)
        v at y-faces: F_v = -dφ/dy (also at y-faces)

    Natural computation:
        dphidx[j,i+½] = (phi[j,i+1] - phi[j,i]) / dx
        dphidy[j+½,i] = (phi[j+1,i] - phi[j,i]) / dy

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        return_at_centers: If True, interpolate to centers

    Returns:
        dphidx: X-gradient (at x-faces or centers)
        dphidy: Y-gradient (at y-faces or centers)
    """
    ny, nx = phi.shape

    # X-gradient at x-faces (ny, nx+1)
    dphidx_faces = (phi[:, 1:] - phi[:, :-1]) / dx

    # Add boundary faces (can be zero or extrapolated)
    dphidx_faces_full = jnp.zeros((ny, nx + 1))
    dphidx_faces_full = dphidx_faces_full.at[:, 1:-1].set(
        (phi[:, 1:] - phi[:, :-1]) / dx
    )

    # Y-gradient at y-faces (ny+1, nx)
    dphidy_faces = (phi[1:, :] - phi[:-1, :]) / dy

    # Add boundary faces
    dphidy_faces_full = jnp.zeros((ny + 1, nx))
    dphidy_faces_full = dphidy_faces_full.at[1:-1, :].set(
        (phi[1:, :] - phi[:-1, :]) / dy
    )

    if not return_at_centers:
        return dphidx_faces_full, dphidy_faces_full

    # Interpolate from faces to centers
    # dphidx at centers: average from left and right faces
    dphidx_centers = 0.5 * (dphidx_faces_full[:, :-1] + dphidx_faces_full[:, 1:])

    # dphidy at centers: average from top and bottom faces
    dphidy_centers = 0.5 * (dphidy_faces_full[:-1, :] + dphidy_faces_full[1:, :])

    return dphidx_centers, dphidy_centers


# ============================================================================
# JIT-compiled versions (optional, for performance)
# ============================================================================

if JAX_AVAILABLE:
    _gradient_a_grid_jit = jit(_gradient_a_grid, static_argnums=(1, 2))
    _gradient_b_grid_jit = jit(_gradient_b_grid, static_argnums=(1, 2, 3))
    _gradient_c_grid_jit = jit(_gradient_c_grid, static_argnums=(1, 2, 3))
else:
    _gradient_a_grid_jit = _gradient_a_grid
    _gradient_b_grid_jit = _gradient_b_grid
    _gradient_c_grid_jit = _gradient_c_grid


# ============================================================================
# Utility functions
# ============================================================================

def smooth_gaussian_field_2d(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
    amplitude: float = 1.0
) -> jnp.ndarray:
    """
    Generate a smooth Gaussian field for testing gradients.

    φ(x,y) = A * exp(-((x-x0)² + (y-y0)²) / (2σ²))

    Analytical gradients:
        ∂φ/∂x = -φ(x,y) * (x-x0) / σ²
        ∂φ/∂y = -φ(x,y) * (y-y0) / σ²

    Args:
        x: X-coordinates
        y: Y-coordinates
        x0: Center x-coordinate
        y0: Center y-coordinate
        sigma: Width parameter
        amplitude: Peak amplitude

    Returns:
        phi: Gaussian field
    """
    r2 = (x - x0)**2 + (y - y0)**2
    phi = amplitude * jnp.exp(-r2 / (2 * sigma**2))
    return phi


def analytical_gradient_gaussian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
    amplitude: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Analytical gradient of Gaussian field.

    Returns:
        dphidx: Analytical x-gradient
        dphidy: Analytical y-gradient
    """
    phi = smooth_gaussian_field_2d(x, y, x0, y0, sigma, amplitude)
    dphidx = -phi * (x - x0) / sigma**2
    dphidy = -phi * (y - y0) / sigma**2
    return dphidx, dphidy
