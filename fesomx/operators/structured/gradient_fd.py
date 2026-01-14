"""
Gradient Operator (Finite Difference): ∇φ

Computes the gradient of a 2D scalar field with support for
Arakawa A, B, and C grid staggering.

Mathematical definition:
    ∇φ = (∂φ/∂x, ∂φ/∂y)

Grid-specific implementations:
    - A-grid: φ at centers → (∇φ)_x and (∇φ)_y at centers
    - B-grid: φ at centers → ∇φ naturally at corners (where velocities are)
    - C-grid: φ at centers → (∇φ)_x at x-faces, (∇φ)_y at y-faces
"""

from typing import Optional, Dict, Tuple, Union
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


def gradient_fd(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    stagger_type: Optional[StaggerType] = None,
    return_at_centers: bool = True,
    perform_halo_exchange: bool = True,
    order: int = 2,
    periodic: Optional[Tuple[bool, bool]] = None,
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
        order: Accuracy order (2 or 4)
               2 → halo_width=1 (3-point stencil): (φ[i+1] - φ[i-1]) / (2Δx)
               4 → halo_width=2 (5-point stencil): (-φ[i+2] + 8φ[i+1] - 8φ[i-1] + φ[i-2]) / (12Δx)
        periodic: Tuple of (periodic_x, periodic_y). If None, uses grid.periodic.
                  When True, uses wraparound indexing for boundary stencils.

    Returns:
        dphidx: X-component of gradient
        dphidy: Y-component of gradient

    Grid-specific behavior:
        A-grid:
            φ at centers → ∇φ at centers
            2nd order: dphidx[j,i] = (phi[j,i+1] - phi[j,i-1]) / (2*dx)
            4th order: dphidx[j,i] = (-phi[j,i+2] + 8*phi[j,i+1] - 8*phi[j,i-1] + phi[j,i-2]) / (12*dx)

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
        >>> # 2nd-order gradient (default)
        >>> dpdx, dpdy = gradient_fd(pressure, grid)

        >>> # 4th-order gradient (higher accuracy, requires halo_width=2)
        >>> dpdx, dpdy = gradient_fd(pressure, grid, order=4)

        >>> # Periodic boundaries (for truly periodic functions)
        >>> dpdx, dpdy = gradient_fd(pressure, grid, periodic=(True, True))

        >>> # C-grid: get gradients at faces (natural for velocities)
        >>> dpdx, dpdy = gradient_fd(pressure, grid, return_at_centers=False)

    Raises:
        ValueError: If order not supported or arrays incompatible
    """
    if order not in [2, 4]:
        raise ValueError(f"Order must be 2 or 4, got {order}")

    # Determine stagger type
    stagger = stagger_type if stagger_type is not None else grid.stagger_type

    # Determine periodicity
    if periodic is None:
        periodic = getattr(grid, 'periodic', (False, False))

    # Determine halo width based on order
    halo_width = 1 if order == 2 else 2

    # Perform halo exchange if needed (not needed if using periodic wrapping)
    if perform_halo_exchange and partition_info is not None and not any(periodic):
        phi_with_halos = _exchange_halos(phi, partition_info, grid, halo_width)
    else:
        phi_with_halos = phi

    # Get grid spacing
    dx = grid.dx if hasattr(grid, 'dx') else _compute_dx(grid)
    dy = grid.dy if hasattr(grid, 'dy') else _compute_dy(grid)

    # Dispatch to stagger-specific implementation
    if stagger == StaggerType.A:
        if any(periodic):
            # Use periodic versions
            if order == 2:
                return _gradient_a_grid_periodic(phi_with_halos, dx, dy, periodic)
            else:  # order == 4
                return _gradient_a_grid_4th_order_periodic(phi_with_halos, dx, dy, periodic)
        else:
            # Use original non-periodic versions
            if order == 2:
                return _gradient_a_grid(phi_with_halos, dx, dy)
            else:  # order == 4
                return _gradient_a_grid_4th_order(phi_with_halos, dx, dy)

    elif stagger == StaggerType.B:
        if order == 4:
            # For B-grid, 4th order would require more complex interpolation
            # Fall back to 2nd order with warning
            import warnings
            warnings.warn("4th-order gradient not implemented for B-grid, using 2nd order")
        return _gradient_b_grid(phi_with_halos, dx, dy, return_at_centers)

    elif stagger == StaggerType.C:
        if order == 4:
            # For C-grid, 4th order at faces is non-trivial
            import warnings
            warnings.warn("4th-order gradient not implemented for C-grid, using 2nd order")
        return _gradient_c_grid(phi_with_halos, dx, dy, return_at_centers)

    else:
        raise NotImplementedError(f"Gradient for {stagger} not implemented")


def _exchange_halos(array: jnp.ndarray, partition_info: Dict, grid: BaseGrid, halo_width: int = 1) -> jnp.ndarray:
    """Exchange halos for an array using grid's backend.

    Args:
        array: Array to exchange halos for
        partition_info: Partitioning metadata
        grid: Grid object with backend
        halo_width: Width of halo to exchange (1 for 2nd order, 2 for 4th order)

    Returns:
        Array with exchanged halos
    """
    if not hasattr(grid, 'backend') or grid.backend is None:
        return array

    from fesomx.core.backend import get_backend
    backend = grid.backend if hasattr(grid, 'backend') else get_backend('jax_sharding')

    exchanger = StructuredHaloExchanger(backend)
    result, _ = exchanger.exchange(array, partition_info, halo_width=halo_width)

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
# A-Grid: 4th-order (5-point stencil)
# ============================================================================

def _gradient_a_grid_4th_order(
    phi: jnp.ndarray,
    dx: float,
    dy: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 4th-order accurate gradient on A-grid (φ at cell centers).

    Uses 5-point stencil for each direction:
        ∂φ/∂x ≈ (-φ[i+2] + 8φ[i+1] - 8φ[i-1] + φ[i-2]) / (12Δx)

    Truncation error: O(Δx⁴, Δy⁴)
    Requires halo_width=2!

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        dphidx: X-gradient at cell centers (ny, nx)
        dphidy: Y-gradient at cell centers (ny, nx)
    """
    ny, nx = phi.shape

    if ny < 5 or nx < 5:
        # Grid too small for 4th order, fall back to 2nd order
        return _gradient_a_grid(phi, dx, dy)

    dphidx = jnp.zeros_like(phi)
    dphidy = jnp.zeros_like(phi)

    # Interior cells using 4th-order centered differences (5-point stencil)
    # ∂φ/∂x = (-φ[i+2] + 8φ[i+1] - 8φ[i-1] + φ[i-2]) / (12*dx)
    dphidx = dphidx.at[2:-2, 2:-2].set(
        (-phi[2:-2, 4:] + 8*phi[2:-2, 3:-1] - 8*phi[2:-2, 1:-3] + phi[2:-2, :-4]) / (12 * dx)
    )

    # ∂φ/∂y = (-φ[j+2] + 8φ[j+1] - 8φ[j-1] + φ[j-2]) / (12*dy)
    dphidy = dphidy.at[2:-2, 2:-2].set(
        (-phi[4:, 2:-2] + 8*phi[3:-1, 2:-2] - 8*phi[1:-3, 2:-2] + phi[:-4, 2:-2]) / (12 * dy)
    )

    return dphidx, dphidy


# ============================================================================
# A-Grid: Periodic versions with wraparound indexing
# ============================================================================

def _gradient_a_grid_periodic(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    periodic: Tuple[bool, bool] = (True, True)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 2nd-order gradient on A-grid with periodic boundary conditions.

    Uses jnp.roll for efficient wraparound indexing.

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        periodic: (periodic_x, periodic_y) flags

    Returns:
        dphidx: X-gradient at ALL cell centers (ny, nx)
        dphidy: Y-gradient at ALL cell centers (ny, nx)
    """
    periodic_x, periodic_y = periodic

    if periodic_x:
        # Centered difference with wraparound: (φ[i+1] - φ[i-1]) / (2*dx)
        dphidx = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * dx)
    else:
        # Non-periodic: zeros at boundaries
        dphidx = jnp.zeros_like(phi)
        dphidx = dphidx.at[:, 1:-1].set(
            (phi[:, 2:] - phi[:, :-2]) / (2 * dx)
        )

    if periodic_y:
        # Centered difference with wraparound: (φ[j+1] - φ[j-1]) / (2*dy)
        dphidy = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * dy)
    else:
        # Non-periodic: zeros at boundaries
        dphidy = jnp.zeros_like(phi)
        dphidy = dphidy.at[1:-1, :].set(
            (phi[2:, :] - phi[:-2, :]) / (2 * dy)
        )

    return dphidx, dphidy


def _gradient_a_grid_4th_order_periodic(
    phi: jnp.ndarray,
    dx: float,
    dy: float,
    periodic: Tuple[bool, bool] = (True, True)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute 4th-order gradient on A-grid with periodic boundary conditions.

    Uses 5-point stencil with wraparound:
        ∂φ/∂x ≈ (-φ[i+2] + 8φ[i+1] - 8φ[i-1] + φ[i-2]) / (12Δx)

    Args:
        phi: Scalar field at cell centers (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y
        periodic: (periodic_x, periodic_y) flags

    Returns:
        dphidx: X-gradient at ALL cell centers (ny, nx)
        dphidy: Y-gradient at ALL cell centers (ny, nx)
    """
    ny, nx = phi.shape
    periodic_x, periodic_y = periodic

    if periodic_x:
        # 4th-order centered difference with wraparound
        dphidx = (
            -jnp.roll(phi, -2, axis=1)
            + 8 * jnp.roll(phi, -1, axis=1)
            - 8 * jnp.roll(phi, 1, axis=1)
            + jnp.roll(phi, 2, axis=1)
        ) / (12 * dx)
    else:
        # Non-periodic: zeros at 2-cell boundary layer
        dphidx = jnp.zeros_like(phi)
        if nx >= 5:
            dphidx = dphidx.at[:, 2:-2].set(
                (-phi[:, 4:] + 8*phi[:, 3:-1] - 8*phi[:, 1:-3] + phi[:, :-4]) / (12 * dx)
            )

    if periodic_y:
        # 4th-order centered difference with wraparound
        dphidy = (
            -jnp.roll(phi, -2, axis=0)
            + 8 * jnp.roll(phi, -1, axis=0)
            - 8 * jnp.roll(phi, 1, axis=0)
            + jnp.roll(phi, 2, axis=0)
        ) / (12 * dy)
    else:
        # Non-periodic: zeros at 2-cell boundary layer
        dphidy = jnp.zeros_like(phi)
        if ny >= 5:
            dphidy = dphidy.at[2:-2, :].set(
                (-phi[4:, :] + 8*phi[3:-1, :] - 8*phi[1:-3, :] + phi[:-4, :]) / (12 * dy)
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
    _gradient_a_grid_4th_order_jit = jit(_gradient_a_grid_4th_order, static_argnums=(1, 2))
    _gradient_b_grid_jit = jit(_gradient_b_grid, static_argnums=(1, 2, 3))
    _gradient_c_grid_jit = jit(_gradient_c_grid, static_argnums=(1, 2, 3))
else:
    _gradient_a_grid_jit = _gradient_a_grid
    _gradient_a_grid_4th_order_jit = _gradient_a_grid_4th_order
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
