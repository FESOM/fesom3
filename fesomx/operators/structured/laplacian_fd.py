"""
Laplacian Operator (Finite Difference): ∇²φ

Computes the Laplacian of a 2D scalar field with support for
Arakawa A, B, and C grid staggering.

Mathematical definition:
    ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

Applications:
    - Heat/diffusion equation: ∂T/∂t = α∇²T
    - Poisson equation: ∇²φ = f
    - Viscosity: ∇²u (momentum diffusion)

Grid-specific implementations:
    - A-grid: φ at centers → ∇²φ at centers (5-point stencil)
    - B-grid: φ at centers → ∇²φ at centers (5-point stencil)
    - C-grid: φ at centers → ∇²φ at centers (5-point stencil)
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


def laplacian_fd(
    phi: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    stagger_type: Optional[StaggerType] = None,
    order: int = 2,
    perform_halo_exchange: bool = True,
) -> jnp.ndarray:
    """
    Compute Laplacian of scalar field: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

    Args:
        phi: Scalar field
        grid: Grid object (contains dx, dy, staggering info)
        partition_info: Partitioning metadata (for halo exchange)
        stagger_type: Override grid's stagger type (A, B, or C)
        order: Accuracy order (2 or 4)
                2 → halo_width=1 (5-point stencil)
                4 → halo_width=2 (9-point stencil)
        perform_halo_exchange: Whether to exchange halos (default: True)

    Returns:
        lapl: Laplacian of phi (∇²φ)

    Stencils:
        2nd order (5-point):
            ∇²φ[j,i] = (φ[j,i+1] - 2φ[j,i] + φ[j,i-1])/dx² +
                       (φ[j+1,i] - 2φ[j,i] + φ[j-1,i])/dy²

        4th order (9-point):
            More accurate but requires halo_width=2
            ∇²φ[j,i] = (-φ[j,i+2] + 16φ[j,i+1] - 30φ[j,i] +
                        16φ[j,i-1] - φ[j,i-2]) / (12dx²) + (y-term)

    Applications:
        >>> # Heat diffusion
        >>> lapl_T = laplacian(temperature, grid)
        >>> dT_dt = alpha * lapl_T

        >>> # 4th order accuracy
        >>> lapl_T = laplacian(temperature, grid, order=4)

    Raises:
        ValueError: If order not supported or arrays incompatible
    """
    if order not in [2, 4]:
        raise ValueError(f"Order must be 2 or 4, got {order}")

    # Determine stagger type
    stagger = stagger_type if stagger_type is not None else grid.stagger_type

    # Perform halo exchange if needed
    halo_width = 1 if order == 2 else 2
    if perform_halo_exchange and partition_info is not None:
        phi_with_halos = _exchange_halos(phi, partition_info, grid, halo_width)
    else:
        phi_with_halos = phi

    # Get grid spacing
    dx = grid.dx if hasattr(grid, 'dx') else _compute_dx(grid)
    dy = grid.dy if hasattr(grid, 'dy') else _compute_dy(grid)

    # Laplacian is same for all stagger types (operates on scalars at centers)
    if order == 2:
        return _laplacian_2nd_order(phi_with_halos, dx, dy)
    else:  # order == 4
        return _laplacian_4th_order(phi_with_halos, dx, dy)


def _exchange_halos(
    array: jnp.ndarray,
    partition_info: Dict,
    grid: BaseGrid,
    halo_width: int
) -> jnp.ndarray:
    """Exchange halos for an array using grid's backend."""
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
# 2nd Order Laplacian (5-point stencil)
# ============================================================================

def _laplacian_2nd_order(phi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute Laplacian using 2nd-order accurate 5-point stencil.

    Stencil pattern:
             φ[j+1,i]
                |
        φ[j,i-1] - φ[j,i] - φ[j,i+1]
                |
             φ[j-1,i]

    Formula:
        ∇²φ = (φ[i+1] - 2φ[i] + φ[i-1])/dx² +
              (φ[j+1] - 2φ[j] + φ[j-1])/dy²

    Truncation error: O(dx², dy²)

    Args:
        phi: Scalar field (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        lapl: Laplacian (ny, nx)
    """
    ny, nx = phi.shape
    lapl = jnp.zeros_like(phi)

    # Interior cells (5-point stencil)
    lapl = lapl.at[1:-1, 1:-1].set(
        (phi[1:-1, 2:] - 2 * phi[1:-1, 1:-1] + phi[1:-1, :-2]) / dx**2 +
        (phi[2:, 1:-1] - 2 * phi[1:-1, 1:-1] + phi[:-2, 1:-1]) / dy**2
    )

    return lapl


# ============================================================================
# 4th Order Laplacian (9-point stencil)
# ============================================================================

def _laplacian_4th_order(phi: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute Laplacian using 4th-order accurate 9-point stencil.

    Stencil pattern:
                φ[j+2,i]
                   |
                φ[j+1,i]
                   |
        φ[j,i-2] - φ[j,i-1] - φ[j,i] - φ[j,i+1] - φ[j,i+2]
                   |
                φ[j-1,i]
                   |
                φ[j-2,i]

    Formula (x-direction):
        ∂²φ/∂x² ≈ (-φ[i+2] + 16φ[i+1] - 30φ[i] + 16φ[i-1] - φ[i-2]) / (12dx²)

    Truncation error: O(dx⁴, dy⁴)

    Requires halo_width=2!

    Args:
        phi: Scalar field (ny, nx)
        dx: Grid spacing in x
        dy: Grid spacing in y

    Returns:
        lapl: Laplacian (ny, nx)
    """
    ny, nx = phi.shape

    if ny < 5 or nx < 5:
        # Grid too small for 4th order, fall back to 2nd order
        return _laplacian_2nd_order(phi, dx, dy)

    lapl = jnp.zeros_like(phi)

    # Interior cells (need 2 cells away in each direction)
    # X-direction second derivative (4th order)
    d2phi_dx2 = (
        -phi[2:-2, 4:] +
        16 * phi[2:-2, 3:-1] -
        30 * phi[2:-2, 2:-2] +
        16 * phi[2:-2, 1:-3] -
        phi[2:-2, :-4]
    ) / (12 * dx**2)

    # Y-direction second derivative (4th order)
    d2phi_dy2 = (
        -phi[4:, 2:-2] +
        16 * phi[3:-1, 2:-2] -
        30 * phi[2:-2, 2:-2] +
        16 * phi[1:-3, 2:-2] -
        phi[:-4, 2:-2]
    ) / (12 * dy**2)

    # Laplacian = sum of second derivatives
    lapl = lapl.at[2:-2, 2:-2].set(d2phi_dx2 + d2phi_dy2)

    return lapl


# ============================================================================
# JIT-compiled versions (optional, for performance)
# ============================================================================

if JAX_AVAILABLE:
    _laplacian_2nd_order_jit = jit(_laplacian_2nd_order, static_argnums=(1, 2))
    _laplacian_4th_order_jit = jit(_laplacian_4th_order, static_argnums=(1, 2))
else:
    _laplacian_2nd_order_jit = _laplacian_2nd_order
    _laplacian_4th_order_jit = _laplacian_4th_order


# ============================================================================
# Utility functions
# ============================================================================

def analytical_laplacian_gaussian(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x0: float = 0.5,
    y0: float = 0.5,
    sigma: float = 0.1,
    amplitude: float = 1.0
) -> jnp.ndarray:
    """
    Analytical Laplacian of 2D Gaussian.

    For φ = A * exp(-r²/(2σ²)) where r² = (x-x0)² + (y-y0)²

    ∇²φ = φ * [ (r²/σ⁴) - (2/σ²) ]
        = φ * [ ((x-x0)² + (y-y0)²)/σ⁴ - 2/σ² ]

    Args:
        x: X-coordinates
        y: Y-coordinates
        x0: Center x
        y0: Center y
        sigma: Width parameter
        amplitude: Peak amplitude

    Returns:
        lapl_phi: Analytical Laplacian
    """
    r2 = (x - x0)**2 + (y - y0)**2
    phi = amplitude * jnp.exp(-r2 / (2 * sigma**2))

    # Laplacian of Gaussian
    lapl_phi = phi * (r2 / sigma**4 - 2 / sigma**2)

    return lapl_phi


def solve_poisson_jacobi(
    f: jnp.ndarray,
    dx: float,
    dy: float,
    n_iterations: int = 1000,
    tolerance: float = 1e-6
) -> jnp.ndarray:
    """
    Solve Poisson equation ∇²φ = f using Jacobi iteration.

    Simple iterative solver for testing Laplacian operator.

    Args:
        f: Right-hand side (source term)
        dx: Grid spacing in x
        dy: Grid spacing in y
        n_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        phi: Solution to ∇²φ = f
    """
    ny, nx = f.shape
    phi = jnp.zeros_like(f)

    dx2 = dx**2
    dy2 = dy**2
    factor = 1.0 / (2 * (1/dx2 + 1/dy2))

    for iteration in range(n_iterations):
        phi_old = phi.copy()

        # Jacobi update
        phi = phi.at[1:-1, 1:-1].set(
            factor * (
                (phi_old[1:-1, 2:] + phi_old[1:-1, :-2]) / dx2 +
                (phi_old[2:, 1:-1] + phi_old[:-2, 1:-1]) / dy2 -
                f[1:-1, 1:-1]
            )
        )

        # Check convergence
        if iteration % 100 == 0:
            residual = jnp.abs(phi - phi_old).max()
            if residual < tolerance:
                print(f"Converged in {iteration} iterations")
                break

    return phi


def heat_diffusion_step(
    T: jnp.ndarray,
    grid: BaseGrid,
    alpha: float,
    dt: float,
    partition_info: Optional[Dict] = None
) -> jnp.ndarray:
    """
    One timestep of heat diffusion: ∂T/∂t = α∇²T

    Forward Euler integration: T^{n+1} = T^n + α*dt*∇²T^n

    Args:
        T: Temperature field at time n
        grid: Grid object
        alpha: Thermal diffusivity
        dt: Timestep
        partition_info: Partition metadata

    Returns:
        T_new: Temperature at time n+1

    Note:
        Stability requires dt < dx²/(4α) for 2D (CFL condition)
    """
    # Compute Laplacian
    lapl_T = laplacian(T, grid, partition_info, perform_halo_exchange=True)

    # Forward Euler step
    T_new = T + alpha * dt * lapl_T

    return T_new
