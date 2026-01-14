"""
Gradient Operator Analytical Verification Tests

Mathematical verification of the gradient operator ∇φ = (∂φ/∂x, ∂φ/∂y)
using multiple test functions with known analytical gradients.

Test Functions and Their Analytical Gradients:
==============================================

1. LINEAR: f(x,y) = ax + by
   ∂f/∂x = a  (constant)
   ∂f/∂y = b  (constant)
   → Should be EXACT for any centered difference scheme

2. QUADRATIC: f(x,y) = x² + y²
   ∂f/∂x = 2x
   ∂f/∂y = 2y
   → 2nd-order centered differences should be EXACT (Taylor: f''Δx²/6 = 0 for quadratic)

3. CUBIC: f(x,y) = x³ + y³
   ∂f/∂x = 3x²
   ∂f/∂y = 3y²
   → Tests non-trivial curvature, O(h²) error expected

4. SINUSOIDAL: f(x,y) = sin(kx)cos(ky), k = 2π
   ∂f/∂x = k·cos(kx)cos(ky)
   ∂f/∂y = -k·sin(kx)sin(ky)
   → Smooth, periodic, standard test case

5. GAUSSIAN: f(x,y) = exp(-r²/(2σ²)), r² = (x-x₀)² + (y-y₀)²
   ∂f/∂x = -f·(x-x₀)/σ²
   ∂f/∂y = -f·(y-y₀)/σ²
   → Localized, smooth

6. POLYNOMIAL: f(x,y) = x²y + xy²
   ∂f/∂x = 2xy + y²
   ∂f/∂y = x² + 2xy
   → Mixed terms

Convergence Theory:
==================
For centered finite differences on uniform grid:
   2nd order: Error ~ O(h²) = O(Δx², Δy²)
   4th order: Error ~ O(h⁴)

The truncation error for 2nd-order centered difference:
   ∂φ/∂x ≈ (φ(x+h) - φ(x-h))/(2h)
         = φ'(x) + φ'''(x)h²/6 + O(h⁴)

So error is proportional to 3rd derivative. For:
   - Linear: φ''' = 0 → EXACT
   - Quadratic: φ''' = 0 → EXACT
   - Cubic: φ''' = 6 → O(h²) error
   - Higher order: O(h²) error
"""

import numpy as np
import pytest
import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False

from fesomx import QuadGrid, StaggerType
from fesomx.operators.structured.gradient_fd import gradient_fd


# =============================================================================
# Test Function Definitions with Analytical Gradients
# =============================================================================

@dataclass
class AnalyticalTestCase:
    """Container for test function and its analytical gradient."""
    name: str
    f: Callable[[np.ndarray, np.ndarray], np.ndarray]
    df_dx: Callable[[np.ndarray, np.ndarray], np.ndarray]
    df_dy: Callable[[np.ndarray, np.ndarray], np.ndarray]
    exact_for_2nd_order: bool = False  # True if 2nd-order FD should be exact
    exact_for_4th_order: bool = False  # True if 4th-order FD should be exact
    description: str = ""


# Test Function 1: LINEAR
def linear_f(x, y, a=2.0, b=3.0):
    """f(x,y) = ax + by"""
    return a * x + b * y

def linear_df_dx(x, y, a=2.0, b=3.0):
    """∂f/∂x = a"""
    return np.full_like(x, a)

def linear_df_dy(x, y, a=2.0, b=3.0):
    """∂f/∂y = b"""
    return np.full_like(y, b)

LINEAR = AnalyticalTestCase(
    name="Linear",
    f=linear_f,
    df_dx=linear_df_dx,
    df_dy=linear_df_dy,
    exact_for_2nd_order=True,
    exact_for_4th_order=True,
    description="f(x,y) = 2x + 3y"
)


# Test Function 2: QUADRATIC
def quadratic_f(x, y):
    """f(x,y) = x² + y²"""
    return x**2 + y**2

def quadratic_df_dx(x, y):
    """∂f/∂x = 2x"""
    return 2 * x

def quadratic_df_dy(x, y):
    """∂f/∂y = 2y"""
    return 2 * y

QUADRATIC = AnalyticalTestCase(
    name="Quadratic",
    f=quadratic_f,
    df_dx=quadratic_df_dx,
    df_dy=quadratic_df_dy,
    exact_for_2nd_order=True,
    exact_for_4th_order=True,
    description="f(x,y) = x² + y²"
)


# Test Function 3: CUBIC
def cubic_f(x, y):
    """f(x,y) = x³ + y³"""
    return x**3 + y**3

def cubic_df_dx(x, y):
    """∂f/∂x = 3x²"""
    return 3 * x**2

def cubic_df_dy(x, y):
    """∂f/∂y = 3y²"""
    return 3 * y**2

CUBIC = AnalyticalTestCase(
    name="Cubic",
    f=cubic_f,
    df_dx=cubic_df_dx,
    df_dy=cubic_df_dy,
    exact_for_2nd_order=False,
    exact_for_4th_order=True,  # ∂⁴(x³)/∂x⁴ = 0, so 4th-order FD is exact
    description="f(x,y) = x³ + y³"
)


# Test Function 4: SINUSOIDAL (periodic)
def sinusoidal_f(x, y, k=2*np.pi):
    """f(x,y) = sin(kx)cos(ky)"""
    return np.sin(k * x) * np.cos(k * y)

def sinusoidal_df_dx(x, y, k=2*np.pi):
    """∂f/∂x = k·cos(kx)cos(ky)"""
    return k * np.cos(k * x) * np.cos(k * y)

def sinusoidal_df_dy(x, y, k=2*np.pi):
    """∂f/∂y = -k·sin(kx)sin(ky)"""
    return -k * np.sin(k * x) * np.sin(k * y)

SINUSOIDAL = AnalyticalTestCase(
    name="Sinusoidal",
    f=sinusoidal_f,
    df_dx=sinusoidal_df_dx,
    df_dy=sinusoidal_df_dy,
    exact_for_2nd_order=False,
    exact_for_4th_order=False,
    description="f(x,y) = sin(2πx)cos(2πy)"
)


# Test Function 5: GAUSSIAN
def gaussian_f(x, y, x0=0.5, y0=0.5, sigma=0.2):
    """f(x,y) = exp(-r²/(2σ²))"""
    r2 = (x - x0)**2 + (y - y0)**2
    return np.exp(-r2 / (2 * sigma**2))

def gaussian_df_dx(x, y, x0=0.5, y0=0.5, sigma=0.2):
    """∂f/∂x = -f·(x-x₀)/σ²"""
    f = gaussian_f(x, y, x0, y0, sigma)
    return -f * (x - x0) / sigma**2

def gaussian_df_dy(x, y, x0=0.5, y0=0.5, sigma=0.2):
    """∂f/∂y = -f·(y-y₀)/σ²"""
    f = gaussian_f(x, y, x0, y0, sigma)
    return -f * (y - y0) / sigma**2

GAUSSIAN = AnalyticalTestCase(
    name="Gaussian",
    f=gaussian_f,
    df_dx=gaussian_df_dx,
    df_dy=gaussian_df_dy,
    exact_for_2nd_order=False,
    exact_for_4th_order=False,
    description="f(x,y) = exp(-r²/(2σ²)), σ=0.2"
)


# Test Function 6: POLYNOMIAL (mixed terms)
def polynomial_f(x, y):
    """f(x,y) = x²y + xy²"""
    return x**2 * y + x * y**2

def polynomial_df_dx(x, y):
    """∂f/∂x = 2xy + y²"""
    return 2 * x * y + y**2

def polynomial_df_dy(x, y):
    """∂f/∂y = x² + 2xy"""
    return x**2 + 2 * x * y

POLYNOMIAL = AnalyticalTestCase(
    name="Polynomial",
    f=polynomial_f,
    df_dx=polynomial_df_dx,
    df_dy=polynomial_df_dy,
    exact_for_2nd_order=False,
    exact_for_4th_order=False,  # Has non-zero 4th derivatives in mixed terms
    description="f(x,y) = x²y + xy²"
)


# All test functions
ALL_TEST_FUNCTIONS = [LINEAR, QUADRATIC, CUBIC, SINUSOIDAL, GAUSSIAN, POLYNOMIAL]


# =============================================================================
# Helper Functions
# =============================================================================

def create_grid(nx: int, ny: int, domain: Tuple[float, float, float, float] = (0., 1., 0., 1.)):
    """Create QuadGrid with cell-center coordinates."""
    grid = QuadGrid("test", nx=nx, ny=ny, stagger_type=StaggerType.A)
    grid.create_mesh(domain=domain)
    grid.compute_neighbors(halo_width=2)
    grid.dx = (domain[1] - domain[0]) / nx
    grid.dy = (domain[3] - domain[2]) / ny
    return grid


def get_cell_centers(grid):
    """Get x, y coordinates at cell centers reshaped to (ny, nx)."""
    centers = grid.get_cell_centers()
    x = centers[:, 0].reshape(grid.ny, grid.nx)
    y = centers[:, 1].reshape(grid.ny, grid.nx)
    return x, y


def compute_errors(numerical, analytical, exclude_boundary=True):
    """Compute L2 and max errors, optionally excluding boundary cells."""
    if exclude_boundary:
        # Interior only (exclude 1-cell boundary layer)
        num = numerical[1:-1, 1:-1]
        ana = analytical[1:-1, 1:-1]
    else:
        num = numerical
        ana = analytical

    diff = np.array(num) - np.array(ana)
    l2_error = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))
    return l2_error, max_error


def compute_weighted_l2(numerical, analytical, cell_areas):
    """Compute area-weighted L2 error for unstructured grids.

    This properly accounts for varying cell sizes in TriGrid/MixedGrid.

    Args:
        numerical: Numerical gradient values (1D array for unstructured)
        analytical: Analytical gradient values
        cell_areas: Area of each cell

    Returns:
        Area-weighted L2 error: sqrt(sum(error² * area) / total_area)
    """
    error = np.array(numerical) - np.array(analytical)
    total_area = np.sum(cell_areas)
    return np.sqrt(np.sum(error**2 * cell_areas) / total_area)


def compute_errors_detailed(numerical, analytical, grid_shape=None, boundary_width=1):
    """Compute errors for interior and boundary separately.

    For structured grids (QuadGrid), separates interior from boundary cells.
    For unstructured grids (grid_shape=None), returns global error only.

    Args:
        numerical: Numerical gradient values
        analytical: Analytical gradient values
        grid_shape: (ny, nx) for structured grids, None for unstructured
        boundary_width: Number of boundary cell layers to separate

    Returns:
        dict with keys: 'interior_L2', 'boundary_L2', 'global_L2',
                       'interior_Linf', 'boundary_Linf'
    """
    numerical = np.array(numerical)
    analytical = np.array(analytical)

    global_l2 = np.sqrt(np.mean((numerical.ravel() - analytical.ravel())**2))
    global_linf = np.max(np.abs(numerical.ravel() - analytical.ravel()))

    if grid_shape is not None:
        ny, nx = grid_shape
        bw = boundary_width

        # Check if we have enough cells for interior
        if ny > 2*bw and nx > 2*bw:
            # Interior region
            interior_num = numerical[bw:-bw, bw:-bw].ravel()
            interior_ana = analytical[bw:-bw, bw:-bw].ravel()
            interior_error = interior_num - interior_ana
            interior_l2 = np.sqrt(np.mean(interior_error**2))
            interior_linf = np.max(np.abs(interior_error))

            # Boundary region (all cells within bw of edge)
            mask = np.ones((ny, nx), dtype=bool)
            mask[bw:-bw, bw:-bw] = False
            boundary_num = numerical[mask]
            boundary_ana = analytical[mask]
            boundary_error = boundary_num - boundary_ana
            boundary_l2 = np.sqrt(np.mean(boundary_error**2)) if len(boundary_error) > 0 else 0.0
            boundary_linf = np.max(np.abs(boundary_error)) if len(boundary_error) > 0 else 0.0
        else:
            # Grid too small for interior/boundary separation
            interior_l2 = global_l2
            interior_linf = global_linf
            boundary_l2 = 0.0
            boundary_linf = 0.0
    else:
        # Unstructured grid - no boundary separation
        interior_l2 = global_l2
        interior_linf = global_linf
        boundary_l2 = 0.0
        boundary_linf = 0.0

    return {
        'interior_L2': interior_l2,
        'boundary_L2': boundary_l2,
        'global_L2': global_l2,
        'interior_Linf': interior_linf,
        'boundary_Linf': boundary_linf,
    }


# =============================================================================
# Pytest Tests: Exactness for Linear/Quadratic
# =============================================================================

class TestGradientExactness:
    """Test that gradient is exact for linear and quadratic functions."""

    def test_linear_exact(self):
        """2nd-order gradient of linear function should be exact (to machine precision)."""
        grid = create_grid(16, 16)
        x, y = get_cell_centers(grid)

        phi = LINEAR.f(x, y)
        exact_dx = LINEAR.df_dx(x, y)
        exact_dy = LINEAR.df_dy(x, y)

        num_dx, num_dy = gradient_fd(phi, grid, order=2)

        l2_x, _ = compute_errors(num_dx, exact_dx)
        l2_y, _ = compute_errors(num_dy, exact_dy)

        # Should be exact to machine precision (~1e-14)
        assert l2_x < 1e-10, f"Linear ∂f/∂x not exact: L2={l2_x:.2e}"
        assert l2_y < 1e-10, f"Linear ∂f/∂y not exact: L2={l2_y:.2e}"

    def test_quadratic_exact(self):
        """2nd-order gradient of quadratic function should be exact."""
        grid = create_grid(16, 16)
        x, y = get_cell_centers(grid)

        phi = QUADRATIC.f(x, y)
        exact_dx = QUADRATIC.df_dx(x, y)
        exact_dy = QUADRATIC.df_dy(x, y)

        num_dx, num_dy = gradient_fd(phi, grid, order=2)

        l2_x, _ = compute_errors(num_dx, exact_dx)
        l2_y, _ = compute_errors(num_dy, exact_dy)

        # Should be exact to machine precision
        assert l2_x < 1e-10, f"Quadratic ∂f/∂x not exact: L2={l2_x:.2e}"
        assert l2_y < 1e-10, f"Quadratic ∂f/∂y not exact: L2={l2_y:.2e}"

    def test_cubic_exact_4th_order_interior(self):
        """4th-order gradient of cubic on interior (excluding 2-cell boundary).

        The 4th-order stencil for ∂(x³)/∂x gives exact result (3x²) since the
        truncation error term involves 5th derivative which is 0 for cubics.
        """
        grid = create_grid(32, 32)  # Larger grid for meaningful interior
        x, y = get_cell_centers(grid)

        phi = CUBIC.f(x, y)
        exact_dx = CUBIC.df_dx(x, y)
        exact_dy = CUBIC.df_dy(x, y)

        # Use non-periodic 4th order, then check interior only
        num_dx, num_dy = gradient_fd(phi, grid, order=4, periodic=(False, False))

        # Exclude 2-cell boundary for 4th-order stencil
        interior_num_dx = np.array(num_dx)[2:-2, 2:-2]
        interior_exact_dx = exact_dx[2:-2, 2:-2]
        interior_num_dy = np.array(num_dy)[2:-2, 2:-2]
        interior_exact_dy = exact_dy[2:-2, 2:-2]

        l2_x = np.sqrt(np.mean((interior_num_dx - interior_exact_dx)**2))
        l2_y = np.sqrt(np.mean((interior_num_dy - interior_exact_dy)**2))

        # Should be exact to machine precision in interior
        assert l2_x < 1e-10, f"Cubic 4th-order ∂f/∂x not exact in interior: L2={l2_x:.2e}"
        assert l2_y < 1e-10, f"Cubic 4th-order ∂f/∂y not exact in interior: L2={l2_y:.2e}"


# =============================================================================
# Pytest Tests: Convergence Order
# =============================================================================

class TestGradientConvergence:
    """Test convergence rates for gradient operator."""

    @pytest.mark.parametrize("test_func", [CUBIC, SINUSOIDAL, GAUSSIAN])
    def test_2nd_order_convergence(self, test_func):
        """2nd-order gradient should converge at O(h²)."""
        resolutions = [8, 16, 32, 64]
        errors = []

        for n in resolutions:
            grid = create_grid(n, n)
            x, y = get_cell_centers(grid)

            phi = test_func.f(x, y)
            exact_dx = test_func.df_dx(x, y)

            num_dx, _ = gradient_fd(phi, grid, order=2)
            l2, _ = compute_errors(num_dx, exact_dx)
            errors.append(l2)

        # Compute convergence orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i] > 1e-14 and errors[i+1] > 1e-14:
                order = np.log(errors[i] / errors[i+1]) / np.log(2)
                orders.append(order)

        avg_order = np.mean(orders)
        assert avg_order > 1.8, f"{test_func.name}: Expected O(h²), got {avg_order:.2f}"

    @pytest.mark.parametrize("test_func", [SINUSOIDAL])
    def test_4th_order_convergence(self, test_func):
        """4th-order gradient should converge at O(h⁴) for periodic functions.

        Note: Only truly periodic functions (like sinusoidal with k=2π) work
        correctly with periodic boundary conditions. Cubic is excluded because
        its 4th-order FD is exact, and Gaussian/Polynomial are non-periodic.
        """
        resolutions = [8, 16, 32]
        errors = []

        for n in resolutions:
            grid = create_grid(n, n)
            x, y = get_cell_centers(grid)

            phi = test_func.f(x, y)
            exact_dx = test_func.df_dx(x, y)

            num_dx, _ = gradient_fd(phi, grid, order=4, periodic=(True, True))
            l2, _ = compute_errors(num_dx, exact_dx, exclude_boundary=False)
            errors.append(l2)

        # Compute convergence orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i] > 1e-14 and errors[i+1] > 1e-14:
                order = np.log(errors[i] / errors[i+1]) / np.log(2)
                orders.append(order)

        avg_order = np.mean(orders)
        assert avg_order > 3.5, f"{test_func.name}: Expected O(h⁴), got {avg_order:.2f}"


# =============================================================================
# Pytest Tests: Symmetry
# =============================================================================

class TestGradientSymmetry:
    """Test gradient symmetry properties."""

    def test_symmetric_function_symmetric_gradient(self):
        """For f(x,y) = x² + y², |∂f/∂x| and |∂f/∂y| should have same statistics."""
        grid = create_grid(32, 32)
        x, y = get_cell_centers(grid)

        phi = QUADRATIC.f(x, y)
        num_dx, num_dy = gradient_fd(phi, grid, order=2)

        # For symmetric function centered at (0.5, 0.5), gradients should be symmetric
        # Compare RMS values
        rms_x = np.sqrt(np.mean(np.array(num_dx)**2))
        rms_y = np.sqrt(np.mean(np.array(num_dy)**2))

        rel_diff = abs(rms_x - rms_y) / max(rms_x, rms_y)
        assert rel_diff < 0.01, f"Gradient not symmetric: RMS_x={rms_x:.4f}, RMS_y={rms_y:.4f}"


# =============================================================================
# Pytest Tests: TriGrid (Unstructured Triangular)
# =============================================================================

from fesomx import TriGrid
from fesomx.operators.unstructured.gradient_fv import gradient_fv


def create_tri_grid(n_per_side: int, domain=(0., 1., 0., 1.)):
    """Create structured triangular grid for testing."""
    from fesomx.utilities.mesh_generation import create_structured_triangular_mesh

    grid = TriGrid("test_tri", periodic=(False, False))
    mesh_data = create_structured_triangular_mesh(n_per_side, n_per_side, domain=domain)
    grid.vertices = mesh_data['vertices']
    grid.cells = [cell for cell in mesh_data['triangles']]  # Convert to list of arrays
    grid.compute_neighbors(halo_width=1)
    return grid


def get_tri_cell_centers(grid):
    """Get cell centers for TriGrid."""
    centers = grid.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]
    return x, y


class TestTriGridGradient:
    """Test gradient operator on triangular grids using finite volume methods."""

    def test_linear_gradient(self):
        """Linear function gradient should be accurate on TriGrid."""
        grid = create_tri_grid(16)
        x, y = get_tri_cell_centers(grid)
        cell_areas = grid.get_cell_volumes()  # 2D: areas

        phi = LINEAR.f(x, y)
        exact_dx = LINEAR.df_dx(x, y)
        exact_dy = LINEAR.df_dy(x, y)

        # Use least-squares gradient (order=2)
        num_dx, num_dy = gradient_fv(phi, grid, order=2)

        # Use area-weighted L2 error
        l2_x = compute_weighted_l2(num_dx, exact_dx, cell_areas)
        l2_y = compute_weighted_l2(num_dy, exact_dy, cell_areas)

        # Also compute unweighted for comparison
        l2_x_unweighted = np.sqrt(np.mean((np.array(num_dx) - exact_dx)**2))

        # Should be fairly accurate for linear functions (FV methods have boundary effects)
        assert l2_x < 0.2, f"TriGrid Linear ∂f/∂x error too large: weighted L2={l2_x:.2e}"
        assert l2_y < 0.2, f"TriGrid Linear ∂f/∂y error too large: weighted L2={l2_y:.2e}"

    def test_quadratic_gradient(self):
        """Quadratic function gradient on TriGrid."""
        grid = create_tri_grid(16)
        x, y = get_tri_cell_centers(grid)
        cell_areas = grid.get_cell_volumes()

        phi = QUADRATIC.f(x, y)
        exact_dx = QUADRATIC.df_dx(x, y)
        exact_dy = QUADRATIC.df_dy(x, y)

        num_dx, num_dy = gradient_fv(phi, grid, order=2)

        # Use area-weighted L2 error
        l2_x = compute_weighted_l2(num_dx, exact_dx, cell_areas)
        l2_y = compute_weighted_l2(num_dy, exact_dy, cell_areas)

        # FV methods are less accurate than FD on structured grids
        assert l2_x < 0.5, f"TriGrid Quadratic ∂f/∂x error too large: weighted L2={l2_x:.2e}"
        assert l2_y < 0.5, f"TriGrid Quadratic ∂f/∂y error too large: weighted L2={l2_y:.2e}"

    @pytest.mark.parametrize("test_func", [CUBIC, SINUSOIDAL])
    def test_convergence(self, test_func):
        """Test that TriGrid gradient converges with mesh refinement (area-weighted)."""
        resolutions = [8, 16, 32]
        errors = []

        for n in resolutions:
            grid = create_tri_grid(n)
            x, y = get_tri_cell_centers(grid)
            cell_areas = grid.get_cell_volumes()

            phi = test_func.f(x, y)
            exact_dx = test_func.df_dx(x, y)

            num_dx, _ = gradient_fv(phi, grid, order=2)
            # Use area-weighted L2 error
            l2 = compute_weighted_l2(num_dx, exact_dx, cell_areas)
            errors.append(l2)

        # Compute convergence orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i] > 1e-14 and errors[i+1] > 1e-14:
                order = np.log(errors[i] / errors[i+1]) / np.log(2)
                orders.append(order)

        avg_order = np.mean(orders)
        # FV least-squares should achieve at least ~1st order convergence
        assert avg_order > 0.5, f"TriGrid {test_func.name}: Expected convergence, got {avg_order:.2f}"


# =============================================================================
# Pytest Tests: MixedGrid (Quads + Triangles)
# =============================================================================

from fesomx import MixedGrid


def create_mixed_grid(n_per_side: int, domain=(0., 1., 0., 1.)):
    """Create mixed quad+triangle grid for testing."""
    from fesomx.utilities.mesh_generation import create_mixed_mesh

    grid = MixedGrid("test_mixed", periodic=(False, False))
    mesh_data = create_mixed_mesh(n_per_side, n_per_side, domain=domain, seed=42)
    grid.vertices = mesh_data['vertices']
    grid.cells = mesh_data['cells']  # Already a list of arrays
    grid.cell_types = np.array(mesh_data['cell_types'])  # Required for get_cell_volumes
    grid.compute_neighbors(halo_width=1)
    return grid


class TestMixedGridGradient:
    """Test gradient operator on mixed grids (quads + triangles)."""

    def test_linear_gradient(self):
        """Linear function gradient on MixedGrid."""
        grid = create_mixed_grid(8)
        x, y = get_tri_cell_centers(grid)  # Same structure as TriGrid
        cell_areas = grid.get_cell_volumes()

        phi = LINEAR.f(x, y)
        exact_dx = LINEAR.df_dx(x, y)
        exact_dy = LINEAR.df_dy(x, y)

        num_dx, num_dy = gradient_fv(phi, grid, order=2)

        # Use area-weighted L2 error
        l2_x = compute_weighted_l2(num_dx, exact_dx, cell_areas)
        l2_y = compute_weighted_l2(num_dy, exact_dy, cell_areas)

        # MixedGrid has higher errors due to irregular cell topology and boundaries
        assert l2_x < 0.35, f"MixedGrid Linear ∂f/∂x error too large: weighted L2={l2_x:.2e}"
        assert l2_y < 0.35, f"MixedGrid Linear ∂f/∂y error too large: weighted L2={l2_y:.2e}"

    def test_sinusoidal_gradient(self):
        """Sinusoidal function gradient on MixedGrid."""
        grid = create_mixed_grid(16)
        x, y = get_tri_cell_centers(grid)
        cell_areas = grid.get_cell_volumes()

        phi = SINUSOIDAL.f(x, y)
        exact_dx = SINUSOIDAL.df_dx(x, y)
        exact_dy = SINUSOIDAL.df_dy(x, y)

        num_dx, num_dy = gradient_fv(phi, grid, order=2)

        # Use area-weighted L2 error
        l2_x = compute_weighted_l2(num_dx, exact_dx, cell_areas)
        l2_y = compute_weighted_l2(num_dy, exact_dy, cell_areas)

        # Allow larger error for complex function on mixed grid
        assert l2_x < 2.0, f"MixedGrid Sinusoidal ∂f/∂x error too large: weighted L2={l2_x:.2e}"
        assert l2_y < 2.0, f"MixedGrid Sinusoidal ∂f/∂y error too large: weighted L2={l2_y:.2e}"

    def test_convergence(self):
        """Test that MixedGrid gradient converges with mesh refinement (area-weighted)."""
        resolutions = [4, 8, 16]
        errors = []

        for n in resolutions:
            grid = create_mixed_grid(n)
            x, y = get_tri_cell_centers(grid)
            cell_areas = grid.get_cell_volumes()

            phi = CUBIC.f(x, y)
            exact_dx = CUBIC.df_dx(x, y)

            num_dx, _ = gradient_fv(phi, grid, order=2)
            # Use area-weighted L2 error
            l2 = compute_weighted_l2(num_dx, exact_dx, cell_areas)
            errors.append(l2)

        # Compute convergence orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i] > 1e-14 and errors[i+1] > 1e-14:
                order = np.log(errors[i] / errors[i+1]) / np.log(2)
                orders.append(order)

        avg_order = np.mean(orders) if orders else 0
        # Mixed grids should show some convergence
        assert avg_order > 0.3, f"MixedGrid Cubic: Expected convergence, got {avg_order:.2f}"


# =============================================================================
# Visualization
# =============================================================================

def create_gradient_test_visualization(output_dir="output/gradient_operator"):
    """Create visualization of gradient operator tests."""
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("GRADIENT OPERATOR ANALYTICAL VERIFICATION")
    print("=" * 70)

    # =========================================================================
    # Figure 1: All Test Functions and Their Gradients
    # =========================================================================
    print("\n1. Creating test functions overview...")

    grid = create_grid(32, 32)
    x, y = get_cell_centers(grid)

    fig, axes = plt.subplots(len(ALL_TEST_FUNCTIONS), 4, figsize=(16, 4*len(ALL_TEST_FUNCTIONS)))

    for i, tf in enumerate(ALL_TEST_FUNCTIONS):
        phi = tf.f(x, y)
        exact_dx = tf.df_dx(x, y)
        exact_dy = tf.df_dy(x, y)
        num_dx, num_dy = gradient_fd(phi, grid, order=2)

        l2_x, _ = compute_errors(num_dx, exact_dx)
        l2_y, _ = compute_errors(num_dy, exact_dy)

        # Column 0: f(x,y)
        im0 = axes[i, 0].imshow(phi, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        axes[i, 0].set_title(f'{tf.name}\n{tf.description}', fontsize=10)
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[i, 0], shrink=0.8)

        # Column 1: Analytical ∂f/∂x
        im1 = axes[i, 1].imshow(exact_dx, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i, 1].set_title(f'Analytical ∂f/∂x', fontsize=10)
        plt.colorbar(im1, ax=axes[i, 1], shrink=0.8)

        # Column 2: Numerical ∂f/∂x
        im2 = axes[i, 2].imshow(np.array(num_dx), origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i, 2].set_title(f'Numerical ∂f/∂x', fontsize=10)
        plt.colorbar(im2, ax=axes[i, 2], shrink=0.8)

        # Column 3: Error
        error = np.abs(np.array(num_dx) - exact_dx)
        im3 = axes[i, 3].imshow(error, origin='lower', extent=[0, 1, 0, 1], cmap='hot')
        exact_str = "EXACT" if tf.exact_for_2nd_order else f"L2={l2_x:.2e}"
        axes[i, 3].set_title(f'|Error| ({exact_str})', fontsize=10)
        plt.colorbar(im3, ax=axes[i, 3], shrink=0.8)

        print(f"  {tf.name}: L2(∂x)={l2_x:.2e}, L2(∂y)={l2_y:.2e}")

    for ax in axes[-1, :]:
        ax.set_xlabel('X')

    plt.suptitle('Gradient Operator: Test Functions and Analytical Verification\n'
                 '2nd-order Centered Differences on 32×32 Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_test_functions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/01_test_functions.png")

    # =========================================================================
    # Figure 2: Convergence Study
    # =========================================================================
    print("\n2. Running convergence study...")

    resolutions = [8, 16, 32, 64, 128]
    test_funcs_for_convergence = [CUBIC, SINUSOIDAL, GAUSSIAN, POLYNOMIAL]

    results_2nd = {tf.name: [] for tf in test_funcs_for_convergence}
    results_4th = {tf.name: [] for tf in test_funcs_for_convergence}

    for n in resolutions:
        grid = create_grid(n, n)
        x, y = get_cell_centers(grid)

        for tf in test_funcs_for_convergence:
            phi = tf.f(x, y)
            exact_dx = tf.df_dx(x, y)

            # 2nd order
            num_dx_2, _ = gradient_fd(phi, grid, order=2)
            l2_2, _ = compute_errors(num_dx_2, exact_dx)
            results_2nd[tf.name].append(l2_2)

            # 4th order (periodic for clean interior)
            num_dx_4, _ = gradient_fd(phi, grid, order=4, periodic=(True, True))
            l2_4, _ = compute_errors(num_dx_4, exact_dx, exclude_boundary=False)
            results_4th[tf.name].append(l2_4)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h_values = [1.0/n for n in resolutions]
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_funcs_for_convergence)))

    # 2nd order convergence
    ax = axes[0]
    for i, tf in enumerate(test_funcs_for_convergence):
        ax.loglog(h_values, results_2nd[tf.name], 'o-', color=colors[i],
                 label=tf.name, linewidth=2, markersize=8)

    # Reference lines
    h = np.array(h_values)
    ax.loglog(h, 2*h**2, 'k--', alpha=0.5, label='O(h²)')
    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('2nd-Order Gradient Convergence', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    # 4th order convergence
    ax = axes[1]
    for i, tf in enumerate(test_funcs_for_convergence):
        ax.loglog(h_values, results_4th[tf.name], 's-', color=colors[i],
                 label=tf.name, linewidth=2, markersize=8)

    ax.loglog(h, 0.5*h**4, 'k:', alpha=0.5, label='O(h⁴)')
    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('4th-Order Gradient Convergence (Periodic)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    plt.suptitle('Gradient Operator Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_convergence.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/02_convergence.png")

    # Print convergence rates
    print("\n  Convergence Orders:")
    for tf in test_funcs_for_convergence:
        orders_2 = []
        orders_4 = []
        for i in range(len(resolutions) - 1):
            if results_2nd[tf.name][i] > 1e-14 and results_2nd[tf.name][i+1] > 1e-14:
                orders_2.append(np.log(results_2nd[tf.name][i] / results_2nd[tf.name][i+1]) / np.log(2))
            if results_4th[tf.name][i] > 1e-14 and results_4th[tf.name][i+1] > 1e-14:
                orders_4.append(np.log(results_4th[tf.name][i] / results_4th[tf.name][i+1]) / np.log(2))

        avg_2 = np.mean(orders_2) if orders_2 else 0
        avg_4 = np.mean(orders_4) if orders_4 else 0
        print(f"    {tf.name}: 2nd order = {avg_2:.2f}, 4th order = {avg_4:.2f}")

    # =========================================================================
    # Figure 3: Exactness Verification for Linear and Quadratic
    # =========================================================================
    print("\n3. Verifying exactness for linear and quadratic functions...")

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, tf in enumerate([LINEAR, QUADRATIC]):
        grid = create_grid(16, 16)
        x, y = get_cell_centers(grid)

        phi = tf.f(x, y)
        exact_dx = tf.df_dx(x, y)
        exact_dy = tf.df_dy(x, y)
        num_dx, num_dy = gradient_fd(phi, grid, order=2)

        error_x = np.array(num_dx) - exact_dx
        error_y = np.array(num_dy) - exact_dy
        max_error_x = np.max(np.abs(error_x[1:-1, 1:-1]))
        max_error_y = np.max(np.abs(error_y[1:-1, 1:-1]))

        # Column 0: f(x,y)
        im0 = axes[i, 0].imshow(phi, origin='lower', extent=[0, 1, 0, 1], cmap='viridis')
        axes[i, 0].set_title(f'{tf.name}: {tf.description}', fontsize=11)
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[i, 0])

        # Column 1: Analytical ∂f/∂x
        im1 = axes[i, 1].imshow(exact_dx, origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i, 1].set_title('Analytical ∂f/∂x', fontsize=11)
        plt.colorbar(im1, ax=axes[i, 1])

        # Column 2: Numerical ∂f/∂x
        im2 = axes[i, 2].imshow(np.array(num_dx), origin='lower', extent=[0, 1, 0, 1], cmap='RdBu_r')
        axes[i, 2].set_title('Numerical ∂f/∂x', fontsize=11)
        plt.colorbar(im2, ax=axes[i, 2])

        # Column 3: Error (should be ~0)
        im3 = axes[i, 3].imshow(np.abs(error_x), origin='lower', extent=[0, 1, 0, 1],
                                cmap='hot', vmin=0, vmax=max(1e-14, max_error_x*2))
        axes[i, 3].set_title(f'|Error| (max={max_error_x:.2e})', fontsize=11)
        plt.colorbar(im3, ax=axes[i, 3])

        status = "EXACT" if max_error_x < 1e-10 else "NOT EXACT"
        print(f"  {tf.name}: max|error| = {max_error_x:.2e} → {status}")

    for ax in axes[-1, :]:
        ax.set_xlabel('X')

    plt.suptitle('Gradient Exactness Verification\n'
                 'For Linear/Quadratic functions, 2nd-order FD should be EXACT',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_exactness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/03_exactness.png")

    # =========================================================================
    # Figure 4: Mathematical Derivation Visual
    # =========================================================================
    print("\n4. Creating stencil visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 2nd order stencil
    ax = axes[0]
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)

    # Draw grid points
    for xi in [-2, -1, 0, 1, 2]:
        ax.plot(xi, 0, 'ko', markersize=15)
        ax.text(xi, -0.3, f'i{xi:+d}' if xi != 0 else 'i', ha='center', fontsize=10)

    # Highlight stencil points
    ax.plot(-1, 0, 'ro', markersize=20, alpha=0.5)
    ax.plot(1, 0, 'bo', markersize=20, alpha=0.5)
    ax.annotate('', xy=(1, 0.5), xytext=(-1, 0.5),
               arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.text(0, 0.7, '2h', ha='center', fontsize=12, color='green')

    ax.text(0, 1.2, r'$\frac{\partial \phi}{\partial x} \approx \frac{\phi_{i+1} - \phi_{i-1}}{2h}$',
           ha='center', fontsize=14)
    ax.text(0, -1.0, 'Error: O(h²)', ha='center', fontsize=11, style='italic')
    ax.set_title('2nd-Order Centered Difference', fontsize=12, fontweight='bold')
    ax.axis('off')

    # 4th order stencil
    ax = axes[1]
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.5, 1.5)

    for xi in [-2, -1, 0, 1, 2]:
        ax.plot(xi, 0, 'ko', markersize=15)
        ax.text(xi, -0.3, f'i{xi:+d}' if xi != 0 else 'i', ha='center', fontsize=10)

    # Highlight stencil points with weights
    weights = [1, -8, 0, 8, -1]
    colors_w = ['orange', 'red', 'gray', 'blue', 'cyan']
    for xi, w, c in zip([-2, -1, 0, 1, 2], weights, colors_w):
        if w != 0:
            ax.plot(xi, 0, 'o', markersize=20, alpha=0.5, color=c)
            ax.text(xi, 0.4, f'{w:+d}', ha='center', fontsize=10, color=c)

    ax.text(0, 1.2, r'$\frac{\partial \phi}{\partial x} \approx \frac{-\phi_{i+2} + 8\phi_{i+1} - 8\phi_{i-1} + \phi_{i-2}}{12h}$',
           ha='center', fontsize=12)
    ax.text(0, -1.0, 'Error: O(h⁴)', ha='center', fontsize=11, style='italic')
    ax.set_title('4th-Order Centered Difference', fontsize=12, fontweight='bold')
    ax.axis('off')

    # Error scaling
    ax = axes[2]
    h_vals = np.logspace(-2, 0, 50)
    ax.loglog(h_vals, h_vals**2, 'b-', linewidth=2, label='O(h²)')
    ax.loglog(h_vals, h_vals**4, 'r-', linewidth=2, label='O(h⁴)')
    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Scaling', fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    plt.suptitle('Finite Difference Stencils for Gradient Operator',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_stencils.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/04_stencils.png")

    print(f"\nAll plots saved to: {output_dir}/")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gradient operator verification")
    parser.add_argument("--test", "-t", action="store_true", help="Run pytest tests")
    parser.add_argument("--visualize", "-v", action="store_true", help="Create visualizations")
    args = parser.parse_args()

    if args.test:
        pytest.main([__file__, "-v"])
    elif args.visualize:
        create_gradient_test_visualization()
    else:
        # Default: run both
        create_gradient_test_visualization()
        print("\nRunning pytest tests...")
        pytest.main([__file__, "-v", "--tb=short"])
