"""
Test: Gradient Verification Across Grid Types

Verifies gradient computation on QuadGrid, TriGrid, and MixedGrid using
the same domain and analytical test function.

Analytical function:
    f(x, y) = sin(πx) * cos(πy)

Analytical gradient:
    ∂f/∂x = π * cos(πx) * cos(πy)
    ∂f/∂y = -π * sin(πx) * sin(πy)

Test verifies:
1. Gradient computation correctness against analytical solution
2. Convergence with grid refinement
3. Visual comparison of numerical vs analytical gradients
"""

import numpy as np
import pytest

from fesomx import QuadGrid, TriGrid, MixedGrid, StaggerType
from fesomx.utilities import (
    create_structured_triangular_mesh,
    create_mixed_mesh,
    plot_grid,
    plot_field,
    save_plot,
)
from fesomx.operators.structured.gradient_fd import gradient_fd
from fesomx.operators.unstructured.gradient_fv import gradient_fv


# =============================================================================
# Analytical Test Functions
# =============================================================================

def analytical_scalar(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Test scalar field: f(x, y) = sin(πx) * cos(πy)

    Properties:
    - Smooth, differentiable everywhere
    - Varies in both x and y
    - Non-trivial gradient
    """
    return np.sin(np.pi * x) * np.cos(np.pi * y)


def analytical_gradient_x(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂f/∂x = π * cos(πx) * cos(πy)"""
    return np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)


def analytical_gradient_y(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂f/∂y = -π * sin(πx) * sin(πy)"""
    return -np.pi * np.sin(np.pi * x) * np.sin(np.pi * y)


# =============================================================================
# Periodic Analytical Test Functions
# =============================================================================

def analytical_scalar_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Periodic test scalar field: f(x, y) = sin(2πx) * sin(2πy)

    Properties:
    - Truly periodic on [0,1] × [0,1]: f(0,y)=f(1,y), f(x,0)=f(x,1)
    - Derivatives also periodic
    - Symmetric in x and y → similar error behavior for ∂f/∂x and ∂f/∂y
    """
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def analytical_gradient_x_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂f/∂x = 2π * cos(2πx) * sin(2πy)"""
    return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def analytical_gradient_y_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂f/∂y = 2π * sin(2πx) * cos(2πy)"""
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


# =============================================================================
# Grid Creation Helpers
# =============================================================================

def create_quad_grid(nx: int, ny: int, domain: tuple, periodic: bool = False) -> QuadGrid:
    """Create a structured quadrilateral grid.

    Args:
        nx, ny: Grid dimensions
        domain: (x_min, x_max, y_min, y_max)
        periodic: If True, create doubly-periodic grid
    """
    grid = QuadGrid(
        name=f"quad_{nx}x{ny}{'_periodic' if periodic else ''}",
        nx=nx,
        ny=ny,
        stagger_type=StaggerType.A,
        periodic=(periodic, periodic)
    )
    grid.create_mesh(domain=domain)
    grid.compute_neighbors(halo_width=2)  # Use halo_width=2 for 4th order
    return grid


def create_tri_grid(nx: int, ny: int, domain: tuple) -> TriGrid:
    """Create an unstructured triangular grid (structured layout)."""
    mesh_data = create_structured_triangular_mesh(nx=nx, ny=ny, domain=domain)

    grid = TriGrid(name=f"tri_{nx}x{ny}", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        triangles=mesh_data['triangles']
    )
    grid.compute_neighbors(halo_width=1)
    return grid


def create_mixed_grid(nx: int, ny: int, domain: tuple, tri_fraction: float = 0.3) -> MixedGrid:
    """Create a mixed grid with quads and triangles."""
    mesh_data = create_mixed_mesh(
        nx=nx, ny=ny, domain=domain,
        tri_fraction=tri_fraction,
        seed=42  # For reproducibility
    )

    grid = MixedGrid(name=f"mixed_{nx}x{ny}", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        cells=mesh_data['cells']
    )
    grid.compute_neighbors(halo_width=1)
    return grid


# =============================================================================
# Gradient Computation Helpers
# =============================================================================

def compute_gradient_quad(grid: QuadGrid, scalar_field: np.ndarray, order: int = 2):
    """Compute gradient on structured quad grid using finite differences.

    Args:
        grid: QuadGrid object
        scalar_field: Scalar field to compute gradient of
        order: Accuracy order (2 or 4)
               2 → 2nd-order centered differences (halo_width=1)
               4 → 4th-order centered differences (halo_width=2)

    Returns:
        grad_x, grad_y: Gradient components
    """
    import jax.numpy as jnp
    scalar_jax = jnp.array(scalar_field)
    grad_x, grad_y = gradient_fd(scalar_jax, grid, perform_halo_exchange=False, order=order)
    return np.array(grad_x), np.array(grad_y)


def compute_gradient_unstructured(grid, scalar_field: np.ndarray, order: int = 1):
    """Compute gradient on unstructured grid using finite volumes.

    Args:
        grid: Unstructured grid (TriGrid or MixedGrid)
        scalar_field: Scalar field to compute gradient of
        order: Accuracy order (1, 2, or 4)
               1 → Green-Gauss method (~1st order on general meshes)
               2 → Least-Squares with direct neighbors (2nd order)
               4 → Extended Least-Squares with neighbor-of-neighbors (higher accuracy)

    Returns:
        grad_x, grad_y: Gradient components
    """
    import jax.numpy as jnp
    scalar_jax = jnp.array(scalar_field)
    grad_x, grad_y = gradient_fv(scalar_jax, grid, perform_halo_exchange=False, order=order)
    return np.array(grad_x), np.array(grad_y)


# =============================================================================
# Error Metrics
# =============================================================================

def compute_errors(numerical: np.ndarray, analytical: np.ndarray,
                   exclude_boundary: bool = True, grid_shape: tuple = None,
                   boundary_width: int = 1):
    """
    Compute error metrics between numerical and analytical solutions.

    Args:
        numerical: Numerical solution
        analytical: Analytical solution
        exclude_boundary: Whether to exclude boundary cells
        grid_shape: Shape of grid (ny, nx) for structured grids
        boundary_width: Width of boundary to exclude (1 for 2nd order, 2 for 4th order)

    Returns:
        dict with L1, L2, Linf errors and statistics
    """
    if exclude_boundary and grid_shape is not None:
        # For structured grids, exclude boundary cells
        ny, nx = grid_shape
        bw = boundary_width
        interior = numerical.reshape(ny, nx)[bw:-bw, bw:-bw].ravel()
        analytical_int = analytical.reshape(ny, nx)[bw:-bw, bw:-bw].ravel()
        error = interior - analytical_int
    else:
        error = numerical.ravel() - analytical.ravel()

    l1_error = np.mean(np.abs(error))
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(np.abs(error))

    return {
        'L1': l1_error,
        'L2': l2_error,
        'Linf': linf_error,
        'mean': np.mean(error),
        'std': np.std(error),
    }


def compute_weighted_l2(numerical: np.ndarray, analytical: np.ndarray,
                        cell_areas: np.ndarray) -> float:
    """Compute area-weighted L2 error for unstructured grids.

    This properly accounts for varying cell sizes in TriGrid/MixedGrid.

    Args:
        numerical: Numerical gradient values (1D array for unstructured)
        analytical: Analytical gradient values
        cell_areas: Area of each cell

    Returns:
        Area-weighted L2 error: sqrt(sum(error² * area) / total_area)
    """
    error = np.array(numerical).ravel() - np.array(analytical).ravel()
    cell_areas = np.array(cell_areas).ravel()
    total_area = np.sum(cell_areas)
    return np.sqrt(np.sum(error**2 * cell_areas) / total_area)


def compute_errors_detailed(numerical: np.ndarray, analytical: np.ndarray,
                            grid_shape: tuple = None, boundary_width: int = 1):
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

        # Reshape for structured grid analysis
        num_2d = numerical.reshape(ny, nx)
        ana_2d = analytical.reshape(ny, nx)

        # Check if we have enough cells for interior
        if ny > 2*bw and nx > 2*bw:
            # Interior region
            interior_num = num_2d[bw:-bw, bw:-bw].ravel()
            interior_ana = ana_2d[bw:-bw, bw:-bw].ravel()
            interior_error = interior_num - interior_ana
            interior_l2 = np.sqrt(np.mean(interior_error**2))
            interior_linf = np.max(np.abs(interior_error))

            # Boundary region (all cells within bw of edge)
            mask = np.ones((ny, nx), dtype=bool)
            mask[bw:-bw, bw:-bw] = False
            boundary_num = num_2d[mask]
            boundary_ana = ana_2d[mask]
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


def identify_boundary_cells_unstructured(grid, domain: tuple, method: str = 'vertex') -> np.ndarray:
    """Identify boundary cells for unstructured grids.

    Args:
        grid: Unstructured grid (TriGrid or MixedGrid)
        domain: (x_min, x_max, y_min, y_max)
        method: Identification method
            'vertex' - cell has at least one vertex on domain boundary (more accurate)
            'center' - cell center within tolerance of boundary (faster)

    Returns:
        Boolean mask where True = boundary cell
    """
    x_min, x_max, y_min, y_max = domain
    tol = 1e-10  # Tolerance for boundary detection
    n_cells = grid.n_cells

    if method == 'vertex':
        # Cell is on boundary if ANY vertex is on domain boundary
        boundary_mask = np.zeros(n_cells, dtype=bool)

        for cell_id, cell in enumerate(grid.cells):
            cell_verts = grid.vertices[cell]
            # Check if any vertex is on boundary
            on_left = np.any(np.abs(cell_verts[:, 0] - x_min) < tol)
            on_right = np.any(np.abs(cell_verts[:, 0] - x_max) < tol)
            on_bottom = np.any(np.abs(cell_verts[:, 1] - y_min) < tol)
            on_top = np.any(np.abs(cell_verts[:, 1] - y_max) < tol)

            boundary_mask[cell_id] = on_left or on_right or on_bottom or on_top

    elif method == 'center':
        # Cell is on boundary if center is close to domain edge
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        # Use cell-size-based tolerance
        cell_areas = grid.get_cell_volumes()
        avg_cell_size = np.sqrt(np.mean(cell_areas))
        center_tol = avg_cell_size * 0.6

        boundary_mask = (
            (x < x_min + center_tol) | (x > x_max - center_tol) |
            (y < y_min + center_tol) | (y > y_max - center_tol)
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'vertex' or 'center'.")

    return boundary_mask


def compute_errors_unstructured(numerical: np.ndarray, analytical: np.ndarray,
                                 boundary_mask: np.ndarray,
                                 cell_areas: np.ndarray = None) -> dict:
    """Compute interior and boundary errors for unstructured grids.

    Args:
        numerical: Numerical gradient values
        analytical: Analytical gradient values
        boundary_mask: Boolean mask (True = boundary cell)
        cell_areas: Optional cell areas for weighted L2

    Returns:
        dict with interior/boundary/global L2 and Linf errors
    """
    numerical = np.array(numerical).ravel()
    analytical = np.array(analytical).ravel()
    error = numerical - analytical

    interior_mask = ~boundary_mask
    interior_error = error[interior_mask]
    boundary_error = error[boundary_mask]

    # Unweighted L2
    interior_l2 = np.sqrt(np.mean(interior_error**2)) if len(interior_error) > 0 else 0.0
    boundary_l2 = np.sqrt(np.mean(boundary_error**2)) if len(boundary_error) > 0 else 0.0
    global_l2 = np.sqrt(np.mean(error**2))

    # Linf
    interior_linf = np.max(np.abs(interior_error)) if len(interior_error) > 0 else 0.0
    boundary_linf = np.max(np.abs(boundary_error)) if len(boundary_error) > 0 else 0.0

    result = {
        'interior_L2': interior_l2,
        'boundary_L2': boundary_l2,
        'global_L2': global_l2,
        'interior_Linf': interior_linf,
        'boundary_Linf': boundary_linf,
        'n_interior': np.sum(interior_mask),
        'n_boundary': np.sum(boundary_mask),
    }

    # Area-weighted L2 if cell areas provided
    if cell_areas is not None:
        cell_areas = np.array(cell_areas).ravel()
        interior_areas = cell_areas[interior_mask]
        boundary_areas = cell_areas[boundary_mask]

        if len(interior_areas) > 0 and np.sum(interior_areas) > 0:
            result['interior_L2_weighted'] = np.sqrt(
                np.sum(interior_error**2 * interior_areas) / np.sum(interior_areas)
            )
        else:
            result['interior_L2_weighted'] = 0.0

        if len(boundary_areas) > 0 and np.sum(boundary_areas) > 0:
            result['boundary_L2_weighted'] = np.sqrt(
                np.sum(boundary_error**2 * boundary_areas) / np.sum(boundary_areas)
            )
        else:
            result['boundary_L2_weighted'] = 0.0

    return result


# =============================================================================
# Test Functions
# =============================================================================

class TestGradientVerification:
    """Test gradient computation across all grid types."""

    # Common test parameters
    DOMAIN = (0.0, 1.0, 0.0, 1.0)
    NX, NY = 16, 16

    def test_quad_grid_gradient(self):
        """Test gradient on structured quadrilateral grid."""
        grid = create_quad_grid(self.NX, self.NY, self.DOMAIN)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(self.NY, self.NX)
        y = centers[:, 1].reshape(self.NY, self.NX)

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient
        grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors (excluding boundaries for FD)
        errors_x = compute_errors(grad_x_num, grad_x_ana,
                                  exclude_boundary=True,
                                  grid_shape=(self.NY, self.NX))
        errors_y = compute_errors(grad_y_num, grad_y_ana,
                                  exclude_boundary=True,
                                  grid_shape=(self.NY, self.NX))

        print(f"\nQuadGrid Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # Assert reasonable accuracy (2nd order FD should give ~h² error)
        h = 1.0 / self.NX
        expected_error = 0.5  # Relaxed for central differences
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_quad_grid_gradient_4th_order(self):
        """Test 4th-order gradient on structured quadrilateral grid."""
        # Need larger grid for 4th order (boundary_width=2)
        nx, ny = 32, 32
        grid = create_quad_grid(nx, ny, self.DOMAIN)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute 4th-order numerical gradient
        grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar, order=4)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors (excluding 2-cell boundary for 4th order)
        errors_x = compute_errors(grad_x_num, grad_x_ana,
                                  exclude_boundary=True,
                                  grid_shape=(ny, nx),
                                  boundary_width=2)
        errors_y = compute_errors(grad_y_num, grad_y_ana,
                                  exclude_boundary=True,
                                  grid_shape=(ny, nx),
                                  boundary_width=2)

        print(f"\nQuadGrid 4th-Order Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # 4th order should have much smaller errors than 2nd order
        expected_error = 0.01  # Should be ~h⁴ ≈ (1/32)⁴ ≈ 1e-6
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_tri_grid_gradient(self):
        """Test gradient on unstructured triangular grid."""
        grid = create_tri_grid(self.NX, self.NY, self.DOMAIN)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nTriGrid Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # FV methods on triangular grids have different error characteristics
        expected_error = 1.0  # More relaxed for FV on unstructured
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_mixed_grid_gradient(self):
        """Test gradient on mixed quad/triangle grid."""
        grid = create_mixed_grid(self.NX, self.NY, self.DOMAIN, tri_fraction=0.3)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nMixedGrid Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        expected_error = 1.0
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_tri_grid_gradient_least_squares(self):
        """Test 2nd-order Least-Squares gradient on triangular grid."""
        grid = create_tri_grid(self.NX, self.NY, self.DOMAIN)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient using Least-Squares (order=2)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=2)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nTriGrid Least-Squares (order=2) Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # Least-squares should give better accuracy than Green-Gauss
        expected_error = 0.5  # Tighter bound for LS
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_tri_grid_gradient_extended_ls(self):
        """Test extended Least-Squares gradient (order=4) on triangular grid."""
        # Use larger grid for extended stencil
        nx, ny = 24, 24
        grid = create_tri_grid(nx, ny, self.DOMAIN)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient using Extended LS (order=4)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=4)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nTriGrid Extended LS (order=4) Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # Extended LS should have good accuracy
        expected_error = 0.3
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_mixed_grid_gradient_least_squares(self):
        """Test 2nd-order Least-Squares gradient on mixed grid."""
        grid = create_mixed_grid(self.NX, self.NY, self.DOMAIN, tri_fraction=0.3)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient using Least-Squares (order=2)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=2)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nMixedGrid Least-Squares (order=2) Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        expected_error = 0.5
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"

    def test_mixed_grid_gradient_extended_ls(self):
        """Test extended Least-Squares gradient (order=4) on mixed grid."""
        # Use larger grid for extended stencil
        nx, ny = 24, 24
        grid = create_mixed_grid(nx, ny, self.DOMAIN, tri_fraction=0.3)

        # Get cell centers
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]

        # Compute scalar field
        scalar = analytical_scalar(x, y)

        # Compute numerical gradient using Extended LS (order=4)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=4)

        # Compute analytical gradient
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        # Compute errors
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)

        print(f"\nMixedGrid Extended LS (order=4) Gradient Errors:")
        print(f"  ∂f/∂x - L2: {errors_x['L2']:.6e}, Linf: {errors_x['Linf']:.6e}")
        print(f"  ∂f/∂y - L2: {errors_y['L2']:.6e}, Linf: {errors_y['Linf']:.6e}")

        # Extended LS should have good accuracy
        expected_error = 0.3
        assert errors_x['L2'] < expected_error, f"X-gradient L2 error too large: {errors_x['L2']}"
        assert errors_y['L2'] < expected_error, f"Y-gradient L2 error too large: {errors_y['L2']}"


# =============================================================================
# Convergence Study
# =============================================================================

def run_convergence_study():
    """Run grid refinement study to verify convergence order."""
    domain = (0.0, 1.0, 0.0, 1.0)
    resolutions = [8, 16, 32, 64]

    results = {
        'quad_2nd': {'h': [], 'L2_x': [], 'L2_y': []},
        'quad_4th': {'h': [], 'L2_x': [], 'L2_y': []},
        'tri_gg': {'h': [], 'L2_x': [], 'L2_y': []},      # Green-Gauss (order=1)
        'tri_ls': {'h': [], 'L2_x': [], 'L2_y': []},      # Least-Squares (order=2)
        'tri_els': {'h': [], 'L2_x': [], 'L2_y': []},     # Extended LS (order=4)
        'mixed_gg': {'h': [], 'L2_x': [], 'L2_y': []},    # Green-Gauss (order=1)
        'mixed_ls': {'h': [], 'L2_x': [], 'L2_y': []},    # Least-Squares (order=2)
    }

    print("\n" + "="*70)
    print("CONVERGENCE STUDY (2nd vs 4th Order Comparison)")
    print("="*70)

    for nx in resolutions:
        ny = nx
        h = 1.0 / nx
        print(f"\nResolution: {nx}x{ny} (h = {h:.4f})")
        print("-" * 50)

        # QuadGrid - 2nd order
        grid = create_quad_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)
        scalar = analytical_scalar(x, y)
        grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar, order=2)
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)
        errors_x = compute_errors(grad_x_num, grad_x_ana, True, (ny, nx), boundary_width=1)
        errors_y = compute_errors(grad_y_num, grad_y_ana, True, (ny, nx), boundary_width=1)
        results['quad_2nd']['h'].append(h)
        results['quad_2nd']['L2_x'].append(errors_x['L2'])
        results['quad_2nd']['L2_y'].append(errors_y['L2'])
        print(f"  QuadGrid (2nd): L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # QuadGrid - 4th order (need at least 5x5 grid)
        if nx >= 8:
            grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar, order=4)
            errors_x = compute_errors(grad_x_num, grad_x_ana, True, (ny, nx), boundary_width=2)
            errors_y = compute_errors(grad_y_num, grad_y_ana, True, (ny, nx), boundary_width=2)
            results['quad_4th']['h'].append(h)
            results['quad_4th']['L2_x'].append(errors_x['L2'])
            results['quad_4th']['L2_y'].append(errors_y['L2'])
            print(f"  QuadGrid (4th): L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # TriGrid - Green-Gauss (order=1)
        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]
        scalar = analytical_scalar(x, y)
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=1)
        errors_x = compute_errors(grad_x_num, grad_x_ana, False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, False)
        results['tri_gg']['h'].append(h)
        results['tri_gg']['L2_x'].append(errors_x['L2'])
        results['tri_gg']['L2_y'].append(errors_y['L2'])
        print(f"  TriGrid GG:  L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # TriGrid - Least-Squares (order=2)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=2)
        errors_x = compute_errors(grad_x_num, grad_x_ana, False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, False)
        results['tri_ls']['h'].append(h)
        results['tri_ls']['L2_x'].append(errors_x['L2'])
        results['tri_ls']['L2_y'].append(errors_y['L2'])
        print(f"  TriGrid LS:  L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # TriGrid - Extended Least-Squares (order=4)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=4)
        errors_x = compute_errors(grad_x_num, grad_x_ana, False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, False)
        results['tri_els']['h'].append(h)
        results['tri_els']['L2_x'].append(errors_x['L2'])
        results['tri_els']['L2_y'].append(errors_y['L2'])
        print(f"  TriGrid ELS: L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # MixedGrid - Green-Gauss (order=1)
        grid = create_mixed_grid(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        x = centers[:, 0]
        y = centers[:, 1]
        scalar = analytical_scalar(x, y)
        grad_x_ana = analytical_gradient_x(x, y)
        grad_y_ana = analytical_gradient_y(x, y)

        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=1)
        errors_x = compute_errors(grad_x_num, grad_x_ana, False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, False)
        results['mixed_gg']['h'].append(h)
        results['mixed_gg']['L2_x'].append(errors_x['L2'])
        results['mixed_gg']['L2_y'].append(errors_y['L2'])
        print(f"  MixedGrid GG: L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # MixedGrid - Least-Squares (order=2)
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid, scalar, order=2)
        errors_x = compute_errors(grad_x_num, grad_x_ana, False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, False)
        results['mixed_ls']['h'].append(h)
        results['mixed_ls']['L2_x'].append(errors_x['L2'])
        results['mixed_ls']['L2_y'].append(errors_y['L2'])
        print(f"  MixedGrid LS: L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

    # Compute convergence rates
    print("\n" + "="*70)
    print("CONVERGENCE RATES (order p where error ~ h^p)")
    print("="*70)

    for grid_type, data in results.items():
        h = np.array(data['h'])
        L2_x = np.array(data['L2_x'])
        L2_y = np.array(data['L2_y'])

        # Compute order using successive refinements
        if len(h) >= 2:
            order_x = np.log(L2_x[:-1] / L2_x[1:]) / np.log(h[:-1] / h[1:])
            order_y = np.log(L2_y[:-1] / L2_y[1:]) / np.log(h[:-1] / h[1:])
            print(f"\n{grid_type}:")
            print(f"  ∂f/∂x orders: {order_x}")
            print(f"  ∂f/∂y orders: {order_y}")
            print(f"  Average order: {np.mean([*order_x, *order_y]):.2f}")

    return results


def run_periodic_convergence_study():
    """Run convergence study with periodic boundaries (QuadGrid only).

    Uses f(x,y) = sin(2πx)sin(2πy) which is truly periodic on [0,1]×[0,1].
    This eliminates boundary treatment errors and shows cleaner convergence.
    """
    domain = (0.0, 1.0, 0.0, 1.0)
    resolutions = [8, 16, 32, 64]

    results = {
        'quad_2nd_periodic': {'h': [], 'L2_x': [], 'L2_y': []},
        'quad_4th_periodic': {'h': [], 'L2_x': [], 'L2_y': []},
    }

    print("\n" + "="*70)
    print("PERIODIC CONVERGENCE STUDY (f = sin(2πx)sin(2πy))")
    print("="*70)

    for nx in resolutions:
        ny = nx
        h = 1.0 / nx
        print(f"\nResolution: {nx}x{ny} (h = {h:.4f})")
        print("-" * 50)

        # Periodic QuadGrid - 2nd order
        grid = create_quad_grid(nx, ny, domain, periodic=True)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)
        scalar = analytical_scalar_periodic(x, y)
        grad_x_ana = analytical_gradient_x_periodic(x, y)
        grad_y_ana = analytical_gradient_y_periodic(x, y)

        grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar, order=2)
        # For periodic, we can include all cells (no boundary exclusion)
        errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
        errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)
        results['quad_2nd_periodic']['h'].append(h)
        results['quad_2nd_periodic']['L2_x'].append(errors_x['L2'])
        results['quad_2nd_periodic']['L2_y'].append(errors_y['L2'])
        print(f"  QuadGrid Periodic (2nd): L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

        # Periodic QuadGrid - 4th order
        if nx >= 8:
            grad_x_num, grad_y_num = compute_gradient_quad(grid, scalar, order=4)
            errors_x = compute_errors(grad_x_num, grad_x_ana, exclude_boundary=False)
            errors_y = compute_errors(grad_y_num, grad_y_ana, exclude_boundary=False)
            results['quad_4th_periodic']['h'].append(h)
            results['quad_4th_periodic']['L2_x'].append(errors_x['L2'])
            results['quad_4th_periodic']['L2_y'].append(errors_y['L2'])
            print(f"  QuadGrid Periodic (4th): L2(∂x)={errors_x['L2']:.4e}, L2(∂y)={errors_y['L2']:.4e}")

    # Compute convergence rates
    print("\n" + "="*70)
    print("PERIODIC CONVERGENCE RATES")
    print("="*70)

    for grid_type, data in results.items():
        h = np.array(data['h'])
        L2_x = np.array(data['L2_x'])
        L2_y = np.array(data['L2_y'])

        if len(h) >= 2:
            order_x = np.log(L2_x[:-1] / L2_x[1:]) / np.log(h[:-1] / h[1:])
            order_y = np.log(L2_y[:-1] / L2_y[1:]) / np.log(h[:-1] / h[1:])
            print(f"\n{grid_type}:")
            print(f"  ∂f/∂x orders: {order_x}")
            print(f"  ∂f/∂y orders: {order_y}")
            print(f"  Average order: {np.mean([*order_x, *order_y]):.2f}")

    return results


def run_timing_study():
    """Measure execution time for different gradient methods across resolutions."""
    import time

    domain = (0.0, 1.0, 0.0, 1.0)
    resolutions = [8, 16, 32, 64]  # Reduced for faster execution
    n_repeats = 3  # Average over multiple runs

    timing_results = {
        'quad_2nd': {'n_cells': [], 'time': []},
        'quad_4th': {'n_cells': [], 'time': []},
        'tri_gg': {'n_cells': [], 'time': []},
        'tri_ls': {'n_cells': [], 'time': []},
        'tri_els': {'n_cells': [], 'time': []},
        'mixed_gg': {'n_cells': [], 'time': []},
        'mixed_ls': {'n_cells': [], 'time': []},
    }

    print("\n" + "="*70)
    print("TIMING STUDY (averaged over {} runs)".format(n_repeats))
    print("="*70)

    for nx in resolutions:
        ny = nx
        print(f"\nResolution: {nx}x{ny}")
        print("-" * 50)

        # QuadGrid timing
        grid = create_quad_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)
        scalar = analytical_scalar(x, y)
        n_cells = nx * ny

        # Warm-up run
        _ = compute_gradient_quad(grid, scalar, order=2)

        # 2nd order timing
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = compute_gradient_quad(grid, scalar, order=2)
            times.append(time.perf_counter() - t0)
        avg_time = np.mean(times)
        timing_results['quad_2nd']['n_cells'].append(n_cells)
        timing_results['quad_2nd']['time'].append(avg_time)
        print(f"  QuadGrid FD (2nd): {avg_time*1000:.3f} ms")

        # 4th order timing
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            _ = compute_gradient_quad(grid, scalar, order=4)
            times.append(time.perf_counter() - t0)
        avg_time = np.mean(times)
        timing_results['quad_4th']['n_cells'].append(n_cells)
        timing_results['quad_4th']['time'].append(avg_time)
        print(f"  QuadGrid FD (4th): {avg_time*1000:.3f} ms")

        # TriGrid timing
        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        scalar = analytical_scalar(centers[:, 0], centers[:, 1])
        n_cells = len(centers)

        # Warm-up
        _ = compute_gradient_unstructured(grid, scalar, order=1)

        for order, key in [(1, 'tri_gg'), (2, 'tri_ls'), (4, 'tri_els')]:
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                _ = compute_gradient_unstructured(grid, scalar, order=order)
                times.append(time.perf_counter() - t0)
            avg_time = np.mean(times)
            timing_results[key]['n_cells'].append(n_cells)
            timing_results[key]['time'].append(avg_time)
            method_name = {'tri_gg': 'GG', 'tri_ls': 'LS', 'tri_els': 'ELS'}[key]
            print(f"  TriGrid FV ({method_name}): {avg_time*1000:.3f} ms")

        # MixedGrid timing
        grid = create_mixed_grid(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        scalar = analytical_scalar(centers[:, 0], centers[:, 1])
        n_cells = len(centers)

        # Warm-up
        _ = compute_gradient_unstructured(grid, scalar, order=1)

        for order, key in [(1, 'mixed_gg'), (2, 'mixed_ls')]:
            times = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                _ = compute_gradient_unstructured(grid, scalar, order=order)
                times.append(time.perf_counter() - t0)
            avg_time = np.mean(times)
            timing_results[key]['n_cells'].append(n_cells)
            timing_results[key]['time'].append(avg_time)
            method_name = {'mixed_gg': 'GG', 'mixed_ls': 'LS'}[key]
            print(f"  MixedGrid FV ({method_name}): {avg_time*1000:.3f} ms")

    return timing_results


# =============================================================================
# Visualization
# =============================================================================

def create_visualizations(output_dir: str = "output/gradient_verification"):
    """Create comprehensive visualizations of gradient verification."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    domain = (0.0, 1.0, 0.0, 1.0)
    nx, ny = 12, 12

    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # =========================================================================
    # Figure 1: Grid Structure Comparison
    # =========================================================================
    print("\n1. Creating grid structure comparison...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # QuadGrid
    grid_quad = create_quad_grid(nx, ny, domain)
    centers_quad = grid_quad.get_cell_centers()
    ax = axes[0]
    plot_grid(grid_quad.vertices, grid_quad.cells, ax=ax,
              show_vertices=True, show_cells=True,
              title=f"QuadGrid ({nx}×{ny})\n{grid_quad.n_cells} cells, {grid_quad.n_vertices} vertices")
    # Mark cell centers
    ax.scatter(centers_quad[:, 0], centers_quad[:, 1],
               c='blue', s=20, marker='s', label='Cell centers', zorder=5)
    ax.legend(loc='upper right', fontsize=8)

    # TriGrid
    grid_tri = create_tri_grid(nx, ny, domain)
    centers_tri = grid_tri.get_cell_centers()
    ax = axes[1]
    plot_grid(grid_tri.vertices, grid_tri.cells, ax=ax,
              show_vertices=True, show_cells=True,
              title=f"TriGrid ({nx}×{ny} base)\n{grid_tri.n_cells} cells, {grid_tri.n_vertices} vertices")
    ax.scatter(centers_tri[:, 0], centers_tri[:, 1],
               c='blue', s=15, marker='s', label='Cell centers', zorder=5)
    ax.legend(loc='upper right', fontsize=8)

    # MixedGrid
    grid_mixed = create_mixed_grid(nx, ny, domain, tri_fraction=0.3)
    centers_mixed = grid_mixed.get_cell_centers()
    ax = axes[2]
    plot_grid(grid_mixed.vertices, grid_mixed.cells, ax=ax,
              show_vertices=True, show_cells=True,
              title=f"MixedGrid ({nx}×{ny}, 30% tri)\n{grid_mixed.n_cells} cells, {grid_mixed.n_vertices} vertices")
    ax.scatter(centers_mixed[:, 0], centers_mixed[:, 1],
               c='blue', s=15, marker='s', label='Cell centers', zorder=5)
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    save_plot(f"{output_dir}/01_grid_structures.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 2: Scalar Field on All Grids
    # =========================================================================
    print("2. Creating scalar field visualization...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # QuadGrid scalar
    x_q = centers_quad[:, 0].reshape(ny, nx)
    y_q = centers_quad[:, 1].reshape(ny, nx)
    scalar_quad = analytical_scalar(x_q, y_q)
    plot_field(grid_quad.vertices, grid_quad.cells, scalar_quad.ravel(),
               ax=axes[0], title="QuadGrid: f(x,y) = sin(πx)cos(πy)",
               cmap='RdBu_r', vmin=-1, vmax=1)

    # TriGrid scalar
    scalar_tri = analytical_scalar(centers_tri[:, 0], centers_tri[:, 1])
    plot_field(grid_tri.vertices, grid_tri.cells, scalar_tri,
               ax=axes[1], title="TriGrid: f(x,y) = sin(πx)cos(πy)",
               cmap='RdBu_r', vmin=-1, vmax=1)

    # MixedGrid scalar
    scalar_mixed = analytical_scalar(centers_mixed[:, 0], centers_mixed[:, 1])
    plot_field(grid_mixed.vertices, grid_mixed.cells, scalar_mixed,
               ax=axes[2], title="MixedGrid: f(x,y) = sin(πx)cos(πy)",
               cmap='RdBu_r', vmin=-1, vmax=1)

    plt.tight_layout()
    save_plot(f"{output_dir}/02_scalar_field.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 3: Gradient Components (Analytical vs Numerical) - QuadGrid
    # =========================================================================
    print("3. Creating QuadGrid gradient comparison...")

    grad_x_num_q, grad_y_num_q = compute_gradient_quad(grid_quad, scalar_quad)
    grad_x_ana_q = analytical_gradient_x(x_q, y_q)
    grad_y_ana_q = analytical_gradient_y(x_q, y_q)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: X-gradient
    vmax_x = max(abs(grad_x_ana_q).max(), abs(grad_x_num_q).max())
    plot_field(grid_quad.vertices, grid_quad.cells, grad_x_ana_q.ravel(),
               ax=axes[0, 0], title="∂f/∂x (Analytical)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    plot_field(grid_quad.vertices, grid_quad.cells, grad_x_num_q.ravel(),
               ax=axes[0, 1], title="∂f/∂x (Numerical FD)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    error_x = grad_x_num_q - grad_x_ana_q
    plot_field(grid_quad.vertices, grid_quad.cells, error_x.ravel(),
               ax=axes[0, 2], title=f"Error ∂f/∂x (L2={np.sqrt(np.mean(error_x**2)):.2e})",
               cmap='RdBu_r')

    # Row 2: Y-gradient
    vmax_y = max(abs(grad_y_ana_q).max(), abs(grad_y_num_q).max())
    plot_field(grid_quad.vertices, grid_quad.cells, grad_y_ana_q.ravel(),
               ax=axes[1, 0], title="∂f/∂y (Analytical)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    plot_field(grid_quad.vertices, grid_quad.cells, grad_y_num_q.ravel(),
               ax=axes[1, 1], title="∂f/∂y (Numerical FD)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    error_y = grad_y_num_q - grad_y_ana_q
    plot_field(grid_quad.vertices, grid_quad.cells, error_y.ravel(),
               ax=axes[1, 2], title=f"Error ∂f/∂y (L2={np.sqrt(np.mean(error_y**2)):.2e})",
               cmap='RdBu_r')

    plt.suptitle("QuadGrid Gradient Verification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/03_quad_gradient.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 4: Gradient Components - TriGrid
    # =========================================================================
    print("4. Creating TriGrid gradient comparison...")

    grad_x_num_t, grad_y_num_t = compute_gradient_unstructured(grid_tri, scalar_tri)
    grad_x_ana_t = analytical_gradient_x(centers_tri[:, 0], centers_tri[:, 1])
    grad_y_ana_t = analytical_gradient_y(centers_tri[:, 0], centers_tri[:, 1])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: X-gradient
    vmax_x = max(abs(grad_x_ana_t).max(), abs(grad_x_num_t).max())
    plot_field(grid_tri.vertices, grid_tri.cells, grad_x_ana_t,
               ax=axes[0, 0], title="∂f/∂x (Analytical)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    plot_field(grid_tri.vertices, grid_tri.cells, grad_x_num_t,
               ax=axes[0, 1], title="∂f/∂x (Numerical FV)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    error_x = grad_x_num_t - grad_x_ana_t
    plot_field(grid_tri.vertices, grid_tri.cells, error_x,
               ax=axes[0, 2], title=f"Error ∂f/∂x (L2={np.sqrt(np.mean(error_x**2)):.2e})",
               cmap='RdBu_r')

    # Row 2: Y-gradient
    vmax_y = max(abs(grad_y_ana_t).max(), abs(grad_y_num_t).max())
    plot_field(grid_tri.vertices, grid_tri.cells, grad_y_ana_t,
               ax=axes[1, 0], title="∂f/∂y (Analytical)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    plot_field(grid_tri.vertices, grid_tri.cells, grad_y_num_t,
               ax=axes[1, 1], title="∂f/∂y (Numerical FV)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    error_y = grad_y_num_t - grad_y_ana_t
    plot_field(grid_tri.vertices, grid_tri.cells, error_y,
               ax=axes[1, 2], title=f"Error ∂f/∂y (L2={np.sqrt(np.mean(error_y**2)):.2e})",
               cmap='RdBu_r')

    plt.suptitle("TriGrid Gradient Verification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/04_tri_gradient.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 5: Gradient Components - MixedGrid
    # =========================================================================
    print("5. Creating MixedGrid gradient comparison...")

    grad_x_num_m, grad_y_num_m = compute_gradient_unstructured(grid_mixed, scalar_mixed)
    grad_x_ana_m = analytical_gradient_x(centers_mixed[:, 0], centers_mixed[:, 1])
    grad_y_ana_m = analytical_gradient_y(centers_mixed[:, 0], centers_mixed[:, 1])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: X-gradient
    vmax_x = max(abs(grad_x_ana_m).max(), abs(grad_x_num_m).max())
    plot_field(grid_mixed.vertices, grid_mixed.cells, grad_x_ana_m,
               ax=axes[0, 0], title="∂f/∂x (Analytical)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    plot_field(grid_mixed.vertices, grid_mixed.cells, grad_x_num_m,
               ax=axes[0, 1], title="∂f/∂x (Numerical FV)",
               cmap='RdBu_r', vmin=-vmax_x, vmax=vmax_x)
    error_x = grad_x_num_m - grad_x_ana_m
    plot_field(grid_mixed.vertices, grid_mixed.cells, error_x,
               ax=axes[0, 2], title=f"Error ∂f/∂x (L2={np.sqrt(np.mean(error_x**2)):.2e})",
               cmap='RdBu_r')

    # Row 2: Y-gradient
    vmax_y = max(abs(grad_y_ana_m).max(), abs(grad_y_num_m).max())
    plot_field(grid_mixed.vertices, grid_mixed.cells, grad_y_ana_m,
               ax=axes[1, 0], title="∂f/∂y (Analytical)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    plot_field(grid_mixed.vertices, grid_mixed.cells, grad_y_num_m,
               ax=axes[1, 1], title="∂f/∂y (Numerical FV)",
               cmap='RdBu_r', vmin=-vmax_y, vmax=vmax_y)
    error_y = grad_y_num_m - grad_y_ana_m
    plot_field(grid_mixed.vertices, grid_mixed.cells, error_y,
               ax=axes[1, 2], title=f"Error ∂f/∂y (L2={np.sqrt(np.mean(error_y**2)):.2e})",
               cmap='RdBu_r')

    plt.suptitle("MixedGrid Gradient Verification", fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/05_mixed_gradient.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 6: Detailed Grid with Nodes, Cell Centers, and Partitions
    # =========================================================================
    print("6. Creating detailed grid visualization with nodes, cell centers, and partitions...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Smaller grids for clearer visualization
    nx_small, ny_small = 6, 6
    n_partitions = 4  # Number of partitions to visualize

    # Partition colors
    partition_colors = plt.cm.Set1(np.linspace(0, 1, n_partitions))

    # -------------------------------------------------------------------------
    # Row 1: Grid structure with nodes and cell centers
    # -------------------------------------------------------------------------

    # QuadGrid detailed
    grid_q = create_quad_grid(nx_small, ny_small, domain)
    centers_q = grid_q.get_cell_centers()
    ax = axes[0, 0]

    # Plot cells with edges
    for cell in grid_q.cells:
        cell_verts = grid_q.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=1)
        ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1], alpha=0.1, color='blue')

    # Plot vertices (nodes)
    ax.scatter(grid_q.vertices[:, 0], grid_q.vertices[:, 1],
               c='red', s=60, marker='o', zorder=5, label='Vertices (nodes)')

    # Plot cell centers
    ax.scatter(centers_q[:, 0], centers_q[:, 1],
               c='blue', s=40, marker='s', zorder=5, label='Cell centers (scalar)')

    ax.set_title(f"QuadGrid Detail\n{grid_q.n_vertices} vertices, {grid_q.n_cells} cells")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # TriGrid detailed
    grid_t = create_tri_grid(nx_small, ny_small, domain)
    centers_t = grid_t.get_cell_centers()
    ax = axes[0, 1]

    for cell in grid_t.cells:
        cell_verts = grid_t.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=1)
        ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1], alpha=0.1, color='green')

    ax.scatter(grid_t.vertices[:, 0], grid_t.vertices[:, 1],
               c='red', s=60, marker='o', zorder=5, label='Vertices (nodes)')
    ax.scatter(centers_t[:, 0], centers_t[:, 1],
               c='blue', s=40, marker='s', zorder=5, label='Cell centers (scalar)')

    ax.set_title(f"TriGrid Detail\n{grid_t.n_vertices} vertices, {grid_t.n_cells} cells")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # MixedGrid detailed
    grid_m = create_mixed_grid(nx_small, ny_small, domain, tri_fraction=0.4)
    centers_m = grid_m.get_cell_centers()
    ax = axes[0, 2]

    tri_cells = grid_m.get_triangular_cells()
    quad_cells = grid_m.get_quadrilateral_cells()

    for i, cell in enumerate(grid_m.cells):
        cell_verts = grid_m.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=1)
        if i in tri_cells:
            ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1], alpha=0.15, color='orange')
        else:
            ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1], alpha=0.15, color='purple')

    ax.scatter(grid_m.vertices[:, 0], grid_m.vertices[:, 1],
               c='red', s=60, marker='o', zorder=5, label='Vertices (nodes)')
    ax.scatter(centers_m[:, 0], centers_m[:, 1],
               c='blue', s=40, marker='s', zorder=5, label='Cell centers (scalar)')

    ax.set_title(f"MixedGrid Detail\n{grid_m.n_vertices} vertices, {grid_m.n_cells} cells\n(orange=tri, purple=quad)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # -------------------------------------------------------------------------
    # Row 2: Partition visualization (cells colored by partition ID)
    # -------------------------------------------------------------------------

    # QuadGrid partitions (structured 2x2 partitioning)
    ax = axes[1, 0]
    partition_info_q = grid_q.create_partition_info(mesh_shape=(2, 2))

    for cell_id, cell in enumerate(grid_q.cells):
        cell_verts = grid_q.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])

        # Determine partition for this cell
        i, j = grid_q.get_logical_indices(cell_id)
        part_i = 0 if i < nx_small // 2 else 1
        part_j = 0 if j < ny_small // 2 else 1
        partition_id = part_j * 2 + part_i

        ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1],
               alpha=0.6, color=partition_colors[partition_id])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=0.5)

    # Add partition labels at centers
    for p in range(n_partitions):
        part_j, part_i = divmod(p, 2)
        x_center = (0.25 + part_i * 0.5) * (domain[1] - domain[0]) + domain[0]
        y_center = (0.25 + part_j * 0.5) * (domain[3] - domain[2]) + domain[2]
        ax.text(x_center, y_center, f'P{p}', fontsize=14, fontweight='bold',
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_title(f"QuadGrid Partitions\n{n_partitions} partitions (2×2 mesh)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # TriGrid partitions (geometric partitioning based on cell center position)
    ax = axes[1, 1]

    for cell_id, cell in enumerate(grid_t.cells):
        cell_verts = grid_t.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])

        # Partition by cell center position (2x2 grid)
        center = centers_t[cell_id]
        part_i = 0 if center[0] < 0.5 else 1
        part_j = 0 if center[1] < 0.5 else 1
        partition_id = part_j * 2 + part_i

        ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1],
               alpha=0.6, color=partition_colors[partition_id])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=0.5)

    # Add partition labels
    for p in range(n_partitions):
        part_j, part_i = divmod(p, 2)
        x_center = 0.25 + part_i * 0.5
        y_center = 0.25 + part_j * 0.5
        ax.text(x_center, y_center, f'P{p}', fontsize=14, fontweight='bold',
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_title(f"TriGrid Partitions\n{n_partitions} partitions (geometric)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # MixedGrid partitions
    ax = axes[1, 2]

    for cell_id, cell in enumerate(grid_m.cells):
        cell_verts = grid_m.vertices[cell]
        cell_verts = np.vstack([cell_verts, cell_verts[0]])

        # Partition by cell center position (2x2 grid)
        center = centers_m[cell_id]
        part_i = 0 if center[0] < 0.5 else 1
        part_j = 0 if center[1] < 0.5 else 1
        partition_id = part_j * 2 + part_i

        ax.fill(cell_verts[:-1, 0], cell_verts[:-1, 1],
               alpha=0.6, color=partition_colors[partition_id])
        ax.plot(cell_verts[:, 0], cell_verts[:, 1], 'k-', linewidth=0.5)

    # Add partition labels
    for p in range(n_partitions):
        part_j, part_i = divmod(p, 2)
        x_center = 0.25 + part_i * 0.5
        y_center = 0.25 + part_j * 0.5
        ax.text(x_center, y_center, f'P{p}', fontsize=14, fontweight='bold',
               ha='center', va='center', color='white',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.set_title(f"MixedGrid Partitions\n{n_partitions} partitions (geometric)")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Add legend for partitions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=partition_colors[i], alpha=0.6,
                            label=f'Partition {i}') for i in range(n_partitions)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
              bbox_to_anchor=(0.5, -0.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    save_plot(f"{output_dir}/06_grid_detail.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 7: Convergence Plot (2nd vs 4th Order Comparison)
    # =========================================================================
    print("7. Creating convergence plot (2nd vs 4th order comparison)...")

    results = run_convergence_study()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Colors and markers for different schemes
    # Labels include: method (order) [halo/stencil]
    # For structured FD: halo width = points needed on each side
    # For unstructured FV: stencil = neighbor ring depth
    colors = {
        'quad_2nd': 'blue',
        'quad_4th': 'darkblue',
        'tri_gg': 'lightgreen',
        'tri_ls': 'green',
        'tri_els': 'darkgreen',
        'mixed_gg': 'lightsalmon',
        'mixed_ls': 'orangered',
    }
    markers = {
        'quad_2nd': 'o',
        'quad_4th': 's',
        'tri_gg': '^',
        'tri_ls': '^',
        'tri_els': '^',
        'mixed_gg': 'v',
        'mixed_ls': 'v',
    }
    # Labels: Method (convergence order) [halo_width or stencil_ring]
    labels = {
        'quad_2nd': 'QuadGrid FD (O=2) [h=1]',
        'quad_4th': 'QuadGrid FD (O=4) [h=2]',
        'tri_gg': 'TriGrid GG (O≈1) [r=1]',
        'tri_ls': 'TriGrid LS (O≈2) [r=1]',
        'tri_els': 'TriGrid ELS (O≈2) [r=2]',
        'mixed_gg': 'MixedGrid GG (O≈1) [r=1]',
        'mixed_ls': 'MixedGrid LS (O≈2) [r=1]',
    }
    linestyles = {
        'quad_2nd': '-',
        'quad_4th': '--',
        'tri_gg': ':',
        'tri_ls': '-',
        'tri_els': '--',
        'mixed_gg': ':',
        'mixed_ls': '-',
    }

    for grid_type, data in results.items():
        if len(data['h']) == 0:
            continue
        h = np.array(data['h'])
        L2_x = np.array(data['L2_x'])
        L2_y = np.array(data['L2_y'])

        axes[0].loglog(h, L2_x, f'{markers[grid_type]}{linestyles[grid_type]}',
                       color=colors[grid_type], label=labels[grid_type],
                       linewidth=2, markersize=8)
        axes[1].loglog(h, L2_y, f'{markers[grid_type]}{linestyles[grid_type]}',
                       color=colors[grid_type], label=labels[grid_type],
                       linewidth=2, markersize=8)

    # Set same axis limits for both plots first
    all_h = []
    all_errors = []
    for data in results.values():
        if len(data['h']) > 0:
            all_h.extend(data['h'])
            all_errors.extend(data['L2_x'])
            all_errors.extend(data['L2_y'])

    if all_h and all_errors:
        h_min, h_max = min(all_h) * 0.8, max(all_h) * 1.2
        err_min, err_max = min(e for e in all_errors if e > 0) * 0.3, max(all_errors) * 2

        for ax in axes:
            ax.set_xlim(h_min, h_max)
            ax.set_ylim(err_min, err_max)

    # Add reference slopes - scaled to be visible within the data range
    h_ref = np.array([h_min, h_max])
    # Scale O(h²) line to pass through middle of 2nd-order data
    h_mid = np.sqrt(h_min * h_max)
    err_2nd_mid = results['quad_2nd']['L2_x'][len(results['quad_2nd']['L2_x'])//2] if results['quad_2nd']['L2_x'] else 1e-2
    c2 = err_2nd_mid / h_mid**2
    axes[0].loglog(h_ref, c2 * h_ref**2, 'k--', alpha=0.5, label='O(h²)', linewidth=1.5)
    axes[1].loglog(h_ref, c2 * h_ref**2, 'k--', alpha=0.5, label='O(h²)', linewidth=1.5)

    # Scale O(h⁴) line to pass through middle of 4th-order data
    err_4th_mid = results['quad_4th']['L2_x'][len(results['quad_4th']['L2_x'])//2] if results['quad_4th']['L2_x'] else 1e-5
    c4 = err_4th_mid / h_mid**4
    axes[0].loglog(h_ref, c4 * h_ref**4, 'k-.', alpha=0.5, label='O(h⁴)', linewidth=1.5)
    axes[1].loglog(h_ref, c4 * h_ref**4, 'k-.', alpha=0.5, label='O(h⁴)', linewidth=1.5)

    axes[0].set_xlabel('Grid spacing h')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('∂f/∂x = π cos(πx)cos(πy)\n(non-zero at boundaries)')
    axes[0].legend(fontsize=7, loc='lower right')
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].set_xlabel('Grid spacing h')
    axes[1].set_ylabel('L2 Error')
    axes[1].set_title('∂f/∂y = -π sin(πx)sin(πy)\n(zero at boundaries)')
    axes[1].legend(fontsize=7, loc='lower right')
    axes[1].grid(True, which='both', alpha=0.3)

    # Add note explaining the difference
    fig.text(0.5, 0.01,
             'Note: ∂f/∂y has lower errors because it vanishes at boundaries (sin(0)=sin(π)=0), '
             'reducing boundary treatment errors.\n'
             'Legend: O=convergence order, h=halo width (FD), r=stencil ring (FV)',
             ha='center', fontsize=8, style='italic')

    plt.suptitle('Gradient Convergence Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_plot(f"{output_dir}/07_convergence.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 8: Periodic Boundary Convergence Comparison
    # =========================================================================
    print("8. Creating periodic boundary convergence plot...")

    periodic_results = run_periodic_convergence_study()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot periodic results
    periodic_colors = {
        'quad_2nd_periodic': 'royalblue',
        'quad_4th_periodic': 'navy',
    }
    periodic_markers = {
        'quad_2nd_periodic': 'o',
        'quad_4th_periodic': 's',
    }
    periodic_labels = {
        'quad_2nd_periodic': 'Periodic FD (O=2) [h=1]',
        'quad_4th_periodic': 'Periodic FD (O=4) [h=2]',
    }

    for grid_type, data in periodic_results.items():
        if len(data['h']) == 0:
            continue
        h = np.array(data['h'])
        L2_x = np.array(data['L2_x'])
        L2_y = np.array(data['L2_y'])

        axes[0].loglog(h, L2_x, f'{periodic_markers[grid_type]}-',
                       color=periodic_colors[grid_type], label=periodic_labels[grid_type],
                       linewidth=2, markersize=8)
        axes[1].loglog(h, L2_y, f'{periodic_markers[grid_type]}-',
                       color=periodic_colors[grid_type], label=periodic_labels[grid_type],
                       linewidth=2, markersize=8)

    # Add reference slopes
    all_h = []
    all_errors = []
    for data in periodic_results.values():
        if len(data['h']) > 0:
            all_h.extend(data['h'])
            all_errors.extend(data['L2_x'])
            all_errors.extend(data['L2_y'])

    if all_h and all_errors:
        h_min, h_max = min(all_h) * 0.8, max(all_h) * 1.2
        err_min, err_max = min(e for e in all_errors if e > 0) * 0.3, max(all_errors) * 2

        for ax in axes:
            ax.set_xlim(h_min, h_max)
            ax.set_ylim(err_min, err_max)

        h_ref = np.array([h_min, h_max])
        h_mid = np.sqrt(h_min * h_max)

        # Reference slopes
        if periodic_results['quad_2nd_periodic']['L2_x']:
            err_2nd = periodic_results['quad_2nd_periodic']['L2_x'][len(periodic_results['quad_2nd_periodic']['L2_x'])//2]
            c2 = err_2nd / h_mid**2
            axes[0].loglog(h_ref, c2 * h_ref**2, 'k--', alpha=0.5, label='O(h²)', linewidth=1.5)
            axes[1].loglog(h_ref, c2 * h_ref**2, 'k--', alpha=0.5, label='O(h²)', linewidth=1.5)

        if periodic_results['quad_4th_periodic']['L2_x']:
            err_4th = periodic_results['quad_4th_periodic']['L2_x'][len(periodic_results['quad_4th_periodic']['L2_x'])//2]
            c4 = err_4th / h_mid**4
            axes[0].loglog(h_ref, c4 * h_ref**4, 'k-.', alpha=0.5, label='O(h⁴)', linewidth=1.5)
            axes[1].loglog(h_ref, c4 * h_ref**4, 'k-.', alpha=0.5, label='O(h⁴)', linewidth=1.5)

    axes[0].set_xlabel('Grid spacing h')
    axes[0].set_ylabel('L2 Error')
    axes[0].set_title('∂f/∂x = 2π cos(2πx)sin(2πy)')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which='both', alpha=0.3)

    axes[1].set_xlabel('Grid spacing h')
    axes[1].set_ylabel('L2 Error')
    axes[1].set_title('∂f/∂y = 2π sin(2πx)cos(2πy)')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, which='both', alpha=0.3)

    fig.text(0.5, 0.01,
             'Periodic BCs: f(x,y)=sin(2πx)sin(2πy). No boundary errors → ∂f/∂x and ∂f/∂y have similar errors.\n'
             'Legend: O=convergence order, h=halo width',
             ha='center', fontsize=8, style='italic')

    plt.suptitle('Periodic Boundary Convergence (QuadGrid)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    save_plot(f"{output_dir}/08_periodic_convergence.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 9: Execution Time Scaling
    # =========================================================================
    print("9. Creating execution time scaling plot...")

    timing_results = run_timing_study()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Reuse colors from convergence plot
    timing_labels = {
        'quad_2nd': 'QuadGrid FD (O=2) [h=1]',
        'quad_4th': 'QuadGrid FD (O=4) [h=2]',
        'tri_gg': 'TriGrid GG (O≈1) [r=1]',
        'tri_ls': 'TriGrid LS (O≈2) [r=1]',
        'tri_els': 'TriGrid ELS (O≈2) [r=2]',
        'mixed_gg': 'MixedGrid GG (O≈1) [r=1]',
        'mixed_ls': 'MixedGrid LS (O≈2) [r=1]',
    }

    # Left plot: Time vs Number of cells
    for grid_type, data in timing_results.items():
        if len(data['n_cells']) == 0:
            continue
        n_cells = np.array(data['n_cells'])
        times = np.array(data['time']) * 1000  # Convert to ms

        axes[0].loglog(n_cells, times, f'{markers[grid_type]}{linestyles[grid_type]}',
                       color=colors[grid_type], label=timing_labels[grid_type],
                       linewidth=2, markersize=8)

    # Add O(N) reference line
    all_n = []
    all_t = []
    for data in timing_results.values():
        if len(data['n_cells']) > 0:
            all_n.extend(data['n_cells'])
            all_t.extend(np.array(data['time']) * 1000)

    if all_n and all_t:
        n_ref = np.array([min(all_n), max(all_n)])
        # Scale to middle of quad data
        n_mid = np.sqrt(min(all_n) * max(all_n))
        t_mid = timing_results['quad_2nd']['time'][len(timing_results['quad_2nd']['time'])//2] * 1000
        c_n = t_mid / n_mid
        axes[0].loglog(n_ref, c_n * n_ref, 'k--', alpha=0.5, label='O(N)', linewidth=1.5)

    axes[0].set_xlabel('Number of cells')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Execution Time vs Problem Size')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[0].grid(True, which='both', alpha=0.3)

    # Right plot: Time per cell (efficiency)
    for grid_type, data in timing_results.items():
        if len(data['n_cells']) == 0:
            continue
        n_cells = np.array(data['n_cells'])
        times = np.array(data['time']) * 1e6  # Convert to microseconds
        time_per_cell = times / n_cells

        axes[1].semilogx(n_cells, time_per_cell, f'{markers[grid_type]}{linestyles[grid_type]}',
                         color=colors[grid_type], label=timing_labels[grid_type],
                         linewidth=2, markersize=8)

    axes[1].set_xlabel('Number of cells')
    axes[1].set_ylabel('Time per cell (μs)')
    axes[1].set_title('Efficiency: Time per Cell')
    axes[1].legend(fontsize=7, loc='upper right')
    axes[1].grid(True, which='both', alpha=0.3)

    plt.suptitle('Gradient Computation Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/09_timing.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 10: Interior vs Boundary Error Comparison (QuadGrid)
    # =========================================================================
    print("10. Creating interior vs boundary error comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Use larger grid for clearer visualization
    nx_large, ny_large = 32, 32
    grid_q32 = create_quad_grid(nx_large, ny_large, domain)
    centers_q32 = grid_q32.get_cell_centers()
    x_q32 = centers_q32[:, 0].reshape(ny_large, nx_large)
    y_q32 = centers_q32[:, 1].reshape(ny_large, nx_large)
    scalar_q32 = analytical_scalar(x_q32, y_q32)

    # Compute gradients
    grad_x_num_2nd, grad_y_num_2nd = compute_gradient_quad(grid_q32, scalar_q32, order=2)
    grad_x_num_4th, grad_y_num_4th = compute_gradient_quad(grid_q32, scalar_q32, order=4)
    grad_x_ana = analytical_gradient_x(x_q32, y_q32)
    grad_y_ana = analytical_gradient_y(x_q32, y_q32)

    # Compute error fields
    error_x_2nd = grad_x_num_2nd - grad_x_ana
    error_y_2nd = grad_y_num_2nd - grad_y_ana
    error_x_4th = grad_x_num_4th - grad_x_ana
    error_y_4th = grad_y_num_4th - grad_y_ana

    # Get detailed error metrics
    detailed_x_2nd = compute_errors_detailed(grad_x_num_2nd, grad_x_ana, (ny_large, nx_large), boundary_width=1)
    detailed_y_2nd = compute_errors_detailed(grad_y_num_2nd, grad_y_ana, (ny_large, nx_large), boundary_width=1)
    detailed_x_4th = compute_errors_detailed(grad_x_num_4th, grad_x_ana, (ny_large, nx_large), boundary_width=2)
    detailed_y_4th = compute_errors_detailed(grad_y_num_4th, grad_y_ana, (ny_large, nx_large), boundary_width=2)

    # Row 1: 2nd order errors
    # Error field with boundary highlighted
    ax = axes[0, 0]
    im = ax.imshow(np.abs(error_x_2nd), origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    plt.colorbar(im, ax=ax, label='|Error|')
    # Draw boundary region
    bw = 1
    rect = plt.Rectangle((bw/(nx_large), bw/(ny_large)),
                          1 - 2*bw/(nx_large), 1 - 2*bw/(ny_large),
                          fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_title(f'2nd Order |Error ∂f/∂x|\nInterior L2={detailed_x_2nd["interior_L2"]:.2e}\n'
                 f'Boundary L2={detailed_x_2nd["boundary_L2"]:.2e}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax = axes[0, 1]
    im = ax.imshow(np.abs(error_y_2nd), origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    plt.colorbar(im, ax=ax, label='|Error|')
    rect = plt.Rectangle((bw/(nx_large), bw/(ny_large)),
                          1 - 2*bw/(nx_large), 1 - 2*bw/(ny_large),
                          fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_title(f'2nd Order |Error ∂f/∂y|\nInterior L2={detailed_y_2nd["interior_L2"]:.2e}\n'
                 f'Boundary L2={detailed_y_2nd["boundary_L2"]:.2e}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Bar chart comparing interior vs boundary
    ax = axes[0, 2]
    labels = ['∂f/∂x', '∂f/∂y']
    interior_errors = [detailed_x_2nd['interior_L2'], detailed_y_2nd['interior_L2']]
    boundary_errors = [detailed_x_2nd['boundary_L2'], detailed_y_2nd['boundary_L2']]

    x_pos = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, interior_errors, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errors, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('2nd Order: Interior vs Boundary\n(bw=1)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

    # Row 2: 4th order errors
    bw = 2
    ax = axes[1, 0]
    im = ax.imshow(np.abs(error_x_4th), origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    plt.colorbar(im, ax=ax, label='|Error|')
    rect = plt.Rectangle((bw/(nx_large), bw/(ny_large)),
                          1 - 2*bw/(nx_large), 1 - 2*bw/(ny_large),
                          fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_title(f'4th Order |Error ∂f/∂x|\nInterior L2={detailed_x_4th["interior_L2"]:.2e}\n'
                 f'Boundary L2={detailed_x_4th["boundary_L2"]:.2e}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    ax = axes[1, 1]
    im = ax.imshow(np.abs(error_y_4th), origin='lower', extent=[0, 1, 0, 1], cmap='hot')
    plt.colorbar(im, ax=ax, label='|Error|')
    rect = plt.Rectangle((bw/(nx_large), bw/(ny_large)),
                          1 - 2*bw/(nx_large), 1 - 2*bw/(ny_large),
                          fill=False, edgecolor='cyan', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.set_title(f'4th Order |Error ∂f/∂y|\nInterior L2={detailed_y_4th["interior_L2"]:.2e}\n'
                 f'Boundary L2={detailed_y_4th["boundary_L2"]:.2e}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Bar chart comparing interior vs boundary
    ax = axes[1, 2]
    interior_errors = [detailed_x_4th['interior_L2'], detailed_y_4th['interior_L2']]
    boundary_errors = [detailed_x_4th['boundary_L2'], detailed_y_4th['boundary_L2']]

    bars1 = ax.bar(x_pos - width/2, interior_errors, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errors, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('4th Order: Interior vs Boundary\n(bw=2)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale('log')
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=7)

    plt.suptitle(f'Interior vs Boundary Error Analysis (QuadGrid {nx_large}×{ny_large})\n'
                 'Cyan dashed box = interior region (excludes boundary cells)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/10_interior_boundary_error.png", dpi=150)
    plt.close()

    # =========================================================================
    # Figure 11: Interior vs Boundary Error for Unstructured Grids
    # =========================================================================
    print("11. Creating unstructured grid interior vs boundary error comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Test data for unstructured grids
    nx_unstruct, ny_unstruct = 16, 16

    # TriGrid analysis
    grid_tri = create_tri_grid(nx_unstruct, ny_unstruct, domain)
    centers_tri = grid_tri.get_cell_centers()
    x_tri, y_tri = centers_tri[:, 0], centers_tri[:, 1]
    scalar_tri = analytical_scalar(x_tri, y_tri)
    grad_x_ana_tri = analytical_gradient_x(x_tri, y_tri)
    grad_y_ana_tri = analytical_gradient_y(x_tri, y_tri)
    cell_areas_tri = grid_tri.get_cell_volumes()

    boundary_mask_tri = identify_boundary_cells_unstructured(grid_tri, domain, method='vertex')

    tri_results = {}
    for order, name in [(1, 'Green-Gauss'), (2, 'Least-Squares'), (4, 'Extended LS')]:
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid_tri, scalar_tri, order=order)
        err_x = compute_errors_unstructured(grad_x_num, grad_x_ana_tri, boundary_mask_tri, cell_areas_tri)
        err_y = compute_errors_unstructured(grad_y_num, grad_y_ana_tri, boundary_mask_tri, cell_areas_tri)
        tri_results[name] = {'x': err_x, 'y': err_y}

    # MixedGrid analysis
    grid_mixed = create_mixed_grid(nx_unstruct, ny_unstruct, domain, tri_fraction=0.3)
    centers_mixed = grid_mixed.get_cell_centers()
    x_mixed, y_mixed = centers_mixed[:, 0], centers_mixed[:, 1]
    scalar_mixed = analytical_scalar(x_mixed, y_mixed)
    grad_x_ana_mixed = analytical_gradient_x(x_mixed, y_mixed)
    grad_y_ana_mixed = analytical_gradient_y(x_mixed, y_mixed)
    cell_areas_mixed = grid_mixed.get_cell_volumes()

    boundary_mask_mixed = identify_boundary_cells_unstructured(grid_mixed, domain, method='vertex')

    mixed_results = {}
    for order, name in [(1, 'Green-Gauss'), (2, 'Least-Squares'), (4, 'Extended LS')]:
        grad_x_num, grad_y_num = compute_gradient_unstructured(grid_mixed, scalar_mixed, order=order)
        err_x = compute_errors_unstructured(grad_x_num, grad_x_ana_mixed, boundary_mask_mixed, cell_areas_mixed)
        err_y = compute_errors_unstructured(grad_y_num, grad_y_ana_mixed, boundary_mask_mixed, cell_areas_mixed)
        mixed_results[name] = {'x': err_x, 'y': err_y}

    # Row 1: TriGrid
    # Show boundary cells visualization
    ax = axes[0, 0]
    # Plot all cells with interior/boundary coloring
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection
    patches = []
    colors = []
    for i, cell in enumerate(grid_tri.cells):
        verts = grid_tri.vertices[cell]
        patches.append(MplPolygon(verts, closed=True))
        colors.append('red' if boundary_mask_tri[i] else 'green')
    pc = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=0.3, alpha=0.6)
    ax.add_collection(pc)
    ax.set_xlim(domain[0]-0.02, domain[1]+0.02)
    ax.set_ylim(domain[2]-0.02, domain[3]+0.02)
    ax.set_aspect('equal')
    ax.set_title(f'TriGrid: {grid_tri.n_cells} cells\n'
                 f'Interior: {np.sum(~boundary_mask_tri)} (green), Boundary: {np.sum(boundary_mask_tri)} (red)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Bar chart for TriGrid ∂f/∂x
    ax = axes[0, 1]
    methods = ['Green-Gauss', 'Least-Squares', 'Extended LS']
    x_pos = np.arange(len(methods))
    width = 0.35
    interior_errs = [tri_results[m]['x']['interior_L2'] for m in methods]
    boundary_errs = [tri_results[m]['x']['boundary_L2'] for m in methods]
    bars1 = ax.bar(x_pos - width/2, interior_errs, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errs, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('TriGrid ∂f/∂x: Interior vs Boundary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['GG', 'LS', 'ELS'])
    ax.legend()
    ax.set_yscale('log')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)

    # Bar chart for TriGrid ∂f/∂y
    ax = axes[0, 2]
    interior_errs = [tri_results[m]['y']['interior_L2'] for m in methods]
    boundary_errs = [tri_results[m]['y']['boundary_L2'] for m in methods]
    bars1 = ax.bar(x_pos - width/2, interior_errs, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errs, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('TriGrid ∂f/∂y: Interior vs Boundary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['GG', 'LS', 'ELS'])
    ax.legend()
    ax.set_yscale('log')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)

    # Row 2: MixedGrid
    ax = axes[1, 0]
    patches = []
    colors = []
    for i, cell in enumerate(grid_mixed.cells):
        verts = grid_mixed.vertices[cell]
        patches.append(MplPolygon(verts, closed=True))
        colors.append('red' if boundary_mask_mixed[i] else 'green')
    pc = PatchCollection(patches, facecolors=colors, edgecolors='black', linewidths=0.3, alpha=0.6)
    ax.add_collection(pc)
    ax.set_xlim(domain[0]-0.02, domain[1]+0.02)
    ax.set_ylim(domain[2]-0.02, domain[3]+0.02)
    ax.set_aspect('equal')
    ax.set_title(f'MixedGrid: {grid_mixed.n_cells} cells\n'
                 f'Interior: {np.sum(~boundary_mask_mixed)} (green), Boundary: {np.sum(boundary_mask_mixed)} (red)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Bar chart for MixedGrid ∂f/∂x
    ax = axes[1, 1]
    interior_errs = [mixed_results[m]['x']['interior_L2'] for m in methods]
    boundary_errs = [mixed_results[m]['x']['boundary_L2'] for m in methods]
    bars1 = ax.bar(x_pos - width/2, interior_errs, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errs, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('MixedGrid ∂f/∂x: Interior vs Boundary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['GG', 'LS', 'ELS'])
    ax.legend()
    ax.set_yscale('log')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)

    # Bar chart for MixedGrid ∂f/∂y
    ax = axes[1, 2]
    interior_errs = [mixed_results[m]['y']['interior_L2'] for m in methods]
    boundary_errs = [mixed_results[m]['y']['boundary_L2'] for m in methods]
    bars1 = ax.bar(x_pos - width/2, interior_errs, width, label='Interior', color='green', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, boundary_errs, width, label='Boundary', color='red', alpha=0.7)
    ax.set_ylabel('L2 Error')
    ax.set_title('MixedGrid ∂f/∂y: Interior vs Boundary')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['GG', 'LS', 'ELS'])
    ax.legend()
    ax.set_yscale('log')
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)
    for bar in bars2:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.1e}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', fontsize=6)

    plt.suptitle(f'Unstructured Grids: Interior vs Boundary Error Analysis\n'
                 f'Boundary cells identified by vertex-on-boundary method',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(f"{output_dir}/11_unstructured_interior_boundary.png", dpi=150)
    plt.close()

    print(f"\nAll visualizations saved to: {output_dir}/")
    print("  01_grid_structures.png  - Grid type comparison")
    print("  02_scalar_field.png     - Scalar field on all grids")
    print("  03_quad_gradient.png    - QuadGrid gradient analysis")
    print("  04_tri_gradient.png     - TriGrid gradient analysis")
    print("  05_mixed_gradient.png   - MixedGrid gradient analysis")
    print("  06_grid_detail.png      - Detailed view with nodes/centers")
    print("  07_convergence.png      - Non-periodic convergence comparison")
    print("  08_periodic_convergence.png - Periodic boundary convergence")
    print("  09_timing.png           - Execution time scaling")
    print("  10_interior_boundary_error.png - QuadGrid interior vs boundary")
    print("  11_unstructured_interior_boundary.png - TriGrid/MixedGrid interior vs boundary")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run all gradient verification tests and visualizations."""
    print("="*70)
    print("GRADIENT VERIFICATION TEST SUITE")
    print("="*70)
    print(f"Analytical function: f(x,y) = sin(πx) * cos(πy)")
    print(f"Domain: [0, 1] × [0, 1]")
    print()

    # Run pytest tests
    print("Running pytest tests...")
    test = TestGradientVerification()
    test.test_quad_grid_gradient()
    test.test_tri_grid_gradient()
    test.test_mixed_grid_gradient()

    # Run convergence study
    run_convergence_study()

    # Create visualizations
    create_visualizations()

    print("\n" + "="*70)
    print("ALL TESTS AND VISUALIZATIONS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
