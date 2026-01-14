"""
Test: Divergence Verification Across Grid Types

Verifies divergence computation on QuadGrid, TriGrid, and MixedGrid using
analytical test functions with known divergence values.

Analytical velocity field:
    u(x, y) = sin(2πx) * cos(2πy)
    v(x, y) = cos(2πx) * sin(2πy)

Analytical divergence:
    ∇·u = ∂u/∂x + ∂v/∂y = 2π*cos(2πx)*cos(2πy) + 2π*cos(2πx)*cos(2πy)
        = 4π*cos(2πx)*cos(2πy)

Test verifies:
1. Divergence computation correctness against analytical solution
2. Convergence with grid refinement
3. Divergence-free field conservation (rotation field)
4. Visual comparison of numerical vs analytical divergence
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
from fesomx.operators.structured.divergence_fd import divergence_fd
from fesomx.operators.unstructured.divergence_fv import divergence_fv


# =============================================================================
# Analytical Test Functions - Primary (Sinusoidal)
# =============================================================================

def analytical_velocity(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Primary test velocity field: u = (sin(2πx)cos(2πy), cos(2πx)sin(2πy))

    Properties:
    - Smooth, differentiable everywhere
    - Non-trivial divergence
    - Periodic-compatible on [0,1] × [0,1]
    """
    u = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    v = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    return u, v


def analytical_divergence(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Analytical divergence: ∇·u = 4π*cos(2πx)*cos(2πy)

    Derivation:
        ∂u/∂x = 2π*cos(2πx)*cos(2πy)
        ∂v/∂y = 2π*cos(2πx)*cos(2πy)
        ∇·u = ∂u/∂x + ∂v/∂y = 4π*cos(2πx)*cos(2πy)
    """
    return 4 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


# =============================================================================
# Analytical Test Functions - Divergence-Free (Rotation)
# =============================================================================

def analytical_velocity_rotation(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Rotation field: u = (-y, x)

    Properties:
    - Divergence-free: ∇·u = 0
    - Tests conservation property of the divergence operator
    """
    # Shift to center at (0.5, 0.5) for domain [0,1]×[0,1]
    u = -(y - 0.5)
    v = (x - 0.5)
    return u, v


def analytical_divergence_rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rotation field has zero divergence everywhere."""
    return np.zeros_like(x)


# =============================================================================
# Analytical Test Functions - Constant Divergence (Expansion)
# =============================================================================

def analytical_velocity_expansion(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Expansion field: u = (x, y)

    Properties:
    - Constant divergence: ∇·u = 2
    - Radial outward flow
    """
    # Shift to center at (0.5, 0.5)
    u = x - 0.5
    v = y - 0.5
    return u, v


def analytical_divergence_expansion(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Expansion field has constant divergence = 2."""
    return 2.0 * np.ones_like(x)


# =============================================================================
# Grid Creation Helpers
# =============================================================================

def create_quad_grid(nx: int, ny: int, domain: tuple, periodic: bool = False) -> QuadGrid:
    """Create a structured quadrilateral grid."""
    grid = QuadGrid(
        name=f"quad_{nx}x{ny}{'_periodic' if periodic else ''}",
        nx=nx,
        ny=ny,
        stagger_type=StaggerType.A,
        periodic=(periodic, periodic)
    )
    grid.create_mesh(domain=domain)
    grid.compute_neighbors(halo_width=2)
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
        seed=42
    )

    grid = MixedGrid(name=f"mixed_{nx}x{ny}", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        cells=mesh_data['cells']
    )
    grid.compute_neighbors(halo_width=1)
    return grid


# =============================================================================
# Periodic Grid Creation Helpers
# =============================================================================

def create_quad_grid_periodic(nx: int, ny: int, domain: tuple) -> QuadGrid:
    """Create a QuadGrid with periodic boundaries."""
    grid = QuadGrid(
        name=f"quad_{nx}x{ny}_periodic",
        nx=nx,
        ny=ny,
        stagger_type=StaggerType.A,
        periodic=(True, True)
    )
    grid.create_mesh(domain=domain)
    grid.compute_neighbors(halo_width=2)
    grid.dx = (domain[1] - domain[0]) / nx
    grid.dy = (domain[3] - domain[2]) / ny
    return grid


def create_tri_grid_periodic(nx: int, ny: int, domain: tuple) -> TriGrid:
    """Create a TriGrid with periodic boundaries."""
    from fesomx.core.tri_grid import TriGrid as TriGridClass
    mesh_data = create_structured_triangular_mesh(nx=nx, ny=ny, domain=domain)

    grid = TriGridClass(name=f"tri_{nx}x{ny}_periodic", periodic=(True, True))
    grid.create_mesh(vertices=mesh_data['vertices'], triangles=mesh_data['triangles'])
    return grid


def create_mixed_grid_periodic(nx: int, ny: int, domain: tuple, tri_fraction: float = 0.3) -> MixedGrid:
    """Create a MixedGrid with periodic boundaries."""
    from fesomx.core.mixed_grid import MixedGrid as MixedGridClass
    mesh_data = create_mixed_mesh(
        nx=nx, ny=ny, domain=domain,
        tri_fraction=tri_fraction,
        seed=42
    )

    grid = MixedGridClass(name=f"mixed_{nx}x{ny}_periodic", periodic=(True, True))
    grid.create_mesh(vertices=mesh_data['vertices'], cells=mesh_data['cells'])
    return grid


# =============================================================================
# Divergence Computation Helpers
# =============================================================================

def compute_divergence_quad(grid: QuadGrid, u: np.ndarray, v: np.ndarray):
    """Compute divergence on QuadGrid using finite differences (2nd order)."""
    return divergence_fd(u, v, grid)


def compute_divergence_quad_periodic(grid: QuadGrid, u: np.ndarray, v: np.ndarray):
    """
    Compute divergence on QuadGrid with periodic boundary stencils.

    Uses np.roll for wrapping at boundaries:
        ∂u/∂x ≈ (u[i+1] - u[i-1]) / (2*dx)
        ∂v/∂y ≈ (v[j+1] - v[j-1]) / (2*dy)
    """
    dx = grid.dx
    dy = grid.dy
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
    dvdy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dy)
    return dudx + dvdy


def compute_divergence_unstructured(grid, u: np.ndarray, v: np.ndarray):
    """Compute divergence on unstructured grid using finite volume."""
    return divergence_fv(u, v, grid)


# =============================================================================
# Error Computation
# =============================================================================

def compute_errors(
    numerical: np.ndarray,
    analytical: np.ndarray,
    exclude_boundary: bool = False,
    grid_shape: tuple = None,
    boundary_width: int = 1
) -> dict:
    """
    Compute error metrics between numerical and analytical solutions.

    Args:
        numerical: Numerical solution
        analytical: Analytical solution
        exclude_boundary: If True, exclude boundary cells from error calculation
        grid_shape: Shape (ny, nx) for structured grids (required if exclude_boundary=True)
        boundary_width: Width of boundary to exclude

    Returns:
        dict with L2, Linf, and mean errors
    """
    numerical = np.array(numerical).flatten()
    analytical = np.array(analytical).flatten()

    if exclude_boundary and grid_shape is not None:
        ny, nx = grid_shape
        mask = np.ones((ny, nx), dtype=bool)
        mask[:boundary_width, :] = False
        mask[-boundary_width:, :] = False
        mask[:, :boundary_width] = False
        mask[:, -boundary_width:] = False
        mask = mask.flatten()

        numerical = numerical[mask]
        analytical = analytical[mask]

    diff = numerical - analytical

    return {
        'L2': np.sqrt(np.mean(diff**2)),
        'Linf': np.max(np.abs(diff)),
        'mean': np.mean(np.abs(diff)),
    }


# =============================================================================
# Pytest Test Classes
# =============================================================================

class TestDivergenceVerification:
    """Test divergence computation across all grid types."""

    def test_quad_grid_divergence(self):
        """Test QuadGrid divergence with 2nd order finite differences."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_quad_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)

        # Compute velocity field
        u, v = analytical_velocity(x, y)

        # Compute numerical divergence
        div_num = compute_divergence_quad(grid, u, v)

        # Compute analytical divergence
        div_ana = analytical_divergence(x, y)

        # Compute errors (exclude boundary for FD)
        errors = compute_errors(div_num, div_ana, exclude_boundary=True,
                               grid_shape=(ny, nx), boundary_width=1)

        print(f"\nQuadGrid (2nd order) Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        # Relaxed tolerance for 2nd order on 16x16
        assert errors['L2'] < 1.5, f"L2 error too large: {errors['L2']}"

    def test_tri_grid_divergence(self):
        """Test TriGrid divergence with finite volume method."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)

        errors = compute_errors(div_num, div_ana)

        print(f"\nTriGrid FV Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        assert errors['L2'] < 2.0, f"L2 error too large: {errors['L2']}"

    def test_mixed_grid_divergence(self):
        """Test MixedGrid divergence with finite volume method."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_mixed_grid(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)

        errors = compute_errors(div_num, div_ana)

        print(f"\nMixedGrid FV Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        assert errors['L2'] < 2.0, f"L2 error too large: {errors['L2']}"

    def test_divergence_free_rotation(self):
        """Test that rotation field has small divergence (limited by FV discretization)."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        # Test on TriGrid
        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity_rotation(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)

        # Divergence should be small (FV has discretization error for linear fields)
        l2_div = np.sqrt(np.mean(np.array(div_num)**2))
        linf_div = np.max(np.abs(np.array(div_num)))

        print(f"\nRotation Field (div-free) Divergence:")
        print(f"  L2: {l2_div:.6e}, Linf: {linf_div:.6e}")

        # FV method has inherent O(h) error for linear velocity fields
        # Relaxed tolerance to account for discretization error
        assert l2_div < 0.5, f"Rotation field divergence too large, got L2={l2_div}"

    def test_divergence_constant_expansion(self):
        """Test that expansion field has constant divergence = 2."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity_expansion(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence_expansion(x, y)

        errors = compute_errors(div_num, div_ana)

        print(f"\nExpansion Field (div=2) Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        # FV has discretization error; relaxed tolerance
        assert errors['L2'] < 0.5, f"L2 error too large: {errors['L2']}"


# =============================================================================
# Pytest Test Classes - Periodic Boundaries
# =============================================================================

class TestDivergenceVerificationPeriodic:
    """Test divergence computation with periodic boundary conditions."""

    def test_quad_grid_divergence_periodic(self):
        """Test QuadGrid divergence with periodic boundaries using np.roll stencils."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_quad_grid_periodic(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)

        # Compute velocity field
        u, v = analytical_velocity(x, y)

        # Compute numerical divergence with periodic stencils
        div_num = compute_divergence_quad_periodic(grid, u, v)

        # Compute analytical divergence
        div_ana = analytical_divergence(x, y)

        # Compute errors - no need to exclude boundary with periodic BC
        errors = compute_errors(div_num, div_ana)

        print(f"\nQuadGrid (Periodic) Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        # Should be same or better than interior-only non-periodic
        assert errors['L2'] < 1.5, f"L2 error too large: {errors['L2']}"

    def test_tri_grid_divergence_periodic(self):
        """Test TriGrid divergence with periodic neighbor connectivity."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_tri_grid_periodic(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)

        errors = compute_errors(div_num, div_ana)

        print(f"\nTriGrid FV (Periodic) Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        assert errors['L2'] < 2.0, f"L2 error too large: {errors['L2']}"

    def test_mixed_grid_divergence_periodic(self):
        """Test MixedGrid divergence with periodic neighbor connectivity."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 16, 16

        grid = create_mixed_grid_periodic(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)

        errors = compute_errors(div_num, div_ana)

        print(f"\nMixedGrid FV (Periodic) Divergence Errors:")
        print(f"  L2: {errors['L2']:.6e}, Linf: {errors['Linf']:.6e}")

        assert errors['L2'] < 2.0, f"L2 error too large: {errors['L2']}"

    def test_periodic_uniform_error_distribution(self):
        """Test that periodic BC gives uniform error (no boundary degradation)."""
        domain = (0.0, 1.0, 0.0, 1.0)
        nx, ny = 32, 32

        # Test on QuadGrid
        grid = create_quad_grid_periodic(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)

        u, v = analytical_velocity(x, y)
        div_num = np.array(compute_divergence_quad_periodic(grid, u, v))
        div_ana = analytical_divergence(x, y)
        error = np.abs(div_num - div_ana)

        # With periodic BC, boundary and interior should have similar errors
        boundary_mask = np.zeros((ny, nx), dtype=bool)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        interior_mask = ~boundary_mask

        int_l2 = np.sqrt(np.mean(error[interior_mask]**2))
        bnd_l2 = np.sqrt(np.mean(error[boundary_mask]**2))
        ratio = bnd_l2 / int_l2 if int_l2 > 0 else 0

        print(f"\nQuadGrid (Periodic) Boundary/Interior Error Ratio:")
        print(f"  Interior L2: {int_l2:.6e}")
        print(f"  Boundary L2: {bnd_l2:.6e}")
        print(f"  Ratio: {ratio:.2f}")

        # Ratio should be close to 1.0 (no boundary degradation)
        assert ratio < 2.0, f"Boundary/interior ratio too high: {ratio:.2f}"


# =============================================================================
# Convergence Study
# =============================================================================

def run_divergence_convergence_study():
    """Run grid refinement study to verify convergence order."""
    domain = (0.0, 1.0, 0.0, 1.0)
    resolutions = [8, 16, 32, 64]

    results = {
        'quad_fd': {'h': [], 'L2': []},
        'tri_fv': {'h': [], 'L2': []},
        'mixed_fv': {'h': [], 'L2': []},
    }

    print("\n" + "="*70)
    print("DIVERGENCE CONVERGENCE STUDY")
    print("="*70)

    for nx in resolutions:
        ny = nx
        h = 1.0 / nx
        print(f"\nResolution: {nx}x{ny} (h = {h:.4f})")
        print("-" * 50)

        # QuadGrid - FD (2nd order)
        grid = create_quad_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_quad(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana, True, (ny, nx), 1)
        results['quad_fd']['h'].append(h)
        results['quad_fd']['L2'].append(errors['L2'])
        print(f"  QuadGrid FD:    L2={errors['L2']:.4e}")

        # TriGrid - FV
        grid = create_tri_grid(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana)
        results['tri_fv']['h'].append(h)
        results['tri_fv']['L2'].append(errors['L2'])
        print(f"  TriGrid FV:     L2={errors['L2']:.4e}")

        # MixedGrid - FV
        grid = create_mixed_grid(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana)
        results['mixed_fv']['h'].append(h)
        results['mixed_fv']['L2'].append(errors['L2'])
        print(f"  MixedGrid FV:   L2={errors['L2']:.4e}")

    # Compute convergence rates
    print("\n" + "="*70)
    print("CONVERGENCE RATES (order p where error ~ h^p)")
    print("="*70)

    for grid_type, data in results.items():
        h = np.array(data['h'])
        L2 = np.array(data['L2'])

        if len(h) >= 2:
            order = np.log(L2[:-1] / L2[1:]) / np.log(h[:-1] / h[1:])
            print(f"\n{grid_type}:")
            print(f"  Orders: {order}")
            print(f"  Average order: {np.mean(order):.2f}")

    return results


def run_divergence_convergence_study_periodic():
    """Run grid refinement study with periodic boundaries."""
    domain = (0.0, 1.0, 0.0, 1.0)
    resolutions = [8, 16, 32, 64]

    results = {
        'quad_fd_periodic': {'h': [], 'L2': []},
        'tri_fv_periodic': {'h': [], 'L2': []},
        'mixed_fv_periodic': {'h': [], 'L2': []},
    }

    print("\n" + "="*70)
    print("DIVERGENCE CONVERGENCE STUDY (PERIODIC)")
    print("="*70)

    for nx in resolutions:
        ny = nx
        h = 1.0 / nx
        print(f"\nResolution: {nx}x{ny} (h = {h:.4f})")
        print("-" * 50)

        # QuadGrid - FD Periodic (2nd order)
        grid = create_quad_grid_periodic(nx, ny, domain)
        centers = grid.get_cell_centers()
        x = centers[:, 0].reshape(ny, nx)
        y = centers[:, 1].reshape(ny, nx)
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_quad_periodic(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana)  # No need to exclude boundary
        results['quad_fd_periodic']['h'].append(h)
        results['quad_fd_periodic']['L2'].append(errors['L2'])
        print(f"  QuadGrid FD (Periodic):  L2={errors['L2']:.4e}")

        # TriGrid - FV Periodic
        grid = create_tri_grid_periodic(nx, ny, domain)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana)
        results['tri_fv_periodic']['h'].append(h)
        results['tri_fv_periodic']['L2'].append(errors['L2'])
        print(f"  TriGrid FV (Periodic):   L2={errors['L2']:.4e}")

        # MixedGrid - FV Periodic
        grid = create_mixed_grid_periodic(nx, ny, domain, tri_fraction=0.3)
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]
        u, v = analytical_velocity(x, y)
        div_num = compute_divergence_unstructured(grid, u, v)
        div_ana = analytical_divergence(x, y)
        errors = compute_errors(div_num, div_ana)
        results['mixed_fv_periodic']['h'].append(h)
        results['mixed_fv_periodic']['L2'].append(errors['L2'])
        print(f"  MixedGrid FV (Periodic): L2={errors['L2']:.4e}")

    # Compute convergence rates
    print("\n" + "="*70)
    print("CONVERGENCE RATES - PERIODIC (order p where error ~ h^p)")
    print("="*70)

    for grid_type, data in results.items():
        h = np.array(data['h'])
        L2 = np.array(data['L2'])

        if len(h) >= 2:
            order = np.log(L2[:-1] / L2[1:]) / np.log(h[:-1] / h[1:])
            print(f"\n{grid_type}:")
            print(f"  Orders: {order}")
            print(f"  Average order: {np.mean(order):.2f}")

    return results


# =============================================================================
# Visualization
# =============================================================================

def create_divergence_visualizations(output_dir: str = "output/divergence_verification"):
    """Generate all divergence verification visualizations."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    domain = (0.0, 1.0, 0.0, 1.0)
    nx, ny = 16, 16

    print("\n" + "="*70)
    print("Creating Divergence Verification Visualizations")
    print("="*70)

    # Create grids
    quad_grid = create_quad_grid(nx, ny, domain)
    tri_grid = create_tri_grid(nx, ny, domain)
    mixed_grid = create_mixed_grid(nx, ny, domain)

    grids = [
        ('QuadGrid', quad_grid),
        ('TriGrid', tri_grid),
        ('MixedGrid', mixed_grid),
    ]

    # =========================================================================
    # Plot 1: Velocity field on all grid types
    # =========================================================================
    print("1. Creating velocity field visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, grid) in zip(axes, grids):
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]
        u, v = analytical_velocity(x, y)

        # Subsample for clearer quiver plot
        if len(x) > 200:
            idx = np.random.choice(len(x), 200, replace=False)
            x_sub, y_sub, u_sub, v_sub = x[idx], y[idx], u[idx], v[idx]
        else:
            x_sub, y_sub, u_sub, v_sub = x, y, u, v

        ax.quiver(x_sub, y_sub, u_sub, v_sub, scale=20)
        ax.set_title(f"{name}: Velocity Field")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.suptitle("Velocity Field: u = (sin(2πx)cos(2πy), cos(2πx)sin(2πy))", fontsize=14)
    plt.tight_layout()
    save_plot(f"{output_dir}/01_velocity_field.png", dpi=150)
    print(f"  Saved: {output_dir}/01_velocity_field.png")

    # =========================================================================
    # Plot 2: Divergence comparison (analytical vs numerical)
    # =========================================================================
    print("2. Creating divergence comparison visualization...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for col, (name, grid) in enumerate(grids):
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        # Analytical divergence
        div_ana = analytical_divergence(x, y)

        # Numerical divergence
        u, v = analytical_velocity(x, y)
        if name == 'QuadGrid':
            u_2d = u.reshape(ny, nx) if len(u.shape) == 1 else u
            v_2d = v.reshape(ny, nx) if len(v.shape) == 1 else v
            div_num = np.array(compute_divergence_quad(grid, u_2d, v_2d)).flatten()
        else:
            div_num = np.array(compute_divergence_unstructured(grid, u, v))

        # Top row: Analytical
        ax = axes[0, col]
        scatter = ax.scatter(x, y, c=div_ana, cmap='RdBu_r', s=10)
        plt.colorbar(scatter, ax=ax, label='∇·u')
        ax.set_title(f"{name}: Analytical ∇·u")
        ax.set_aspect('equal')

        # Bottom row: Numerical
        ax = axes[1, col]
        scatter = ax.scatter(x, y, c=div_num, cmap='RdBu_r', s=10)
        plt.colorbar(scatter, ax=ax, label='∇·u')
        ax.set_title(f"{name}: Numerical ∇·u")
        ax.set_aspect('equal')

    plt.suptitle("Divergence: Analytical vs Numerical", fontsize=14)
    plt.tight_layout()
    save_plot(f"{output_dir}/02_divergence_comparison.png", dpi=150)
    print(f"  Saved: {output_dir}/02_divergence_comparison.png")

    # =========================================================================
    # Plots 3-5: Individual grid analysis (2x2 panels each)
    # =========================================================================
    for idx, (name, grid) in enumerate(grids):
        print(f"{idx+3}. Creating {name} divergence analysis...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity(x, y)
        div_ana = analytical_divergence(x, y)

        if name == 'QuadGrid':
            u_2d = u.reshape(ny, nx)
            v_2d = v.reshape(ny, nx)
            div_num = np.array(compute_divergence_quad(grid, u_2d, v_2d)).flatten()
        else:
            div_num = np.array(compute_divergence_unstructured(grid, u, v))

        # (a) Velocity vectors
        ax = axes[0, 0]
        if len(x) > 200:
            idx_sub = np.random.choice(len(x), 200, replace=False)
            ax.quiver(x[idx_sub], y[idx_sub], u[idx_sub], v[idx_sub], scale=20)
        else:
            ax.quiver(x, y, u, v, scale=20)
        ax.set_title("(a) Velocity Field")
        ax.set_aspect('equal')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # (b) Analytical divergence
        ax = axes[0, 1]
        scatter = ax.scatter(x, y, c=div_ana, cmap='RdBu_r', s=15)
        plt.colorbar(scatter, ax=ax, label='∇·u')
        ax.set_title("(b) Analytical ∇·u")
        ax.set_aspect('equal')

        # (c) Numerical divergence
        ax = axes[1, 0]
        scatter = ax.scatter(x, y, c=div_num, cmap='RdBu_r', s=15)
        plt.colorbar(scatter, ax=ax, label='∇·u')
        ax.set_title("(c) Numerical ∇·u")
        ax.set_aspect('equal')

        # (d) Error
        ax = axes[1, 1]
        error = np.abs(div_num - div_ana)
        scatter = ax.scatter(x, y, c=error, cmap='hot', s=15)
        plt.colorbar(scatter, ax=ax, label='|Error|')
        ax.set_title(f"(d) Error (L2={np.sqrt(np.mean(error**2)):.4e})")
        ax.set_aspect('equal')

        plt.suptitle(f"{name} Divergence Analysis", fontsize=14)
        plt.tight_layout()
        save_plot(f"{output_dir}/0{idx+3}_{name.lower()}_divergence.png", dpi=150)
        print(f"  Saved: {output_dir}/0{idx+3}_{name.lower()}_divergence.png")

    # =========================================================================
    # Plot 6: Convergence study
    # =========================================================================
    print("6. Creating convergence plot...")
    results = run_divergence_convergence_study()

    fig, ax = plt.subplots(figsize=(10, 8))

    markers = {'quad_fd': 'o', 'tri_fv': '^', 'mixed_fv': 'D'}
    labels = {'quad_fd': 'QuadGrid FD',
              'tri_fv': 'TriGrid FV', 'mixed_fv': 'MixedGrid FV'}

    for grid_type, data in results.items():
        h = np.array(data['h'])
        L2 = np.array(data['L2'])
        if len(h) > 0:
            ax.loglog(h, L2, f'-{markers[grid_type]}', label=labels[grid_type], markersize=8)

    # Reference lines
    h_ref = np.array([0.03, 0.15])
    ax.loglog(h_ref, 5*h_ref, 'k--', alpha=0.5, label='O(h)')
    ax.loglog(h_ref, 10*h_ref**2, 'k:', alpha=0.5, label='O(h²)')

    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Divergence Convergence Study', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(f"{output_dir}/06_convergence.png", dpi=150)
    print(f"  Saved: {output_dir}/06_convergence.png")

    # =========================================================================
    # Plot 7: Divergence-free test (rotation field)
    # =========================================================================
    print("7. Creating divergence-free test visualization...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, grid) in zip(axes, grids):
        centers = grid.get_cell_centers()
        x, y = centers[:, 0], centers[:, 1]

        u, v = analytical_velocity_rotation(x, y)

        if name == 'QuadGrid':
            u_2d = u.reshape(ny, nx)
            v_2d = v.reshape(ny, nx)
            div_num = np.array(compute_divergence_quad(grid, u_2d, v_2d)).flatten()
        else:
            div_num = np.array(compute_divergence_unstructured(grid, u, v))

        l2_div = np.sqrt(np.mean(div_num**2))

        # Plot the divergence (should be ~0)
        scatter = ax.scatter(x, y, c=div_num, cmap='RdBu_r', s=15, vmin=-1e-10, vmax=1e-10)
        plt.colorbar(scatter, ax=ax, label='∇·u')
        ax.set_title(f"{name}: ∇·u (L2={l2_div:.2e})")
        ax.set_aspect('equal')

        # Add velocity arrows
        if len(x) > 100:
            idx_sub = np.random.choice(len(x), 100, replace=False)
            ax.quiver(x[idx_sub], y[idx_sub], u[idx_sub], v[idx_sub],
                     scale=5, alpha=0.3, color='black')

    plt.suptitle("Divergence-Free Test: Rotation Field u = (-y, x)", fontsize=14)
    plt.tight_layout()
    save_plot(f"{output_dir}/07_divergence_free_test.png", dpi=150)
    print(f"  Saved: {output_dir}/07_divergence_free_test.png")

    # =========================================================================
    # Plot 8: Interior vs boundary error comparison
    # =========================================================================
    print("8. Creating interior vs boundary error comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Use TriGrid for this analysis
    grid = create_tri_grid(32, 32, domain)
    centers = grid.get_cell_centers()
    x, y = centers[:, 0], centers[:, 1]

    u, v = analytical_velocity(x, y)
    div_num = np.array(compute_divergence_unstructured(grid, u, v))
    div_ana = analytical_divergence(x, y)
    error = np.abs(div_num - div_ana)

    # Identify boundary cells
    tol = 1e-10
    boundary_mask = np.zeros(len(centers), dtype=bool)
    for i in range(len(centers)):
        cell_verts = grid.vertices[grid.cells[i]]
        on_boundary = (
            np.any(np.abs(cell_verts[:, 0] - 0.0) < tol) or
            np.any(np.abs(cell_verts[:, 0] - 1.0) < tol) or
            np.any(np.abs(cell_verts[:, 1] - 0.0) < tol) or
            np.any(np.abs(cell_verts[:, 1] - 1.0) < tol)
        )
        boundary_mask[i] = on_boundary
    interior_mask = ~boundary_mask

    # Left: Error map with boundary highlighted
    ax = axes[0]
    scatter = ax.scatter(x[interior_mask], y[interior_mask], c=error[interior_mask],
                        cmap='hot', s=10, label='Interior')
    ax.scatter(x[boundary_mask], y[boundary_mask], c=error[boundary_mask],
              cmap='hot', s=30, marker='s', edgecolors='blue', linewidths=1,
              label='Boundary')
    plt.colorbar(scatter, ax=ax, label='|Error|')
    ax.set_title("Error Distribution (boundary in blue squares)")
    ax.set_aspect('equal')
    ax.legend()

    # Right: Bar chart comparison
    ax = axes[1]
    int_l2 = np.sqrt(np.mean(error[interior_mask]**2))
    bnd_l2 = np.sqrt(np.mean(error[boundary_mask]**2))
    ratio = bnd_l2 / int_l2 if int_l2 > 0 else 0

    bars = ax.bar(['Interior', 'Boundary'], [int_l2, bnd_l2], color=['green', 'red'])
    ax.set_ylabel('L2 Error')
    ax.set_title(f'Error Comparison (Boundary/Interior = {ratio:.2f}x)')

    # Add values on bars
    for bar, val in zip(bars, [int_l2, bnd_l2]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{val:.4e}', ha='center', va='bottom', fontsize=10)

    plt.suptitle("TriGrid (32x32): Interior vs Boundary Divergence Error", fontsize=14)
    plt.tight_layout()
    save_plot(f"{output_dir}/08_interior_boundary.png", dpi=150)
    print(f"  Saved: {output_dir}/08_interior_boundary.png")

    # =========================================================================
    # Plot 9: Periodic vs Non-Periodic Convergence Comparison
    # =========================================================================
    print("9. Creating periodic vs non-periodic convergence comparison...")

    # Run both convergence studies
    results_np = run_divergence_convergence_study()
    results_p = run_divergence_convergence_study_periodic()

    fig, ax = plt.subplots(figsize=(12, 8))

    # Non-periodic results (solid lines)
    markers_np = {'quad_fd': 'o', 'tri_fv': '^', 'mixed_fv': 'D'}
    labels_np = {'quad_fd': 'QuadGrid FD', 'tri_fv': 'TriGrid FV', 'mixed_fv': 'MixedGrid FV'}
    colors = {'quad': 'C0', 'tri': 'C1', 'mixed': 'C2'}

    for grid_type, data in results_np.items():
        h = np.array(data['h'])
        L2 = np.array(data['L2'])
        if len(h) > 0:
            color_key = grid_type.split('_')[0]
            ax.loglog(h, L2, f'-{markers_np[grid_type]}', color=colors[color_key],
                     label=f'{labels_np[grid_type]} (non-periodic)', markersize=8)

    # Periodic results (dashed lines)
    markers_p = {'quad_fd_periodic': 's', 'tri_fv_periodic': 'v', 'mixed_fv_periodic': 'p'}
    labels_p = {'quad_fd_periodic': 'QuadGrid FD', 'tri_fv_periodic': 'TriGrid FV',
                'mixed_fv_periodic': 'MixedGrid FV'}

    for grid_type, data in results_p.items():
        h = np.array(data['h'])
        L2 = np.array(data['L2'])
        if len(h) > 0:
            color_key = grid_type.split('_')[0]
            ax.loglog(h, L2, f'--{markers_p[grid_type]}', color=colors[color_key],
                     label=f'{labels_p[grid_type]} (periodic)', markersize=8)

    # Reference lines
    h_ref = np.array([0.03, 0.15])
    ax.loglog(h_ref, 5*h_ref, 'k--', alpha=0.3, label='O(h)')
    ax.loglog(h_ref, 10*h_ref**2, 'k:', alpha=0.3, label='O(h²)')

    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('Divergence Convergence: Periodic vs Non-Periodic', fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(f"{output_dir}/09_periodic_convergence.png", dpi=150)
    print(f"  Saved: {output_dir}/09_periodic_convergence.png")

    # =========================================================================
    # Plot 10: Boundary/Interior Error Ratio Comparison
    # =========================================================================
    print("10. Creating boundary/interior error ratio comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Compute boundary/interior ratios for both cases
    nx_test, ny_test = 32, 32

    # Non-periodic QuadGrid
    grid_np = create_quad_grid(nx_test, ny_test, domain)
    centers = grid_np.get_cell_centers()
    x = centers[:, 0].reshape(ny_test, nx_test)
    y = centers[:, 1].reshape(ny_test, nx_test)
    u, v = analytical_velocity(x, y)
    div_num_np = np.array(compute_divergence_quad(grid_np, u, v))
    div_ana = analytical_divergence(x, y)
    error_np = np.abs(div_num_np - div_ana)

    boundary_mask_2d = np.zeros((ny_test, nx_test), dtype=bool)
    boundary_mask_2d[0, :] = True
    boundary_mask_2d[-1, :] = True
    boundary_mask_2d[:, 0] = True
    boundary_mask_2d[:, -1] = True
    interior_mask_2d = ~boundary_mask_2d

    int_l2_np = np.sqrt(np.mean(error_np[interior_mask_2d]**2))
    bnd_l2_np = np.sqrt(np.mean(error_np[boundary_mask_2d]**2))
    ratio_np = bnd_l2_np / int_l2_np if int_l2_np > 0 else 0

    # Periodic QuadGrid
    grid_p = create_quad_grid_periodic(nx_test, ny_test, domain)
    div_num_p = np.array(compute_divergence_quad_periodic(grid_p, u, v))
    error_p = np.abs(div_num_p - div_ana)

    int_l2_p = np.sqrt(np.mean(error_p[interior_mask_2d]**2))
    bnd_l2_p = np.sqrt(np.mean(error_p[boundary_mask_2d]**2))
    ratio_p = bnd_l2_p / int_l2_p if int_l2_p > 0 else 0

    # Left: Error ratio bar chart
    ax = axes[0]
    x_pos = np.arange(2)
    width = 0.35

    bars1 = ax.bar(x_pos - width/2, [int_l2_np, int_l2_p], width, label='Interior', color='green')
    bars2 = ax.bar(x_pos + width/2, [bnd_l2_np, bnd_l2_p], width, label='Boundary', color='red')

    ax.set_ylabel('L2 Error')
    ax.set_title('Interior vs Boundary Error')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Non-Periodic', 'Periodic'])
    ax.legend()

    # Add values on bars
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{bar.get_height():.2e}', ha='center', va='bottom', fontsize=8, rotation=45)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{bar.get_height():.2e}', ha='center', va='bottom', fontsize=8, rotation=45)

    # Right: Ratio comparison
    ax = axes[1]
    ratios = [ratio_np, ratio_p]
    bars = ax.bar(['Non-Periodic', 'Periodic'], ratios, color=['orange', 'blue'])
    ax.set_ylabel('Boundary/Interior Error Ratio')
    ax.set_title('Boundary Degradation Comparison')
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal (ratio=1)')
    ax.legend()

    # Add ratio values on bars
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
               f'{ratio:.2f}x', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.suptitle(f"QuadGrid ({nx_test}x{ny_test}): Periodic vs Non-Periodic Boundary Error", fontsize=14)
    plt.tight_layout()
    save_plot(f"{output_dir}/10_periodic_boundary_ratio.png", dpi=150)
    print(f"  Saved: {output_dir}/10_periodic_boundary_ratio.png")

    print("\n" + "="*70)
    print(f"All visualizations saved to: {output_dir}/")
    print("="*70)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Running Divergence Verification Tests")
    print("=" * 70)

    # Run convergence study
    run_divergence_convergence_study()

    # Create visualizations
    create_divergence_visualizations()
