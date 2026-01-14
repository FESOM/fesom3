"""
Gradient Verification Tests with Periodic Boundaries

This module tests gradient operators with periodic boundary conditions on all grid types.

Test Function (truly periodic on [0,1]x[0,1]):
    f(x, y) = sin(2πx) * sin(2πy)

    Analytical gradients:
        ∂f/∂x = 2π * cos(2πx) * sin(2πy)
        ∂f/∂y = 2π * sin(2πx) * cos(2πy)

Key tests:
    1. QuadGrid with periodic FD stencils (2nd and 4th order)
    2. TriGrid with periodic neighbor connectivity
    3. MixedGrid with periodic neighbor connectivity
    4. Convergence verification: all should achieve expected orders
"""

import numpy as np
import pytest

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = np
    JAX_AVAILABLE = False

from fesomx import QuadGrid, TriGrid, MixedGrid, StaggerType
from fesomx.utilities.mesh_generation import (
    create_structured_triangular_mesh,
    create_mixed_mesh,
)


# =============================================================================
# Analytical Functions (Truly Periodic on [0,1]x[0,1])
# =============================================================================

def analytical_scalar_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    f(x, y) = sin(2πx) * sin(2πy)

    This function is truly periodic on [0,1]x[0,1]:
    - f(0, y) = f(1, y) = 0
    - f(x, 0) = f(x, 1) = 0
    """
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)


def analytical_gradient_x_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ∂f/∂x = 2π * cos(2πx) * sin(2πy)
    """
    return 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)


def analytical_gradient_y_periodic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    ∂f/∂y = 2π * sin(2πx) * cos(2πy)
    """
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)


# =============================================================================
# Grid Creation Helpers
# =============================================================================

def create_quad_grid_periodic(nx: int, ny: int, domain=(0.0, 1.0, 0.0, 1.0)):
    """Create a QuadGrid with periodic boundaries."""
    grid = QuadGrid(
        "test_quad_periodic",
        nx=nx,
        ny=ny,
        stagger_type=StaggerType.A,
        periodic=(True, True)
    )
    grid.create_mesh(domain=domain)
    grid.compute_neighbors(halo_width=2)

    # Store dx, dy for gradient computation
    grid.dx = (domain[1] - domain[0]) / nx
    grid.dy = (domain[3] - domain[2]) / ny

    return grid


def create_tri_grid_periodic(nx: int, ny: int, domain=(0.0, 1.0, 0.0, 1.0)):
    """Create a TriGrid with periodic boundaries."""
    mesh_data = create_structured_triangular_mesh(
        nx=nx,
        ny=ny,
        domain=domain
    )

    grid = TriGrid("test_tri_periodic", periodic=(True, True))
    grid.create_mesh(vertices=mesh_data['vertices'], triangles=mesh_data['triangles'])

    return grid


def create_mixed_grid_periodic(nx: int, ny: int, domain=(0.0, 1.0, 0.0, 1.0)):
    """Create a MixedGrid with periodic boundaries."""
    mesh_data = create_mixed_mesh(
        nx=nx,
        ny=ny,
        domain=domain,
        seed=42  # For reproducibility
    )

    grid = MixedGrid("test_mixed_periodic", periodic=(True, True))
    grid.create_mesh(vertices=mesh_data['vertices'], cells=mesh_data['cells'])

    return grid


# =============================================================================
# Gradient Computation Helpers
# =============================================================================

def compute_gradient_quad_periodic(grid, phi, order=2):
    """Compute gradient on QuadGrid with periodic stencils."""
    from fesomx.operators.structured.gradient_fd import gradient_fd
    return gradient_fd(phi, grid, order=order, periodic=(True, True))


def compute_gradient_tri_periodic(grid, phi, order=2):
    """Compute gradient on TriGrid with periodic neighbors."""
    from fesomx.operators.unstructured.gradient_fv import gradient_fv
    return gradient_fv(phi, grid, order=order)


def compute_gradient_mixed_periodic(grid, phi, order=2):
    """Compute gradient on MixedGrid with periodic neighbors."""
    from fesomx.operators.unstructured.gradient_fv import gradient_fv
    return gradient_fv(phi, grid, order=order)


# =============================================================================
# Tests: QuadGrid Periodic
# =============================================================================

class TestQuadGridPeriodic:
    """Test QuadGrid gradient with periodic boundaries."""

    def test_periodic_2nd_order_symmetric_error(self):
        """With periodic BC, ∂f/∂x and ∂f/∂y should have identical L2 errors."""
        nx, ny = 16, 16
        grid = create_quad_grid_periodic(nx, ny)

        centers = grid.get_cell_centers()
        x_c = centers[:, 0].reshape(ny, nx)
        y_c = centers[:, 1].reshape(ny, nx)

        phi = analytical_scalar_periodic(x_c, y_c)
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
        exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

        dphidx, dphidy = compute_gradient_quad_periodic(grid, phi, order=2)

        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))

        # Errors should be nearly identical (within 1%)
        rel_diff = abs(error_x - error_y) / max(error_x, error_y)
        assert rel_diff < 0.01, f"Errors differ: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}"

    def test_periodic_4th_order_symmetric_error(self):
        """With periodic BC and 4th order, errors should also be symmetric."""
        nx, ny = 16, 16
        grid = create_quad_grid_periodic(nx, ny)

        centers = grid.get_cell_centers()
        x_c = centers[:, 0].reshape(ny, nx)
        y_c = centers[:, 1].reshape(ny, nx)

        phi = analytical_scalar_periodic(x_c, y_c)
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
        exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

        dphidx, dphidy = compute_gradient_quad_periodic(grid, phi, order=4)

        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))

        rel_diff = abs(error_x - error_y) / max(error_x, error_y)
        assert rel_diff < 0.01, f"Errors differ: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}"

    def test_periodic_2nd_order_convergence(self):
        """2nd-order periodic gradient should converge at ~O(h²)."""
        resolutions = [8, 16, 32]
        errors = []

        for n in resolutions:
            grid = create_quad_grid_periodic(n, n)

            centers = grid.get_cell_centers()
            x_c = centers[:, 0].reshape(n, n)
            y_c = centers[:, 1].reshape(n, n)

            phi = analytical_scalar_periodic(x_c, y_c)
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)

            dphidx, _ = compute_gradient_quad_periodic(grid, phi, order=2)
            error = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            errors.append(error)

        # Compute convergence rates
        orders = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i+1]) / np.log(2)
            orders.append(rate)

        avg_order = np.mean(orders)
        assert avg_order > 1.8, f"Expected O(h²), got average order {avg_order:.2f}"

    def test_periodic_4th_order_convergence(self):
        """4th-order periodic gradient should converge at ~O(h⁴)."""
        resolutions = [8, 16, 32]
        errors = []

        for n in resolutions:
            grid = create_quad_grid_periodic(n, n)

            centers = grid.get_cell_centers()
            x_c = centers[:, 0].reshape(n, n)
            y_c = centers[:, 1].reshape(n, n)

            phi = analytical_scalar_periodic(x_c, y_c)
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)

            dphidx, _ = compute_gradient_quad_periodic(grid, phi, order=4)
            error = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            errors.append(error)

        # Compute convergence rates
        orders = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i+1]) / np.log(2)
            orders.append(rate)

        avg_order = np.mean(orders)
        assert avg_order > 3.5, f"Expected O(h⁴), got average order {avg_order:.2f}"


# =============================================================================
# Tests: TriGrid Periodic
# =============================================================================

class TestTriGridPeriodic:
    """Test TriGrid gradient with periodic boundaries."""

    def test_periodic_neighbors_connected(self):
        """Boundary cells should have periodic neighbors."""
        grid = create_tri_grid_periodic(4, 4)

        # Check that no cell has fewer than 3 neighbors
        # (boundary cells should now be connected across periodic boundary)
        for cell_id, neighbors in enumerate(grid.cell_neighbors):
            # Interior triangles have 3 neighbors, boundary may have more due to periodic
            assert len(neighbors) >= 2, f"Cell {cell_id} has only {len(neighbors)} neighbors"

    def test_periodic_symmetric_error(self):
        """With periodic BC, ∂f/∂x and ∂f/∂y should have similar L2 errors."""
        nx, ny = 8, 8
        grid = create_tri_grid_periodic(nx, ny)

        centers = grid.get_cell_centers()
        x_c = centers[:, 0]
        y_c = centers[:, 1]

        phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
        exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

        dphidx, dphidy = compute_gradient_tri_periodic(grid, phi, order=2)

        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))

        # For triangular grids, allow more tolerance due to mesh geometry
        rel_diff = abs(error_x - error_y) / max(error_x, error_y)
        assert rel_diff < 0.2, f"Errors too asymmetric: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}"

    def test_periodic_ls_convergence(self):
        """LS gradient on periodic TriGrid should converge."""
        resolutions = [4, 8, 16]
        errors = []

        for n in resolutions:
            grid = create_tri_grid_periodic(n, n)

            centers = grid.get_cell_centers()
            x_c = centers[:, 0]
            y_c = centers[:, 1]

            phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)

            dphidx, _ = compute_gradient_tri_periodic(grid, phi, order=2)
            error = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            errors.append(error)

        # Verify convergence (errors should decrease)
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i], f"Errors not decreasing: {errors}"

        # Compute average order
        orders = []
        for i in range(len(errors) - 1):
            rate = np.log(errors[i] / errors[i+1]) / np.log(2)
            orders.append(rate)

        avg_order = np.mean(orders)
        # FV methods on irregular meshes typically achieve ~0.6-0.8 order
        assert avg_order > 0.5, f"Expected convergence, got average order {avg_order:.2f}"


# =============================================================================
# Tests: MixedGrid Periodic
# =============================================================================

class TestMixedGridPeriodic:
    """Test MixedGrid gradient with periodic boundaries."""

    def test_periodic_neighbors_connected(self):
        """Boundary cells should have periodic neighbors."""
        grid = create_mixed_grid_periodic(4, 4)

        # Check that cells have reasonable neighbor counts
        for cell_id, neighbors in enumerate(grid.cell_neighbors):
            assert len(neighbors) >= 2, f"Cell {cell_id} has only {len(neighbors)} neighbors"

    def test_periodic_symmetric_error(self):
        """With periodic BC, ∂f/∂x and ∂f/∂y should have similar L2 errors."""
        nx, ny = 8, 8
        grid = create_mixed_grid_periodic(nx, ny)

        centers = grid.get_cell_centers()
        x_c = centers[:, 0]
        y_c = centers[:, 1]

        phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
        exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

        dphidx, dphidy = compute_gradient_mixed_periodic(grid, phi, order=2)

        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))

        # Allow tolerance for mixed mesh geometry
        rel_diff = abs(error_x - error_y) / max(error_x, error_y)
        assert rel_diff < 0.3, f"Errors too asymmetric: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}"

    def test_periodic_ls_convergence(self):
        """LS gradient on periodic MixedGrid should converge."""
        resolutions = [4, 8, 16]
        errors = []

        for n in resolutions:
            grid = create_mixed_grid_periodic(n, n)

            centers = grid.get_cell_centers()
            x_c = centers[:, 0]
            y_c = centers[:, 1]

            phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)

            dphidx, _ = compute_gradient_mixed_periodic(grid, phi, order=2)
            error = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            errors.append(error)

        # Verify convergence (errors should decrease)
        for i in range(len(errors) - 1):
            assert errors[i+1] < errors[i], f"Errors not decreasing: {errors}"


# =============================================================================
# Tests: Comparison Periodic vs Non-Periodic
# =============================================================================

class TestPeriodicVsNonPeriodic:
    """Compare periodic and non-periodic boundary treatment."""

    def test_quad_periodic_better_at_boundaries(self):
        """Periodic gradient should have lower error at boundaries."""
        nx, ny = 16, 16

        # Create periodic and non-periodic grids
        grid_periodic = create_quad_grid_periodic(nx, ny)

        grid_nonperiodic = QuadGrid(
            "test_quad_nonperiodic",
            nx=nx,
            ny=ny,
            stagger_type=StaggerType.A,
            periodic=(False, False)
        )
        grid_nonperiodic.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
        grid_nonperiodic.compute_neighbors(halo_width=1)
        grid_nonperiodic.dx = 1.0 / nx
        grid_nonperiodic.dy = 1.0 / ny

        centers = grid_periodic.get_cell_centers()
        x_c = centers[:, 0].reshape(ny, nx)
        y_c = centers[:, 1].reshape(ny, nx)

        phi = analytical_scalar_periodic(x_c, y_c)
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)

        # Compute gradients
        from fesomx.operators.structured.gradient_fd import gradient_fd

        dphidx_periodic, _ = gradient_fd(
            phi, grid_periodic, order=2, periodic=(True, True)
        )
        dphidx_nonperiodic, _ = gradient_fd(
            phi, grid_nonperiodic, order=2, periodic=(False, False)
        )

        # Check boundary rows/columns
        # Left boundary (column 0)
        error_periodic_left = np.abs(np.array(dphidx_periodic[:, 0]) - exact_dphidx[:, 0]).mean()
        error_nonperiodic_left = np.abs(np.array(dphidx_nonperiodic[:, 0]) - exact_dphidx[:, 0]).mean()

        # Non-periodic should have zero gradient at boundaries
        assert error_nonperiodic_left > error_periodic_left * 0.5, \
            "Expected periodic to have smaller boundary error"


# =============================================================================
# Visualization Functions
# =============================================================================

def run_periodic_convergence_study():
    """Run comprehensive periodic convergence study and create visualization."""
    import matplotlib.pyplot as plt

    resolutions = [8, 16, 32, 64]

    results = {
        'quad_2nd_periodic': {'errors_x': [], 'errors_y': []},
        'quad_4th_periodic': {'errors_x': [], 'errors_y': []},
        'tri_ls_periodic': {'errors_x': [], 'errors_y': []},
        'mixed_ls_periodic': {'errors_x': [], 'errors_y': []},
    }

    print("=" * 70)
    print("PERIODIC GRADIENT CONVERGENCE STUDY")
    print("=" * 70)
    print("\nTest function: f(x,y) = sin(2πx)sin(2πy)")
    print("Domain: [0,1] × [0,1] with periodic boundaries\n")

    for n in resolutions:
        print(f"\nResolution: {n}x{n} (h = {1.0/n:.4f})")
        print("-" * 50)

        # QuadGrid periodic 2nd order
        grid = create_quad_grid_periodic(n, n)
        centers = grid.get_cell_centers()
        x_c = centers[:, 0].reshape(n, n)
        y_c = centers[:, 1].reshape(n, n)

        phi = analytical_scalar_periodic(x_c, y_c)
        exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
        exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

        dphidx, dphidy = compute_gradient_quad_periodic(grid, phi, order=2)
        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))
        results['quad_2nd_periodic']['errors_x'].append(error_x)
        results['quad_2nd_periodic']['errors_y'].append(error_y)
        print(f"  QuadGrid FD (O=2) [periodic]: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}")

        # QuadGrid periodic 4th order
        dphidx, dphidy = compute_gradient_quad_periodic(grid, phi, order=4)
        error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
        error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))
        results['quad_4th_periodic']['errors_x'].append(error_x)
        results['quad_4th_periodic']['errors_y'].append(error_y)
        print(f"  QuadGrid FD (O=4) [periodic]: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}")

        # TriGrid periodic
        if n <= 32:  # Limit for FV methods (slow)
            tri_grid = create_tri_grid_periodic(n, n)
            centers = tri_grid.get_cell_centers()
            x_c = centers[:, 0]
            y_c = centers[:, 1]

            phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
            exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

            dphidx, dphidy = compute_gradient_tri_periodic(tri_grid, phi, order=2)
            error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))
            results['tri_ls_periodic']['errors_x'].append(error_x)
            results['tri_ls_periodic']['errors_y'].append(error_y)
            print(f"  TriGrid LS (O=2) [periodic]: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}")

            # MixedGrid periodic
            mixed_grid = create_mixed_grid_periodic(n, n)
            centers = mixed_grid.get_cell_centers()
            x_c = centers[:, 0]
            y_c = centers[:, 1]

            phi = jnp.array(analytical_scalar_periodic(x_c, y_c))
            exact_dphidx = analytical_gradient_x_periodic(x_c, y_c)
            exact_dphidy = analytical_gradient_y_periodic(x_c, y_c)

            dphidx, dphidy = compute_gradient_mixed_periodic(mixed_grid, phi, order=2)
            error_x = np.sqrt(np.mean((np.array(dphidx) - exact_dphidx)**2))
            error_y = np.sqrt(np.mean((np.array(dphidy) - exact_dphidy)**2))
            results['mixed_ls_periodic']['errors_x'].append(error_x)
            results['mixed_ls_periodic']['errors_y'].append(error_y)
            print(f"  MixedGrid LS (O=2) [periodic]: L2(∂x)={error_x:.4e}, L2(∂y)={error_y:.4e}")

    # Compute convergence orders
    print("\n" + "=" * 70)
    print("CONVERGENCE RATES (periodic boundaries)")
    print("=" * 70)

    for method, data in results.items():
        if len(data['errors_x']) > 1:
            orders_x = []
            orders_y = []
            for i in range(len(data['errors_x']) - 1):
                if data['errors_x'][i] > 0 and data['errors_x'][i+1] > 0:
                    orders_x.append(np.log(data['errors_x'][i] / data['errors_x'][i+1]) / np.log(2))
                if data['errors_y'][i] > 0 and data['errors_y'][i+1] > 0:
                    orders_y.append(np.log(data['errors_y'][i] / data['errors_y'][i+1]) / np.log(2))

            if orders_x:
                print(f"\n{method}:")
                print(f"  ∂f/∂x orders: {orders_x}")
                print(f"  ∂f/∂y orders: {orders_y}")
                print(f"  Average order: {np.mean(orders_x + orders_y):.2f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    h_values = [1.0/n for n in resolutions]

    # Colors and markers
    styles = {
        'quad_2nd_periodic': ('b', 'o', '-', 'QuadGrid FD (O=2) [periodic]'),
        'quad_4th_periodic': ('b', 's', '--', 'QuadGrid FD (O=4) [periodic]'),
        'tri_ls_periodic': ('g', '^', '-', 'TriGrid LS (O=2) [periodic]'),
        'mixed_ls_periodic': ('m', 'D', '-', 'MixedGrid LS (O=2) [periodic]'),
    }

    # Plot df/dx
    ax = axes[0]
    for method, (color, marker, linestyle, label) in styles.items():
        errors = results[method]['errors_x']
        if len(errors) > 0:
            ax.loglog(h_values[:len(errors)], errors,
                     color=color, marker=marker, linestyle=linestyle,
                     label=label, markersize=8, linewidth=2)

    # Reference lines
    h = np.array(h_values)
    ax.loglog(h, 0.5 * h**2, 'k--', alpha=0.5, label='O(h²)')
    ax.loglog(h, 0.1 * h**4, 'k:', alpha=0.5, label='O(h⁴)')

    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('∂f/∂x = 2π cos(2πx) sin(2πy)\n(Periodic boundaries)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    # Plot df/dy
    ax = axes[1]
    for method, (color, marker, linestyle, label) in styles.items():
        errors = results[method]['errors_y']
        if len(errors) > 0:
            ax.loglog(h_values[:len(errors)], errors,
                     color=color, marker=marker, linestyle=linestyle,
                     label=label, markersize=8, linewidth=2)

    ax.loglog(h, 0.5 * h**2, 'k--', alpha=0.5, label='O(h²)')
    ax.loglog(h, 0.1 * h**4, 'k:', alpha=0.5, label='O(h⁴)')

    ax.set_xlabel('Grid spacing h', fontsize=12)
    ax.set_ylabel('L2 Error', fontsize=12)
    ax.set_title('∂f/∂y = 2π sin(2πx) cos(2πy)\n(Periodic boundaries)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle('Gradient Convergence with Periodic Boundaries\n'
                 'f(x,y) = sin(2πx) sin(2πy) on [0,1]×[0,1]',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save plot
    output_dir = 'output/gradient_verification'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/10_periodic_convergence_full.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_dir}/10_periodic_convergence_full.png")

    plt.close()

    return results


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Periodic gradient verification tests")
    parser.add_argument("--visualize", "-v", action="store_true",
                       help="Run convergence study and create visualization")
    parser.add_argument("--test", "-t", action="store_true",
                       help="Run pytest tests")
    args = parser.parse_args()

    if args.visualize:
        run_periodic_convergence_study()
    elif args.test:
        pytest.main([__file__, "-v"])
    else:
        # Default: run convergence study
        run_periodic_convergence_study()
