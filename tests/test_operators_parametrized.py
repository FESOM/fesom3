"""
Parametrized Tests for Operators on Quad and Triangle Grids

Tests all operators (divergence, gradient, laplacian) on both
structured (QuadGrid) and unstructured (TriGrid) grids.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from fesomx import QuadGrid, TriGrid, StaggerType, divergence, gradient, laplacian

# Import test helpers
from .test_helpers import (
    create_structured_triangular_mesh,
    create_divergence_free_field_unstructured,
    create_gaussian_field_unstructured,
    analytical_gradient_gaussian_unstructured,
    analytical_laplacian_gaussian_unstructured,
    compute_error
)

# Import structured grid helpers from existing operators
from fesomx.operators.divergence import divergence_free_field_2d
from fesomx.operators.gradient import smooth_gaussian_field_2d, analytical_gradient_gaussian
from fesomx.operators.laplacian import analytical_laplacian_gaussian


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=['quad', 'tri'])
def grid_type(request):
    """Grid type to test."""
    return request.param


@pytest.fixture(params=[32])  # Single size for now to keep tests fast
def grid_size(request):
    """Grid size to test."""
    return request.param


@pytest.fixture(params=[StaggerType.A])  # Only A-grid for triangles
def stagger_type(request):
    """Stagger type (only A for triangles)."""
    return request.param


@pytest.fixture
def create_grid(grid_type, grid_size, stagger_type):
    """Factory to create appropriate grid type."""

    def _create_grid(nx=None, ny=None, stagger=None):
        _nx = nx or grid_size
        _ny = ny or grid_size
        _stagger = stagger or stagger_type

        # Skip unsupported combinations
        if grid_type == 'tri' and _stagger != StaggerType.A:
            pytest.skip("Triangular grids only support A-grid staggering")

        if grid_type == 'quad':
            grid = QuadGrid(
                name="test_grid",
                nx=_nx,
                ny=_ny,
                stagger_type=_stagger,
                periodic=(False, False)
            )
            grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
            grid.dx = 1.0 / (_nx - 1)
            grid.dy = 1.0 / (_ny - 1)

        elif grid_type == 'tri':
            # Create structured triangular mesh
            mesh_data = create_structured_triangular_mesh(_nx, _ny, domain=(0.0, 1.0, 0.0, 1.0))

            grid = TriGrid(name="test_grid", stagger_type=StaggerType.A)
            grid.create_mesh(vertices=mesh_data['vertices'], triangles=mesh_data['triangles'])
            grid.compute_neighbors(halo_width=1)

        return grid, grid_type

    return _create_grid


# ============================================================================
# Divergence Tests
# ============================================================================

class TestDivergenceParametrized:
    """Parametrized tests for divergence on quad and triangle grids."""

    def test_divergence_free_field(self, create_grid):
        """Test divergence of divergence-free field."""
        grid, grid_type = create_grid()

        if grid_type == 'quad':
            # Structured grid: create field on 2D array
            x = np.linspace(0, 1, grid.nx)
            y = np.linspace(0, 1, grid.ny)
            X, Y = np.meshgrid(x, y)
            u, v = divergence_free_field_2d(jnp.array(X), jnp.array(Y), mode='rotation')

        else:  # tri
            # Unstructured grid: create field at cell centers
            u, v = create_divergence_free_field_unstructured(grid, mode='rotation')

        # Compute divergence
        div = divergence(u, v, grid, perform_halo_exchange=False)

        # Check error
        error = compute_error(div, jnp.zeros_like(div), grid, margin=2)

        # Triangles less accurate due to mesh irregularity
        tolerance = 1e-1 if grid_type == 'tri' else 1e-2
        assert error < tolerance, f"Divergence error {error:.2e} too large for {grid_type}-grid"

    def test_divergence_shape(self, create_grid):
        """Test that divergence returns correct shape."""
        grid, grid_type = create_grid()

        if grid_type == 'quad':
            u = jnp.zeros((grid.ny, grid.nx))
            v = jnp.zeros((grid.ny, grid.nx))
            expected_shape = (grid.ny, grid.nx)
        else:  # tri
            u = jnp.zeros(grid.n_cells)
            v = jnp.zeros(grid.n_cells)
            expected_shape = (grid.n_cells,)

        div = divergence(u, v, grid, perform_halo_exchange=False)

        assert div.shape == expected_shape, \
            f"Divergence shape {div.shape} != {expected_shape} for {grid_type}-grid"


# ============================================================================
# Gradient Tests
# ============================================================================

class TestGradientParametrized:
    """Parametrized tests for gradient on quad and triangle grids."""

    def test_gradient_gaussian(self, create_grid):
        """Test gradient of Gaussian field."""
        grid, grid_type = create_grid(nx=64, ny=64)  # Larger grid for better accuracy

        if grid_type == 'quad':
            # Structured grid
            x = np.linspace(0, 1, grid.nx)
            y = np.linspace(0, 1, grid.ny)
            X, Y = np.meshgrid(x, y)

            phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
            dphidx_exact, dphidy_exact = analytical_gradient_gaussian(
                jnp.array(X), jnp.array(Y), sigma=0.15
            )

        else:  # tri
            # Unstructured grid
            phi = create_gaussian_field_unstructured(grid, sigma=0.15)
            dphidx_exact, dphidy_exact = analytical_gradient_gaussian_unstructured(
                grid, sigma=0.15
            )

        # Numerical gradient
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        # Check errors
        error_x = compute_error(dphidx, dphidx_exact, grid, margin=3)
        error_y = compute_error(dphidy, dphidy_exact, grid, margin=3)

        # Triangles less accurate
        tolerance = 1.0 if grid_type == 'tri' else 0.5
        assert error_x < tolerance, f"Gradient x-error {error_x:.2e} too large for {grid_type}-grid"
        assert error_y < tolerance, f"Gradient y-error {error_y:.2e} too large for {grid_type}-grid"

    def test_gradient_constant(self, create_grid):
        """Test gradient of constant field (should be zero)."""
        grid, grid_type = create_grid()

        if grid_type == 'quad':
            phi = jnp.ones((grid.ny, grid.nx)) * 42.0
        else:  # tri
            phi = jnp.ones(grid.n_cells) * 42.0

        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        error_x = compute_error(dphidx, jnp.zeros_like(dphidx), grid, margin=1)
        error_y = compute_error(dphidy, jnp.zeros_like(dphidy), grid, margin=1)

        tolerance = 1e-6 if grid_type == 'quad' else 1e-4  # Triangles have small numerical errors
        assert error_x < tolerance, f"Gradient of constant in x: {error_x:.2e} != 0"
        assert error_y < tolerance, f"Gradient of constant in y: {error_y:.2e} != 0"


# ============================================================================
# Laplacian Tests
# ============================================================================

class TestLaplacianParametrized:
    """Parametrized tests for Laplacian on quad and triangle grids."""

    def test_laplacian_gaussian(self, create_grid):
        """Test Laplacian of Gaussian field."""
        grid, grid_type = create_grid(nx=64, ny=64)

        if grid_type == 'quad':
            # Structured grid
            x = np.linspace(0, 1, grid.nx)
            y = np.linspace(0, 1, grid.ny)
            X, Y = np.meshgrid(x, y)

            phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
            lapl_exact = analytical_laplacian_gaussian(
                jnp.array(X), jnp.array(Y), sigma=0.15
            )

        else:  # tri
            # Unstructured grid
            phi = create_gaussian_field_unstructured(grid, sigma=0.15)
            lapl_exact = analytical_laplacian_gaussian_unstructured(grid, sigma=0.15)

        # Numerical Laplacian
        lapl = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        # Check error
        error = compute_error(lapl, lapl_exact, grid, margin=3)

        # Triangles significantly less accurate (FV method + unstructured)
        tolerance = 10.0 if grid_type == 'tri' else 5.0
        assert error < tolerance, f"Laplacian error {error:.2e} too large for {grid_type}-grid"

    def test_laplacian_constant(self, create_grid):
        """Test Laplacian of constant field (should be zero)."""
        grid, grid_type = create_grid()

        if grid_type == 'quad':
            phi = jnp.ones((grid.ny, grid.nx)) * 42.0
        else:  # tri
            phi = jnp.ones(grid.n_cells) * 42.0

        lapl = laplacian(phi, grid, perform_halo_exchange=False)

        error = compute_error(lapl, jnp.zeros_like(lapl), grid, margin=1)

        tolerance = 1e-6 if grid_type == 'quad' else 1e-3
        assert error < tolerance, f"Laplacian of constant: {error:.2e} != 0 for {grid_type}-grid"


# ============================================================================
# Grid Comparison Tests
# ============================================================================

class TestGridComparison:
    """Tests comparing results between quad and triangle grids."""

    def test_divergence_convergence(self):
        """Test that both grid types converge with refinement."""
        # This test compares convergence behavior
        # Skip for now as it requires multiple grid sizes
        pytest.skip("Convergence test requires multiple grid sizes")

    def test_same_problem_both_grids(self):
        """Test same problem on both quad and triangle grids."""
        # Create both grid types
        nx, ny = 32, 32

        # Quad grid
        quad_grid = QuadGrid(name="quad", nx=nx, ny=ny, stagger_type=StaggerType.A, periodic=(False, False))
        quad_grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
        quad_grid.dx = 1.0 / (nx - 1)
        quad_grid.dy = 1.0 / (ny - 1)

        # Triangle grid
        mesh_data = create_structured_triangular_mesh(nx, ny)
        tri_grid = TriGrid(name="tri", stagger_type=StaggerType.A)
        tri_grid.create_mesh(vertices=mesh_data['vertices'], triangles=mesh_data['triangles'])
        tri_grid.compute_neighbors(halo_width=1)

        # Create Gaussian field on both
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)
        phi_quad = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
        phi_tri = create_gaussian_field_unstructured(tri_grid, sigma=0.15)

        # Compute gradient on both
        dpdx_quad, dpdy_quad = gradient(phi_quad, quad_grid, perform_halo_exchange=False)
        dpdx_tri, dpdy_tri = gradient(phi_tri, tri_grid, perform_halo_exchange=False)

        # Both should have reasonable magnitudes (not testing exact equality)
        assert abs(float(dpdx_quad.max())) > 0.1, "Quad gradient too small"
        assert abs(float(dpdx_tri.max())) > 0.1, "Triangle gradient too small"

        # Both should be similar order of magnitude
        ratio = abs(float(dpdx_quad.max()) / float(dpdx_tri.max()))
        assert 0.1 < ratio < 10.0, f"Gradients differ by too much: ratio = {ratio}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
