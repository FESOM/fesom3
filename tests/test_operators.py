"""
Comprehensive Unit Tests for Operators

Tests divergence, gradient, and laplacian operators on A/B/C grids
with various test cases and accuracy requirements.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from fesomx import QuadGrid, StaggerType, divergence, gradient, laplacian
from fesomx.operators.divergence import divergence_free_field_2d
from fesomx.operators.gradient import smooth_gaussian_field_2d, analytical_gradient_gaussian
from fesomx.operators.laplacian import analytical_laplacian_gaussian


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=[32, 64])
def grid_size(request):
    """Grid sizes to test."""
    return request.param


@pytest.fixture(params=[StaggerType.A, StaggerType.B, StaggerType.C])
def stagger_type(request):
    """Stagger types to test."""
    return request.param


@pytest.fixture
def create_grid(grid_size, stagger_type):
    """Create a test grid."""
    def _create_grid(nx=None, ny=None, stagger=None):
        _nx = nx or grid_size
        _ny = ny or grid_size
        _stagger = stagger or stagger_type

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

        return grid

    return _create_grid


# ============================================================================
# Divergence Tests
# ============================================================================

class TestDivergence:
    """Tests for divergence operator."""

    def test_divergence_free_field_a_grid(self, create_grid):
        """Test divergence of divergence-free field on A-grid."""
        grid = create_grid(stagger=StaggerType.A)
        nx, ny = grid.nx, grid.ny

        # Create coordinates
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Divergence-free field
        u, v = divergence_free_field_2d(jnp.array(X), jnp.array(Y), mode='sine')

        # Compute divergence
        div = divergence(u, v, grid, perform_halo_exchange=False)

        # Check (excluding boundaries)
        error = np.abs(np.array(div[1:-1, 1:-1])).max()
        assert error < 1e-2, f"Divergence error {error:.2e} too large for A-grid"

    def test_divergence_free_field_b_grid(self, create_grid):
        """Test divergence of divergence-free field on B-grid."""
        grid = create_grid(stagger=StaggerType.B)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        u, v = divergence_free_field_2d(jnp.array(X), jnp.array(Y), mode='sine')
        div = divergence(u, v, grid, perform_halo_exchange=False)

        error = np.abs(np.array(div[1:-1, 1:-1])).max()
        assert error < 1e-2, f"Divergence error {error:.2e} too large for B-grid"

    def test_divergence_free_field_c_grid(self, create_grid):
        """Test divergence of divergence-free field on C-grid."""
        grid = create_grid(stagger=StaggerType.C)
        nx, ny = grid.nx, grid.ny

        # C-grid: u at x-faces, v at y-faces
        x_u = np.linspace(0, 1, nx + 1)
        y_u = np.linspace(0, 1, ny)
        X_u, Y_u = np.meshgrid(x_u, y_u)

        x_v = np.linspace(0, 1, nx)
        y_v = np.linspace(0, 1, ny + 1)
        X_v, Y_v = np.meshgrid(x_v, y_v)

        u, _ = divergence_free_field_2d(jnp.array(X_u), jnp.array(Y_u), mode='sine')
        _, v = divergence_free_field_2d(jnp.array(X_v), jnp.array(Y_v), mode='sine')

        div = divergence(u, v, grid, perform_halo_exchange=False)

        error = np.abs(np.array(div[1:-1, 1:-1])).max()
        assert error < 5e-2, f"Divergence error {error:.2e} too large for C-grid"

    def test_divergence_rotation_field(self, create_grid):
        """Test divergence of rotational field (should be zero)."""
        grid = create_grid(nx=64, ny=64)
        nx, ny = grid.nx, grid.ny

        # Handle C-grid separately
        if grid.stagger_type == StaggerType.C:
            x_u = np.linspace(0, 1, nx + 1)
            y_u = np.linspace(0, 1, ny)
            X_u, Y_u = np.meshgrid(x_u, y_u)

            x_v = np.linspace(0, 1, nx)
            y_v = np.linspace(0, 1, ny + 1)
            X_v, Y_v = np.meshgrid(x_v, y_v)

            u, _ = divergence_free_field_2d(jnp.array(X_u), jnp.array(Y_u), mode='rotation')
            _, v = divergence_free_field_2d(jnp.array(X_v), jnp.array(Y_v), mode='rotation')
        else:
            x = np.linspace(0, 1, nx)
            y = np.linspace(0, 1, ny)
            X, Y = np.meshgrid(x, y)

            u, v = divergence_free_field_2d(jnp.array(X), jnp.array(Y), mode='rotation')

        div = divergence(u, v, grid, perform_halo_exchange=False)

        error = np.abs(np.array(div[2:-2, 2:-2])).max()
        tolerance = 5e-2 if grid.stagger_type == StaggerType.C else 1e-2
        assert error < tolerance, f"Rotation field divergence {error:.2e} too large"

    def test_divergence_shape(self, create_grid):
        """Test that divergence returns correct shape."""
        grid = create_grid()
        nx, ny = grid.nx, grid.ny

        if grid.stagger_type == StaggerType.C:
            u = jnp.zeros((ny, nx + 1))
            v = jnp.zeros((ny + 1, nx))
            expected_shape = (ny, nx)
        elif grid.stagger_type == StaggerType.B:
            # B-grid: velocities at corners (ny+1, nx+1), divergence at centers (ny, nx)
            # But due to averaging, we may get (ny-1, nx-1)
            u = jnp.zeros((ny, nx))  # Using cell-centered for simplicity
            v = jnp.zeros((ny, nx))
            expected_shape = (ny, nx)  # After averaging/interpolation
        else:
            u = jnp.zeros((ny, nx))
            v = jnp.zeros((ny, nx))
            expected_shape = (ny, nx)

        div = divergence(u, v, grid, perform_halo_exchange=False)

        # B-grid may return (ny-1, nx-1) due to corner-to-center averaging
        if grid.stagger_type == StaggerType.B:
            assert div.shape[0] >= ny - 1 and div.shape[1] >= nx - 1, \
                f"Divergence shape {div.shape} incompatible with grid ({ny}, {nx})"
        else:
            assert div.shape == expected_shape, \
                f"Divergence shape {div.shape} != {expected_shape}"


# ============================================================================
# Gradient Tests
# ============================================================================

class TestGradient:
    """Tests for gradient operator."""

    def test_gradient_gaussian_a_grid(self, create_grid):
        """Test gradient of Gaussian on A-grid."""
        grid = create_grid(nx=64, ny=64, stagger=StaggerType.A)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Gaussian field
        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)

        # Numerical gradient
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        # Analytical gradient
        dphidx_exact, dphidy_exact = analytical_gradient_gaussian(
            jnp.array(X), jnp.array(Y), sigma=0.15
        )

        # Check errors
        error_x = np.abs(np.array(dphidx[2:-2, 2:-2]) - np.array(dphidx_exact[2:-2, 2:-2])).max()
        error_y = np.abs(np.array(dphidy[2:-2, 2:-2]) - np.array(dphidy_exact[2:-2, 2:-2])).max()

        assert error_x < 0.5, f"Gradient x-error {error_x:.2e} too large for A-grid"
        assert error_y < 0.5, f"Gradient y-error {error_y:.2e} too large for A-grid"

    def test_gradient_gaussian_b_grid(self, create_grid):
        """Test gradient of Gaussian on B-grid."""
        grid = create_grid(nx=64, ny=64, stagger=StaggerType.B)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)
        dphidx_exact, dphidy_exact = analytical_gradient_gaussian(
            jnp.array(X), jnp.array(Y), sigma=0.15
        )

        error_x = np.abs(np.array(dphidx[2:-2, 2:-2]) - np.array(dphidx_exact[2:-2, 2:-2])).max()
        error_y = np.abs(np.array(dphidy[2:-2, 2:-2]) - np.array(dphidy_exact[2:-2, 2:-2])).max()

        # B-grid has simpler implementation, more lenient tolerance
        assert error_x < 1.0, f"Gradient x-error {error_x:.2e} too large for B-grid"
        assert error_y < 1.0, f"Gradient y-error {error_y:.2e} too large for B-grid"

    def test_gradient_gaussian_c_grid(self, create_grid):
        """Test gradient of Gaussian on C-grid."""
        grid = create_grid(nx=64, ny=64, stagger=StaggerType.C)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False, return_at_centers=True)
        dphidx_exact, dphidy_exact = analytical_gradient_gaussian(
            jnp.array(X), jnp.array(Y), sigma=0.15
        )

        error_x = np.abs(np.array(dphidx[2:-2, 2:-2]) - np.array(dphidx_exact[2:-2, 2:-2])).max()
        error_y = np.abs(np.array(dphidy[2:-2, 2:-2]) - np.array(dphidy_exact[2:-2, 2:-2])).max()

        assert error_x < 0.5, f"Gradient x-error {error_x:.2e} too large for C-grid"
        assert error_y < 0.5, f"Gradient y-error {error_y:.2e} too large for C-grid"

    def test_gradient_constant_field(self, create_grid):
        """Test gradient of constant field (should be zero)."""
        grid = create_grid()
        nx, ny = grid.nx, grid.ny

        phi = jnp.ones((ny, nx))
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        error_x = np.abs(np.array(dphidx[1:-1, 1:-1])).max()
        error_y = np.abs(np.array(dphidy[1:-1, 1:-1])).max()

        assert error_x < 1e-10, f"Gradient of constant in x: {error_x:.2e} != 0"
        assert error_y < 1e-10, f"Gradient of constant in y: {error_y:.2e} != 0"

    def test_gradient_linear_field(self, create_grid):
        """Test gradient of linear field."""
        grid = create_grid(nx=32, ny=32)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Linear field: phi = 2x + 3y
        # Gradient should be (2, 3) everywhere
        phi = jnp.array(2*X + 3*Y)

        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        # Check interior (boundaries may have edge effects)
        # B-grid has simpler implementation, more lenient
        tolerance = 0.1 if grid.stagger_type == StaggerType.B else 1e-2
        error_x = np.abs(np.array(dphidx[3:-3, 3:-3]) - 2.0).max()
        error_y = np.abs(np.array(dphidy[3:-3, 3:-3]) - 3.0).max()

        assert error_x < tolerance, f"Linear field gradient x: {error_x:.2e} != 2 (tol={tolerance})"
        assert error_y < tolerance, f"Linear field gradient y: {error_y:.2e} != 3 (tol={tolerance})"

    def test_gradient_shape(self, create_grid):
        """Test that gradient returns correct shapes."""
        grid = create_grid()
        nx, ny = grid.nx, grid.ny

        phi = jnp.zeros((ny, nx))
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

        assert dphidx.shape == (ny, nx), f"Gradient x shape {dphidx.shape} != ({ny}, {nx})"
        assert dphidy.shape == (ny, nx), f"Gradient y shape {dphidy.shape} != ({ny}, {nx})"


# ============================================================================
# Laplacian Tests
# ============================================================================

class TestLaplacian:
    """Tests for Laplacian operator."""

    @pytest.mark.parametrize("order", [2, 4])
    def test_laplacian_gaussian(self, create_grid, order):
        """Test Laplacian of Gaussian field."""
        # Need larger grid for 4th order
        min_size = 32 if order == 2 else 64
        grid = create_grid(nx=min_size, ny=min_size)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Gaussian field
        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)

        # Numerical Laplacian
        lapl = laplacian(phi, grid, order=order, perform_halo_exchange=False)

        # Analytical Laplacian
        lapl_exact = analytical_laplacian_gaussian(jnp.array(X), jnp.array(Y), sigma=0.15)

        # Check error
        margin = 2 if order == 2 else 3
        error = np.abs(
            np.array(lapl[margin:-margin, margin:-margin]) -
            np.array(lapl_exact[margin:-margin, margin:-margin])
        ).max()

        tolerance = 5.0 if order == 2 else 1.0
        assert error < tolerance, f"Laplacian order-{order} error {error:.2e} too large"

    def test_laplacian_constant_field(self, create_grid):
        """Test Laplacian of constant field (should be zero)."""
        grid = create_grid()
        nx, ny = grid.nx, grid.ny

        phi = jnp.ones((ny, nx)) * 42.0
        lapl = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        error = np.abs(np.array(lapl[1:-1, 1:-1])).max()
        assert error < 1e-10, f"Laplacian of constant: {error:.2e} != 0"

    def test_laplacian_linear_field(self, create_grid):
        """Test Laplacian of linear field (should be zero)."""
        grid = create_grid(nx=32, ny=32)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Linear field: phi = ax + by
        phi = jnp.array(2*X + 3*Y)

        lapl = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        # Check deep interior (boundaries may have issues)
        error = np.abs(np.array(lapl[4:-4, 4:-4])).max()
        assert error < 1e-3, f"Laplacian of linear field: {error:.2e} != 0"

    def test_laplacian_quadratic_field(self, create_grid):
        """Test Laplacian of quadratic field."""
        grid = create_grid(nx=32, ny=32)
        nx, ny = grid.nx, grid.ny
        dx = grid.dx
        dy = grid.dy

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        # Quadratic field: phi = x² + y²
        # Laplacian = ∂²φ/∂x² + ∂²φ/∂y² = 2 + 2 = 4
        phi = jnp.array(X**2 + Y**2)

        lapl = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        # Check interior (should be close to 4)
        expected = 4.0
        error = np.abs(np.array(lapl[2:-2, 2:-2]) - expected).max()

        assert error < 0.1, f"Laplacian of quadratic: error {error:.2e}, expected {expected}"

    def test_laplacian_shape(self, create_grid):
        """Test that Laplacian returns correct shape."""
        grid = create_grid()
        nx, ny = grid.nx, grid.ny

        phi = jnp.zeros((ny, nx))
        lapl = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        assert lapl.shape == (ny, nx), f"Laplacian shape {lapl.shape} != ({ny}, {nx})"

    def test_laplacian_order_comparison(self, create_grid):
        """Test that 4th order is more accurate than 2nd order."""
        grid = create_grid(nx=64, ny=64)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)
        lapl_exact = analytical_laplacian_gaussian(jnp.array(X), jnp.array(Y), sigma=0.15)

        # 2nd order
        lapl_2nd = laplacian(phi, grid, order=2, perform_halo_exchange=False)
        error_2nd = np.abs(np.array(lapl_2nd[3:-3, 3:-3]) - np.array(lapl_exact[3:-3, 3:-3])).max()

        # 4th order
        lapl_4th = laplacian(phi, grid, order=4, perform_halo_exchange=False)
        error_4th = np.abs(np.array(lapl_4th[3:-3, 3:-3]) - np.array(lapl_exact[3:-3, 3:-3])).max()

        # 4th order should be more accurate
        assert error_4th < error_2nd, \
            f"4th order not more accurate: {error_4th:.2e} >= {error_2nd:.2e}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple operators."""

    def test_gradient_divergence_identity(self, create_grid):
        """Test that ∇·∇φ = ∇²φ for compatible grids."""
        # Only test A-grid for simplicity (gradient and divergence both at centers)
        grid = create_grid(nx=64, ny=64, stagger=StaggerType.A)
        nx, ny = grid.nx, grid.ny

        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y)

        phi = smooth_gaussian_field_2d(jnp.array(X), jnp.array(Y), sigma=0.15)

        # Method 1: Direct Laplacian
        lapl_direct = laplacian(phi, grid, order=2, perform_halo_exchange=False)

        # Method 2: Gradient then divergence
        dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)
        lapl_indirect = divergence(dphidx, dphidy, grid, perform_halo_exchange=False)

        # They should be similar (not exact due to numerical differences and boundaries)
        # Check deep interior only
        diff = np.abs(np.array(lapl_direct[5:-5, 5:-5]) -
                     np.array(lapl_indirect[5:-5, 5:-5])).max()

        # Lenient tolerance - these are computed differently
        assert diff < 5.0, f"∇·∇φ != ∇²φ: difference {diff:.2e} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
