"""
Quick Validation Script for Operators

Tests divergence, gradient, and laplacian operators on A/B/C grids
using analytical solutions to verify basic functionality.

Run this before creating full examples to catch obvious bugs.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jax.numpy as jnp
from jax import devices

from core import QuadGrid, StaggerType
from operators import divergence, gradient, laplacian
from operators.divergence import divergence_free_field_2d
from operators.gradient import smooth_gaussian_field_2d, analytical_gradient_gaussian
from operators.laplacian import analytical_laplacian_gaussian


def create_test_grid(stagger_type: StaggerType, nx=32, ny=32):
    """Create a simple test grid."""
    grid = QuadGrid(
        name="test_grid",
        nx=nx,
        ny=ny,
        stagger_type=stagger_type,
        periodic=(False, False)
    )

    # Create mesh
    grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))

    # Add dx, dy for convenience (operators need these)
    grid.dx = 1.0 / (nx - 1)
    grid.dy = 1.0 / (ny - 1)

    return grid


def test_divergence(stagger_type: StaggerType):
    """Test divergence operator on divergence-free field."""
    print(f"\n  Testing divergence ({stagger_type.name}-grid)...")

    grid = create_test_grid(stagger_type, nx=32, ny=32)

    # Create coordinate arrays
    if stagger_type == StaggerType.C:
        # C-grid: u at x-faces (ny, nx+1), v at y-faces (ny+1, nx)
        x_u = np.linspace(0, 1, 33)  # x-faces
        y_u = np.linspace(0, 1, 32)  # cell centers in y
        X_u, Y_u = np.meshgrid(x_u, y_u)

        x_v = np.linspace(0, 1, 32)  # cell centers in x
        y_v = np.linspace(0, 1, 33)  # y-faces
        X_v, Y_v = np.meshgrid(x_v, y_v)

        # Generate divergence-free field at face locations
        u, _ = divergence_free_field_2d(jnp.array(X_u), jnp.array(Y_u), mode='sine')
        _, v = divergence_free_field_2d(jnp.array(X_v), jnp.array(Y_v), mode='sine')
    else:
        # A-grid and B-grid: u,v at centers
        x = np.linspace(0, 1, 32)
        y = np.linspace(0, 1, 32)
        X, Y = np.meshgrid(x, y)

        # Generate divergence-free field
        u, v = divergence_free_field_2d(jnp.array(X), jnp.array(Y), mode='sine')

    # Compute divergence (should be ~0)
    div = divergence(u, v, grid, perform_halo_exchange=False)

    # Check error (excluding boundaries)
    error = np.abs(np.array(div[1:-1, 1:-1])).max()

    # Tolerance depends on grid type and spacing
    # C-grid has higher error due to sampling at face locations
    tolerance = 5e-2 if stagger_type == StaggerType.C else 1e-2

    if error < tolerance:
        print(f"    ✓ PASS: max|div| = {error:.2e} < {tolerance:.2e}")
        return True
    else:
        print(f"    ✗ FAIL: max|div| = {error:.2e} >= {tolerance:.2e}")
        return False


def test_gradient(stagger_type: StaggerType):
    """Test gradient operator on Gaussian field."""
    print(f"\n  Testing gradient ({stagger_type.name}-grid)...")

    grid = create_test_grid(stagger_type, nx=32, ny=32)

    # Create coordinate arrays
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)

    # Generate Gaussian field
    phi = smooth_gaussian_field_2d(
        jnp.array(X), jnp.array(Y),
        x0=0.5, y0=0.5, sigma=0.15
    )

    # Compute numerical gradient
    dphidx, dphidy = gradient(phi, grid, perform_halo_exchange=False)

    # Analytical gradient
    dphidx_exact, dphidy_exact = analytical_gradient_gaussian(
        jnp.array(X), jnp.array(Y),
        x0=0.5, y0=0.5, sigma=0.15
    )

    # Check errors (excluding boundaries)
    error_x = np.abs(np.array(dphidx[2:-2, 2:-2]) - np.array(dphidx_exact[2:-2, 2:-2])).max()
    error_y = np.abs(np.array(dphidy[2:-2, 2:-2]) - np.array(dphidy_exact[2:-2, 2:-2])).max()

    # B-grid has simpler gradient implementation, so more lenient tolerance
    tolerance = 1.0 if stagger_type == StaggerType.B else 0.5

    if error_x < tolerance and error_y < tolerance:
        print(f"    ✓ PASS: max|∂φ/∂x error| = {error_x:.2e}, max|∂φ/∂y error| = {error_y:.2e}")
        return True
    else:
        print(f"    ✗ FAIL: max|∂φ/∂x error| = {error_x:.2e}, max|∂φ/∂y error| = {error_y:.2e}")
        return False


def test_laplacian(stagger_type: StaggerType, order=2):
    """Test Laplacian operator on Gaussian field."""
    print(f"\n  Testing Laplacian (order={order}, {stagger_type.name}-grid)...")

    grid = create_test_grid(stagger_type, nx=32, ny=32)

    # Create coordinate arrays
    x = np.linspace(0, 1, 32)
    y = np.linspace(0, 1, 32)
    X, Y = np.meshgrid(x, y)

    # Generate Gaussian field
    phi = smooth_gaussian_field_2d(
        jnp.array(X), jnp.array(Y),
        x0=0.5, y0=0.5, sigma=0.15
    )

    # Compute numerical Laplacian
    lapl = laplacian(phi, grid, order=order, perform_halo_exchange=False)

    # Analytical Laplacian
    lapl_exact = analytical_laplacian_gaussian(
        jnp.array(X), jnp.array(Y),
        x0=0.5, y0=0.5, sigma=0.15
    )

    # Check error (excluding boundaries)
    margin = 2 if order == 2 else 3
    error = np.abs(
        np.array(lapl[margin:-margin, margin:-margin]) -
        np.array(lapl_exact[margin:-margin, margin:-margin])
    ).max()

    # 4th order should be more accurate
    tolerance = 5.0 if order == 2 else 1.0

    if error < tolerance:
        print(f"    ✓ PASS: max|∇²φ error| = {error:.2e} < {tolerance:.2e}")
        return True
    else:
        print(f"    ✗ FAIL: max|∇²φ error| = {error:.2e} >= {tolerance:.2e}")
        return False


def main():
    """Run all quick validation tests."""
    print("=" * 70)
    print("Quick Validation of Operators (A/B/C Grids)")
    print("=" * 70)

    # Check JAX devices
    print(f"\nJAX devices: {devices()}")

    results = []

    # Test each grid type
    for stagger in [StaggerType.A, StaggerType.B, StaggerType.C]:
        print(f"\n{'=' * 70}")
        print(f"{stagger.name}-Grid Tests")
        print('=' * 70)

        # Divergence test
        results.append(('Divergence', stagger.name, test_divergence(stagger)))

        # Gradient test
        results.append(('Gradient', stagger.name, test_gradient(stagger)))

        # Laplacian tests (2nd and 4th order)
        results.append(('Laplacian-2nd', stagger.name, test_laplacian(stagger, order=2)))
        results.append(('Laplacian-4th', stagger.name, test_laplacian(stagger, order=4)))

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print('=' * 70)

    passed = sum(1 for _, _, result in results if result)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    # Detailed results
    print("\nDetailed Results:")
    for op, grid, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {op:20s} ({grid}-grid)")

    if passed == total:
        print("\n✓ All tests passed! Operators are working correctly.")
        print("  Ready to proceed with examples and comprehensive tests.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review operator implementations.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
