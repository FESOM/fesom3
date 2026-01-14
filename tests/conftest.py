"""Pytest configuration and fixtures for fesomx tests."""

import pytest
import numpy as np

from fesomx import QuadGrid, TriGrid, MixedGrid, StaggerType, get_backend
from fesomx.utilities import create_rectangular_mesh, create_structured_triangular_mesh, create_mixed_mesh


@pytest.fixture
def small_quad_grid():
    """Create a small quadrilateral grid for testing."""
    grid = QuadGrid("test_quad", nx=4, ny=4, stagger_type=StaggerType.A)
    grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
    grid.compute_neighbors(halo_width=1)
    return grid


@pytest.fixture
def medium_quad_grid():
    """Create a medium-sized quadrilateral grid for testing."""
    grid = QuadGrid("test_quad_medium", nx=10, ny=10, stagger_type=StaggerType.C)
    grid.create_mesh(domain=(0.0, 10.0, 0.0, 10.0))
    grid.compute_neighbors(halo_width=2)
    return grid


@pytest.fixture
def small_tri_grid():
    """Create a small triangular grid for testing."""
    mesh_data = create_structured_triangular_mesh(nx=3, ny=3)

    grid = TriGrid("test_tri", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        triangles=mesh_data['triangles']
    )
    grid.compute_neighbors(halo_width=1)
    return grid


@pytest.fixture
def small_mixed_grid():
    """Create a small mixed grid for testing."""
    mesh_data = create_mixed_mesh(nx=4, ny=4, tri_fraction=0.3, seed=42)

    grid = MixedGrid("test_mixed", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        cells=mesh_data['cells']
    )
    grid.compute_neighbors(halo_width=1)
    return grid


@pytest.fixture
def jax_backend():
    """Get JAX sharding backend."""
    try:
        backend = get_backend("jax_sharding")
        return backend
    except Exception as e:
        pytest.skip(f"JAX backend not available: {e}")


@pytest.fixture
def sample_field_data():
    """Create sample field data for testing."""
    nx, ny = 10, 10
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Create a simple field (e.g., sin wave)
    field = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    return field


@pytest.fixture
def periodic_quad_grid():
    """Create a quadrilateral grid with periodic boundaries."""
    grid = QuadGrid(
        "test_periodic",
        nx=8,
        ny=8,
        stagger_type=StaggerType.A,
        periodic=(True, True)
    )
    grid.create_mesh(domain=(0.0, 2*np.pi, 0.0, 2*np.pi))
    grid.compute_neighbors(halo_width=1)
    return grid


# Markers for different test categories
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "cpu: marks tests for CPU-only execution")
    config.addinivalue_line("markers", "parallel: marks tests for parallel execution")
