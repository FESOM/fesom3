"""Tests for grid classes."""

import pytest
import numpy as np

from fesomx import QuadGrid, TriGrid, MixedGrid, StaggerType
from fesomx.core.staggering import StaggerLocation


class TestQuadGrid:
    """Tests for QuadGrid class."""

    def test_grid_creation(self, small_quad_grid):
        """Test basic grid creation."""
        assert small_quad_grid.nx == 4
        assert small_quad_grid.ny == 4
        assert small_quad_grid.n_cells == 16
        assert small_quad_grid.n_vertices == 25

    def test_cell_centers(self, small_quad_grid):
        """Test cell center computation."""
        centers = small_quad_grid.get_cell_centers()
        assert centers.shape == (16, 2)

        # Check that centers are within domain
        assert np.all(centers[:, 0] >= 0) and np.all(centers[:, 0] <= 1)
        assert np.all(centers[:, 1] >= 0) and np.all(centers[:, 1] <= 1)

    def test_cell_volumes(self, small_quad_grid):
        """Test cell area computation."""
        areas = small_quad_grid.get_cell_volumes()
        assert areas.shape == (16,)
        assert np.all(areas > 0)

        # For uniform grid, all cells should have same area
        expected_area = (1.0 / 4) * (1.0 / 4)
        np.testing.assert_allclose(areas, expected_area, rtol=1e-10)

    def test_logical_indices(self, small_quad_grid):
        """Test conversion between linear and logical indices."""
        # Test a few conversions
        assert small_quad_grid.get_linear_index(0, 0) == 0
        assert small_quad_grid.get_linear_index(3, 0) == 3
        assert small_quad_grid.get_linear_index(0, 3) == 12
        assert small_quad_grid.get_linear_index(3, 3) == 15

        # Test round-trip conversion
        for linear_idx in range(16):
            i, j = small_quad_grid.get_logical_indices(linear_idx)
            assert small_quad_grid.get_linear_index(i, j) == linear_idx

    def test_field_operations(self, small_quad_grid):
        """Test adding and retrieving fields."""
        # Create test field
        field_data = np.random.rand(4, 4)

        small_quad_grid.add_field("test_field", field_data)
        assert "test_field" in small_quad_grid.fields

        # Retrieve field
        retrieved = small_quad_grid.get_field_numpy("test_field")
        np.testing.assert_array_equal(retrieved, field_data)

    def test_stagger_types(self):
        """Test different stagger types."""
        for stagger in [StaggerType.A, StaggerType.B, StaggerType.C]:
            grid = QuadGrid("test", nx=4, ny=4, stagger_type=stagger)
            assert grid.stagger_type == stagger


class TestTriGrid:
    """Tests for TriGrid class."""

    def test_grid_creation(self, small_tri_grid):
        """Test triangular grid creation."""
        assert small_tri_grid.n_cells > 0
        assert small_tri_grid.n_vertices > 0
        assert small_tri_grid.cells.shape[1] == 3  # Triangles have 3 vertices

    def test_cell_centers(self, small_tri_grid):
        """Test triangle centroid computation."""
        centers = small_tri_grid.get_cell_centers()
        assert centers.shape[0] == small_tri_grid.n_cells
        assert centers.shape[1] == 2

    def test_cell_volumes(self, small_tri_grid):
        """Test triangle area computation."""
        areas = small_tri_grid.get_cell_volumes()
        assert areas.shape == (small_tri_grid.n_cells,)
        assert np.all(areas > 0)

    def test_neighbor_connectivity(self, small_tri_grid):
        """Test neighbor finding for triangular grid."""
        assert small_tri_grid.cell_neighbors is not None
        # Each interior triangle should have 3 neighbors
        # Boundary triangles have fewer


class TestMixedGrid:
    """Tests for MixedGrid class."""

    def test_grid_creation(self, small_mixed_grid):
        """Test mixed grid creation."""
        assert small_mixed_grid.n_cells > 0
        assert small_mixed_grid.n_vertices > 0
        assert small_mixed_grid.cell_types is not None

        # Check cell types are valid
        assert np.all(np.isin(small_mixed_grid.cell_types, [3, 4]))

    def test_cell_type_queries(self, small_mixed_grid):
        """Test querying different cell types."""
        tri_cells = small_mixed_grid.get_triangular_cells()
        quad_cells = small_mixed_grid.get_quadrilateral_cells()

        # Should have both types
        assert len(tri_cells) > 0
        assert len(quad_cells) > 0

        # Total should match
        assert len(tri_cells) + len(quad_cells) == small_mixed_grid.n_cells

    def test_cell_volumes_mixed(self, small_mixed_grid):
        """Test area computation for mixed cells."""
        areas = small_mixed_grid.get_cell_volumes()
        assert areas.shape == (small_mixed_grid.n_cells,)
        assert np.all(areas > 0)


class TestPeriodicBoundaries:
    """Tests for periodic boundary conditions."""

    def test_periodic_grid_creation(self, periodic_quad_grid):
        """Test grid with periodic boundaries."""
        assert periodic_quad_grid.periodic == (True, True)

    def test_periodic_neighbors(self, periodic_quad_grid):
        """Test neighbor connectivity with periodic boundaries."""
        # With periodic boundaries, edge cells should wrap around
        neighbors = periodic_quad_grid.neighbors
        assert 'left' in neighbors
        assert 'right' in neighbors


@pytest.mark.parametrize("nx,ny", [(4, 4), (8, 8), (5, 7)])
def test_different_grid_sizes(nx, ny):
    """Test grids with different sizes."""
    grid = QuadGrid("test", nx=nx, ny=ny)
    grid.create_mesh()

    assert grid.n_cells == nx * ny
    assert grid.n_vertices == (nx + 1) * (ny + 1)
