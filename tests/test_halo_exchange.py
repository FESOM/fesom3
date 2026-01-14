"""Tests for halo exchange operations."""

import pytest
import numpy as np

from fesomx.core.halo_patterns import (
    HaloWidth,
    HaloPattern,
    get_halo_indices_2d,
    get_interior_indices_2d,
    compute_halo_buffer_size,
)
from fesomx.core.halo_exchange import BaseHaloExchanger, StructuredHaloExchanger, exchange_halo


class TestHaloWidth:
    """Tests for HaloWidth configuration."""

    def test_uniform_width(self):
        """Test uniform halo width."""
        hw = HaloWidth(2)
        assert hw.uniform is True
        assert hw.width == 2
        assert hw.width_x == 2
        assert hw.width_y == 2

    def test_directional_width_tuple(self):
        """Test directional width from tuple."""
        hw = HaloWidth((2, 3))
        assert hw.uniform is False
        assert hw.width_x == 2
        assert hw.width_y == 3
        assert hw.width == 3

    def test_directional_width_dict(self):
        """Test directional width from dict."""
        hw = HaloWidth({'left': 1, 'right': 2, 'top': 3, 'bottom': 1})
        assert hw.uniform is False
        assert hw.get_direction_width('left') == 1
        assert hw.get_direction_width('right') == 2
        assert hw.get_direction_width('top') == 3


class TestHaloIndices:
    """Tests for halo index computation."""

    def test_halo_indices_2d(self):
        """Test computation of halo region indices."""
        shape = (10, 10)
        hw = HaloWidth(2)

        # Test left halo
        slice_y, slice_x = get_halo_indices_2d(shape, hw, 'left')
        assert slice_x == slice(0, 2)

        # Test right halo
        slice_y, slice_x = get_halo_indices_2d(shape, hw, 'right')
        assert slice_x == slice(8, 10)

        # Test bottom halo
        slice_y, slice_x = get_halo_indices_2d(shape, hw, 'bottom')
        assert slice_y == slice(0, 2)

        # Test top halo
        slice_y, slice_x = get_halo_indices_2d(shape, hw, 'top')
        assert slice_y == slice(8, 10)

    def test_interior_indices(self):
        """Test computation of interior region indices."""
        shape = (10, 10)
        hw = HaloWidth(2)

        slice_y, slice_x = get_interior_indices_2d(shape, hw)
        assert slice_y == slice(2, 8)
        assert slice_x == slice(2, 8)


class TestHaloBufferSize:
    """Tests for halo buffer size computation."""

    def test_buffer_size_horizontal(self):
        """Test buffer size for horizontal halos."""
        shape = (10, 20)
        hw = HaloWidth(2)

        size_left = compute_halo_buffer_size(shape, hw, 'left')
        size_right = compute_halo_buffer_size(shape, hw, 'right')

        assert size_left == 10 * 2  # ny * width
        assert size_right == 10 * 2

    def test_buffer_size_vertical(self):
        """Test buffer size for vertical halos."""
        shape = (10, 20)
        hw = HaloWidth(2)

        size_top = compute_halo_buffer_size(shape, hw, 'top')
        size_bottom = compute_halo_buffer_size(shape, hw, 'bottom')

        assert size_top == 20 * 2  # nx * width
        assert size_bottom == 20 * 2


class TestHaloExchange:
    """Tests for halo exchange operations."""

    @pytest.mark.cpu
    def test_periodic_exchange_numpy(self):
        """Test periodic halo exchange with numpy arrays."""
        # Create a simple array
        data = np.arange(100).reshape(10, 10)

        # For now, just test that the function runs
        # Full implementation will test correctness
        hw = HaloWidth(1)

        # This is a placeholder test
        assert data.shape == (10, 10)

    @pytest.mark.gpu
    def test_jax_halo_exchange(self, jax_backend, small_quad_grid):
        """Test halo exchange with JAX backend."""
        # Create test field
        field_data = np.random.rand(4, 4)
        small_quad_grid.add_field("test", field_data)

        # Perform halo exchange
        small_quad_grid.halo_exchange("test", halo_width=1)

        # Field should still exist
        assert "test" in small_quad_grid.fields

    def test_periodic_boundaries(self, periodic_quad_grid):
        """Test halo exchange with periodic boundaries."""
        # Create a field with known pattern
        nx, ny = periodic_quad_grid.nx, periodic_quad_grid.ny
        field_data = np.arange(nx * ny).reshape(ny, nx).astype(float)

        periodic_quad_grid.add_field("periodic_test", field_data)

        # Perform halo exchange (should handle periodic boundaries)
        periodic_quad_grid.halo_exchange("periodic_test", halo_width=1)

        # Verify field still has correct shape
        result = periodic_quad_grid.get_field_numpy("periodic_test")
        assert result.shape == field_data.shape


class TestHaloPatterns:
    """Tests for different halo patterns."""

    def test_star_pattern(self):
        """Test 4-point star halo pattern."""
        from fesomx.core.halo_patterns import get_neighbor_directions

        dirs = get_neighbor_directions(HaloPattern.STAR)
        assert len(dirs) == 4
        assert 'left' in dirs
        assert 'right' in dirs
        assert 'top' in dirs
        assert 'bottom' in dirs

    def test_box_pattern(self):
        """Test 8-point box halo pattern."""
        from fesomx.core.halo_patterns import get_neighbor_directions

        dirs = get_neighbor_directions(HaloPattern.BOX)
        assert len(dirs) == 8
        assert 'top-left' in dirs
        assert 'top-right' in dirs


@pytest.mark.parametrize("halo_width", [1, 2, 3])
def test_different_halo_widths(small_quad_grid, halo_width):
    """Test halo exchange with different widths."""
    field_data = np.random.rand(4, 4)
    small_quad_grid.add_field("test", field_data)

    # Should not raise error
    small_quad_grid.halo_exchange("test", halo_width=halo_width)


@pytest.mark.slow
def test_large_grid_halo_exchange(medium_quad_grid):
    """Test halo exchange on larger grid."""
    field_data = np.random.rand(10, 10)
    medium_quad_grid.add_field("large_test", field_data)

    medium_quad_grid.halo_exchange("large_test", halo_width=2)

    # Verify result
    result = medium_quad_grid.get_field_numpy("large_test")
    assert result.shape == field_data.shape
