"""Tests for parallel backends."""

import pytest
import numpy as np

from fesomx.core.backend import Backend, get_backend, register_backend
from fesomx import JAXShardingBackend


class TestBackendRegistry:
    """Tests for backend registration and retrieval."""

    def test_get_jax_backend(self):
        """Test getting JAX sharding backend."""
        try:
            backend = get_backend("jax_sharding")
            assert isinstance(backend, JAXShardingBackend)
            assert backend.name == "jax_sharding"
        except ImportError:
            pytest.skip("JAX not available")

    def test_invalid_backend(self):
        """Test error for invalid backend name."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent_backend")


class TestJAXBackend:
    """Tests for JAX sharding backend."""

    @pytest.fixture
    def jax_backend(self):
        """Get JAX backend fixture."""
        try:
            backend = get_backend("jax_sharding")
            return backend
        except ImportError:
            pytest.skip("JAX not available")

    def test_backend_initialization(self, jax_backend):
        """Test backend initialization."""
        assert jax_backend._initialized is True
        assert jax_backend.size > 0

    def test_array_creation(self, jax_backend):
        """Test creating backend arrays."""
        data = np.random.rand(10, 10)
        backend_array = jax_backend.array(data)

        assert backend_array is not None
        assert backend_array.shape == data.shape

    def test_to_numpy_conversion(self, jax_backend):
        """Test converting backend array to numpy."""
        data = np.random.rand(10, 10)
        backend_array = jax_backend.array(data)
        result = jax_backend.to_numpy(backend_array)

        np.testing.assert_array_almost_equal(result, data)

    def test_array_shapes(self, jax_backend):
        """Test getting array shapes."""
        data = np.random.rand(5, 8)
        backend_array = jax_backend.array(data)

        global_shape = jax_backend.get_global_shape(backend_array)
        local_shape = jax_backend.get_local_shape(backend_array)

        assert global_shape == (5, 8)
        # Local shape depends on sharding, but should be valid
        assert len(local_shape) == 2

    def test_barrier(self, jax_backend):
        """Test backend barrier synchronization."""
        # Should not raise error
        jax_backend.barrier()

    @pytest.mark.gpu
    def test_gpu_devices(self, jax_backend):
        """Test detection of GPU devices."""
        # This test only runs if GPU marker is active
        assert jax_backend.devices is not None
        # May or may not have GPU depending on system


class TestBackendSwitching:
    """Tests for switching between backends."""

    def test_backend_persistence(self):
        """Test that getting same backend returns same instance."""
        try:
            backend1 = get_backend("jax_sharding")
            backend2 = get_backend("jax_sharding")

            assert backend1 is backend2
        except ImportError:
            pytest.skip("JAX not available")

    def test_grid_with_backend(self, small_quad_grid, jax_backend):
        """Test using grid with specific backend."""
        assert small_quad_grid.backend is not None
        assert small_quad_grid.backend.name == "jax_sharding"


class TestBackendOperations:
    """Tests for backend-specific operations."""

    def test_array_operations(self, jax_backend):
        """Test basic array operations with backend."""
        # Create arrays
        a = np.random.rand(10, 10)
        b = np.random.rand(10, 10)

        a_backend = jax_backend.array(a)
        b_backend = jax_backend.array(b)

        # Convert back
        a_result = jax_backend.to_numpy(a_backend)
        b_result = jax_backend.to_numpy(b_backend)

        np.testing.assert_array_almost_equal(a_result, a)
        np.testing.assert_array_almost_equal(b_result, b)

    @pytest.mark.slow
    def test_large_array(self, jax_backend):
        """Test backend with larger arrays."""
        data = np.random.rand(100, 100)
        backend_array = jax_backend.array(data)
        result = jax_backend.to_numpy(backend_array)

        assert result.shape == data.shape


@pytest.mark.parametrize("shape", [(10, 10), (20, 30), (5, 15)])
def test_various_shapes(jax_backend, shape):
    """Test backend with various array shapes."""
    data = np.random.rand(*shape)
    backend_array = jax_backend.array(data)

    assert jax_backend.get_global_shape(backend_array) == shape


class TestBackendProperties:
    """Tests for backend properties."""

    def test_rank_and_size(self, jax_backend):
        """Test backend rank and size properties."""
        rank = jax_backend.rank
        size = jax_backend.size

        assert rank >= 0
        assert size > 0
        assert rank < size


# Placeholder for future MPI backend tests
@pytest.mark.skip(reason="MPI backend not yet implemented")
class TestMPIBackend:
    """Tests for MPI backends (future implementation)."""

    def test_jax_mpi_backend(self):
        """Test JAX MPI backend."""
        pass

    def test_numpy_mpi_backend(self):
        """Test NumPy MPI backend."""
        pass
