"""Abstract backend interface for parallel computing frameworks."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
import numpy as np


class Backend(ABC):
    """Abstract base class for parallel computing backends.

    This interface allows switching between different parallel frameworks:
    - JAX with sharding
    - JAX with MPI
    - NumPy with MPI
    """

    def __init__(self, name: str):
        self.name = name
        self._initialized = False

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the backend with optional configuration."""
        pass

    @abstractmethod
    def array(self, data: np.ndarray, sharding_spec: Optional[Any] = None) -> Any:
        """Create a backend-specific array from numpy array.

        Args:
            data: Input numpy array
            sharding_spec: Optional sharding/distribution specification

        Returns:
            Backend-specific array (JAX array, distributed array, etc.)
        """
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert backend array to numpy array.

        Args:
            array: Backend-specific array

        Returns:
            Numpy array
        """
        pass

    @abstractmethod
    def halo_exchange(
        self,
        array: Any,
        halo_width: int,
        neighbors: dict,
        periodic: Tuple[bool, ...] = (False, False)
    ) -> Any:
        """Perform halo exchange between neighboring partitions.

        Args:
            array: Distributed array
            halo_width: Width of halo region
            neighbors: Dictionary mapping direction to neighbor rank/index
            periodic: Tuple of bools indicating periodic boundaries

        Returns:
            Array with updated halo regions
        """
        pass

    @abstractmethod
    def get_local_shape(self, array: Any) -> Tuple[int, ...]:
        """Get the shape of the local portion of a distributed array."""
        pass

    @abstractmethod
    def get_global_shape(self, array: Any) -> Tuple[int, ...]:
        """Get the global shape of a distributed array."""
        pass

    @abstractmethod
    def barrier(self) -> None:
        """Synchronize all processes/devices."""
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        """Get the rank/ID of current process/device."""
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """Get total number of processes/devices."""
        pass

    def finalize(self) -> None:
        """Clean up resources (optional, for MPI backends)."""
        pass


# Global backend registry
_BACKENDS = {}
_CURRENT_BACKEND = None


def register_backend(name: str, backend_class: type) -> None:
    """Register a backend implementation."""
    _BACKENDS[name] = backend_class


def get_backend(name: str = "jax_sharding", **kwargs) -> Backend:
    """Get or create a backend instance.

    Args:
        name: Backend name ('jax_sharding', 'jax_mpi', 'numpy_mpi')
        **kwargs: Backend-specific initialization arguments

    Returns:
        Backend instance
    """
    global _CURRENT_BACKEND

    if _CURRENT_BACKEND is not None and _CURRENT_BACKEND.name == name:
        return _CURRENT_BACKEND

    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}. Available: {list(_BACKENDS.keys())}")

    backend = _BACKENDS[name]()  # Create backend without arguments
    backend.initialize(**kwargs)  # Pass arguments to initialize
    _CURRENT_BACKEND = backend
    return backend


def set_backend(backend: Backend) -> None:
    """Set the current global backend."""
    global _CURRENT_BACKEND
    _CURRENT_BACKEND = backend
