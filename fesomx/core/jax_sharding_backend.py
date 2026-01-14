"""JAX sharding backend implementation."""

from typing import Any, Tuple, Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding, Mesh
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from .backend import Backend, register_backend


class JAXShardingBackend(Backend):
    """JAX backend using sharding for parallelism.

    Uses JAX's built-in sharding capabilities for data parallelism
    across multiple devices (GPUs or CPUs).
    """

    def __init__(self):
        if not JAX_AVAILABLE:
            raise ImportError("JAX is not available. Install with: pip install jax jaxlib")
        super().__init__("jax_sharding")
        self.devices = None
        self.mesh = None

    def initialize(self, mesh_shape: Optional[Tuple[int, ...]] = None, **kwargs) -> None:
        """Initialize JAX backend with device mesh.

        Args:
            mesh_shape: Shape of device mesh, e.g., (2, 2) for 2x2 grid.
                       If None, uses all available devices in 1D layout.
        """
        self.devices = jax.devices()
        n_devices = len(self.devices)

        if mesh_shape is None:
            # Default to 1D mesh with all devices
            mesh_shape = (n_devices,)
            axis_names = ("x",)
        elif len(mesh_shape) == 2:
            axis_names = ("x", "y")
        elif len(mesh_shape) == 1:
            axis_names = ("x",)
        else:
            raise ValueError(f"Unsupported mesh shape: {mesh_shape}")

        # Verify mesh shape is compatible with device count
        mesh_size = np.prod(mesh_shape)
        if mesh_size != n_devices:
            raise ValueError(
                f"Mesh shape {mesh_shape} requires {mesh_size} devices, "
                f"but only {n_devices} available"
            )

        self.mesh = Mesh(np.array(self.devices).reshape(mesh_shape), axis_names)
        self._initialized = True

        print(f"JAX Sharding Backend initialized:")
        print(f"  Devices: {n_devices} ({[d.device_kind for d in self.devices]})")
        print(f"  Mesh shape: {mesh_shape}")
        print(f"  Mesh axes: {axis_names}")

    def array(self, data: np.ndarray, sharding_spec: Optional[Any] = None) -> jax.Array:
        """Create JAX array with optional sharding.

        Args:
            data: Input numpy array
            sharding_spec: PartitionSpec for sharding, e.g., P('x', None) for
                          sharding along first dimension across 'x' axis

        Returns:
            JAX array (potentially sharded across devices)
        """
        if not self._initialized:
            self.initialize()

        if sharding_spec is None:
            # No sharding, replicate on all devices
            return jnp.array(data)

        # Create sharded array
        sharding = NamedSharding(self.mesh, sharding_spec)
        return jax.device_put(data, sharding)

    def to_numpy(self, array: jax.Array) -> np.ndarray:
        """Convert JAX array to numpy array."""
        return np.array(array)

    def halo_exchange(
        self,
        array: jax.Array,
        halo_width: int,
        neighbors: dict,
        periodic: Tuple[bool, ...] = (False, False)
    ) -> jax.Array:
        """Perform halo exchange for sharded array.

        Args:
            array: Sharded JAX array
            halo_width: Width of halo region
            neighbors: Dictionary mapping direction to neighbor indices
            periodic: Periodic boundary conditions

        Returns:
            Array with updated halos

        Note:
            This is a placeholder implementation. Full halo exchange logic
            will be implemented in the halo_exchange module.
        """
        # For now, return array unchanged
        # Full implementation will use JAX's collective operations
        # (jax.lax.ppermute, psum, etc.) for inter-device communication
        return array

    def get_local_shape(self, array: jax.Array) -> Tuple[int, ...]:
        """Get shape of local shard."""
        # For JAX sharding, local shape is the addressable shard shape
        sharding = array.sharding
        if hasattr(sharding, 'shard_shape'):
            return sharding.shard_shape(array.shape)
        return array.shape

    def get_global_shape(self, array: jax.Array) -> Tuple[int, ...]:
        """Get global shape of array."""
        return array.shape

    def barrier(self) -> None:
        """Synchronize all devices."""
        # JAX operations are synchronous by default
        # Explicit barrier via dummy computation
        dummy = jnp.ones(1)
        for device in self.devices:
            jax.device_put(dummy, device).block_until_ready()

    @property
    def rank(self) -> int:
        """Get current device index (process ID equivalent)."""
        if jax.process_count() > 1:
            return jax.process_index()
        return 0

    @property
    def size(self) -> int:
        """Get total number of devices."""
        if self.devices is None:
            return len(jax.devices())
        return len(self.devices)


# Register this backend
if JAX_AVAILABLE:
    register_backend("jax_sharding", JAXShardingBackend)
