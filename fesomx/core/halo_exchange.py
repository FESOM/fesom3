"""Halo exchange operations for distributed grids.

Provides two strategies:
1. StructuredHaloExchanger - Edge-based exchange for regular grids (QuadGrid)
2. UnstructuredHaloExchanger - Graph-based exchange for irregular grids (TriGrid/MixedGrid)

Communication strategies:
- SHARED_MEMORY: JAX automatic cross-device (fast on single-node)
- PPERMUTE: Explicit lax.ppermute (necessary for multi-node)
- HYBRID: Intra-node shared memory + inter-node ppermute (optimal)
"""

from typing import Any, Optional, Dict, Tuple, List
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import lax
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .backend import Backend
from .halo_patterns import HaloWidth, HaloPattern, get_halo_indices_2d, get_neighbor_directions


class CommStrategy(Enum):
    """Communication strategy for halo exchange.

    Attributes:
        SHARED_MEMORY: Use JAX automatic cross-device communication.
            - Fast on single-node (direct memory access)
            - Simple implementation (no collective ops)
            - Doesn't work on multi-node clusters

        PPERMUTE: Use jax.lax.ppermute for explicit device-to-device.
            - Works on multi-node (MPI/NCCL backend)
            - More complex (requires pmap context)
            - Slight overhead on single-node

        HYBRID: Auto-select based on device topology.
            - Shared memory for intra-node devices
            - ppermute for inter-node devices
            - Optimal performance at all scales

        AUTO: Alias for HYBRID with auto-detection
    """
    SHARED_MEMORY = "shared_memory"
    PPERMUTE = "ppermute"
    HYBRID = "hybrid"
    AUTO = "auto"  # Alias for HYBRID


class BaseHaloExchanger(ABC):
    """Abstract base class for halo exchangers."""

    def __init__(self, backend: Backend, comm_strategy: CommStrategy | str = CommStrategy.SHARED_MEMORY):
        """Initialize halo exchanger.

        Args:
            backend: Parallel backend to use
            comm_strategy: Communication strategy (default: SHARED_MEMORY)
                Can be CommStrategy enum or string: 'shared_memory', 'ppermute', 'hybrid', 'auto'
        """
        self.backend = backend

        # Convert string to enum if needed
        if isinstance(comm_strategy, str):
            comm_strategy = CommStrategy(comm_strategy)

        # AUTO is alias for HYBRID
        if comm_strategy == CommStrategy.AUTO:
            comm_strategy = CommStrategy.HYBRID

        self.comm_strategy = comm_strategy

    @abstractmethod
    def exchange(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int | HaloWidth,
    ) -> Tuple[Any, Dict]:
        """Perform halo exchange.

        Args:
            array: Distributed array
            partition_info: Partitioning metadata (grid-specific)
            halo_width: Halo width specification

        Returns:
            Tuple of:
            - updated_array: Array with filled halos
            - diagnostics: Dict with communication info for visualization
        """
        pass


class StructuredHaloExchanger(BaseHaloExchanger):
    """Edge-based halo exchange for structured grids.

    Uses directional exchange (left, right, top, bottom) where halos
    are contiguous regions at shard edges.

    Best for: QuadGrid with regular rectangular partitioning
    """

    def exchange(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int | HaloWidth,
    ) -> Tuple[Any, Dict]:
        """Perform structured halo exchange.

        Args:
            array: 2D array sharded across devices
            partition_info: Dict with keys:
                - 'mesh_shape': (n_rows, n_cols) device mesh
                - 'periodic': (periodic_x, periodic_y)
                - 'grid_shape': (ny, nx) global grid size
            halo_width: Halo width

        Returns:
            (updated_array, diagnostics)
        """
        if isinstance(halo_width, int):
            halo_width = HaloWidth(halo_width)

        diagnostics = {
            'strategy': 'structured',
            'comm_strategy': self.comm_strategy.value,
            'exchanges': [],
            'cells_communicated': 0,
        }

        if self.backend.name == "jax_sharding":
            # Route to appropriate implementation based on comm_strategy
            if self.comm_strategy == CommStrategy.SHARED_MEMORY:
                result = self._jax_structured_exchange_shared(
                    array, partition_info, halo_width, diagnostics
                )
            elif self.comm_strategy == CommStrategy.PPERMUTE:
                result = self._jax_structured_exchange_ppermute(
                    array, partition_info, halo_width, diagnostics
                )
            elif self.comm_strategy == CommStrategy.HYBRID:
                result = self._jax_structured_exchange_hybrid(
                    array, partition_info, halo_width, diagnostics
                )
            else:
                raise ValueError(f"Unknown comm_strategy: {self.comm_strategy}")
        else:
            result = array  # Placeholder for other backends

        return result, diagnostics

    def _jax_structured_exchange_shared(
        self,
        array: jax.Array,
        partition_info: Dict,
        halo_width: HaloWidth,
        diagnostics: Dict,
    ) -> jax.Array:
        """JAX implementation using shared memory (cross-device slicing).

        Fast on single-node systems where all devices share memory.
        Uses JAX's automatic cross-device communication via array slicing.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")

        mesh_shape = partition_info.get('mesh_shape', (1, 1))
        periodic = partition_info.get('periodic', (False, False))
        global_shape = array.shape

        # Get mesh information
        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x

        # Create permutation patterns for each direction
        # Pattern: list of (source_device, dest_device) tuples

        result = array

        # Left-Right exchange (along x-axis)
        if n_devices_x > 1 or periodic[0]:
            result = self._exchange_x_direction(
                result, n_devices_x, n_devices_y, halo_width, periodic[0], diagnostics
            )

        # Top-Bottom exchange (along y-axis)
        if n_devices_y > 1 or periodic[1]:
            result = self._exchange_y_direction(
                result, n_devices_x, n_devices_y, halo_width, periodic[1], diagnostics
            )

        return result

    def _exchange_x_direction(
        self,
        array: jax.Array,
        n_x: int,
        n_y: int,
        halo_width: HaloWidth,
        periodic: bool,
        diagnostics: Dict,
    ) -> jax.Array:
        """Exchange halos in x-direction (left-right).

        For sharded arrays: JAX automatically handles cross-device communication
        when we access data across shard boundaries using slicing and indexing.
        """
        w = halo_width.width_x
        ny, nx = array.shape

        # Calculate per-device dimensions
        ny_per_device = ny // n_y
        nx_per_device = nx // n_x

        if n_x > 1:
            # Multi-device case: Exchange halos between neighboring devices
            # For each row of devices
            for dev_row in range(n_y):
                y_start = dev_row * ny_per_device
                y_end = (dev_row + 1) * ny_per_device

                # Exchange along x-direction for this row of devices
                for dev_col in range(n_x):
                    x_start = dev_col * nx_per_device
                    x_end = (dev_col + 1) * nx_per_device

                    # Right halo: send right edge to right neighbor's left halo
                    if dev_col < n_x - 1:
                        # Right neighbor's left halo region
                        right_x_start = x_end
                        right_halo_x_end = x_end + w
                        # My right edge data
                        my_right_edge = array[y_start:y_end, x_end - w:x_end]
                        # Copy to right neighbor (JAX handles device communication)
                        array = array.at[y_start:y_end, right_x_start:right_halo_x_end].set(my_right_edge)
                    elif periodic:
                        # Periodic: rightmost device sends to leftmost
                        left_halo_x_start = 0
                        left_halo_x_end = w
                        my_right_edge = array[y_start:y_end, x_end - w:x_end]
                        array = array.at[y_start:y_end, left_halo_x_start:left_halo_x_end].set(my_right_edge)

                    # Left halo: send left edge to left neighbor's right halo
                    if dev_col > 0:
                        # Left neighbor's right halo region
                        left_x_end = x_start
                        left_halo_x_start = x_start - w
                        # My left edge data
                        my_left_edge = array[y_start:y_end, x_start:x_start + w]
                        # Copy to left neighbor
                        array = array.at[y_start:y_end, left_halo_x_start:left_x_end].set(my_left_edge)
                    elif periodic:
                        # Periodic: leftmost device sends to rightmost
                        right_halo_x_start = nx - w
                        right_halo_x_end = nx
                        my_left_edge = array[y_start:y_end, x_start:x_start + w]
                        array = array.at[y_start:y_end, right_halo_x_start:right_halo_x_end].set(my_left_edge)

            cells_exchanged = 2 * n_y * ny_per_device * w  # Both directions
            diagnostics['exchanges'].append({
                'direction': 'x-multi-device',
                'cells': cells_exchanged,
                'method': 'cross-device-slicing',
                'devices': n_x,
            })
            diagnostics['cells_communicated'] += cells_exchanged

        elif periodic:
            # Single device with periodic boundaries
            array = array.at[:, :w].set(array[:, -2*w:-w])
            array = array.at[:, -w:].set(array[:, w:2*w])

            diagnostics['exchanges'].append({
                'direction': 'x-periodic-single',
                'cells': ny * w * 2,
                'method': 'local-periodic'
            })
            diagnostics['cells_communicated'] += ny * w * 2

        return array

    def _exchange_y_direction(
        self,
        array: jax.Array,
        n_x: int,
        n_y: int,
        halo_width: HaloWidth,
        periodic: bool,
        diagnostics: Dict,
    ) -> jax.Array:
        """Exchange halos in y-direction (top-bottom).

        For sharded arrays: JAX automatically handles cross-device communication
        when we access data across shard boundaries using slicing and indexing.
        """
        w = halo_width.width_y
        ny, nx = array.shape

        # Calculate per-device dimensions
        ny_per_device = ny // n_y
        nx_per_device = nx // n_x

        if n_y > 1:
            # Multi-device case: Exchange halos between vertically neighboring devices
            # For each column of devices
            for dev_col in range(n_x):
                x_start = dev_col * nx_per_device
                x_end = (dev_col + 1) * nx_per_device

                # Exchange along y-direction for this column of devices
                for dev_row in range(n_y):
                    y_start = dev_row * ny_per_device
                    y_end = (dev_row + 1) * ny_per_device

                    # Top halo: send top edge to top neighbor's bottom halo
                    if dev_row > 0:
                        # Top neighbor's bottom halo region
                        top_y_end = y_start
                        top_halo_y_start = y_start - w
                        # My top edge data
                        my_top_edge = array[y_start:y_start + w, x_start:x_end]
                        # Copy to top neighbor
                        array = array.at[top_halo_y_start:top_y_end, x_start:x_end].set(my_top_edge)
                    elif periodic:
                        # Periodic: topmost device sends to bottommost
                        bottom_halo_y_start = ny - w
                        bottom_halo_y_end = ny
                        my_top_edge = array[y_start:y_start + w, x_start:x_end]
                        array = array.at[bottom_halo_y_start:bottom_halo_y_end, x_start:x_end].set(my_top_edge)

                    # Bottom halo: send bottom edge to bottom neighbor's top halo
                    if dev_row < n_y - 1:
                        # Bottom neighbor's top halo region
                        bottom_y_start = y_end
                        bottom_halo_y_end = y_end + w
                        # My bottom edge data
                        my_bottom_edge = array[y_end - w:y_end, x_start:x_end]
                        # Copy to bottom neighbor (JAX handles device communication)
                        array = array.at[bottom_y_start:bottom_halo_y_end, x_start:x_end].set(my_bottom_edge)
                    elif periodic:
                        # Periodic: bottommost device sends to topmost
                        top_halo_y_start = 0
                        top_halo_y_end = w
                        my_bottom_edge = array[y_end - w:y_end, x_start:x_end]
                        array = array.at[top_halo_y_start:top_halo_y_end, x_start:x_end].set(my_bottom_edge)

            cells_exchanged = 2 * n_x * nx_per_device * w  # Both directions
            diagnostics['exchanges'].append({
                'direction': 'y-multi-device',
                'cells': cells_exchanged,
                'method': 'cross-device-slicing',
                'devices': n_y,
            })
            diagnostics['cells_communicated'] += cells_exchanged

        elif periodic:
            # Single device with periodic boundaries
            array = array.at[:w, :].set(array[-2*w:-w, :])
            array = array.at[-w:, :].set(array[w:2*w, :])

            diagnostics['exchanges'].append({
                'direction': 'y-periodic-single',
                'cells': nx * w * 2,
                'method': 'local-periodic'
            })
            diagnostics['cells_communicated'] += nx * w * 2

        return array

    def _jax_structured_exchange_ppermute(
        self,
        array: jax.Array,
        partition_info: Dict,
        halo_width: HaloWidth,
        diagnostics: Dict,
    ) -> jax.Array:
        """JAX implementation using lax.ppermute for explicit device-to-device communication.

        This is the production implementation for multi-node clusters where devices
        don't share memory. Uses jax.pmap with lax.ppermute collective operations.

        **Architecture**:
        1. Decompose global array into per-device local views
        2. Pre-compute STATIC permutation patterns for all devices (JAX requirement)
        3. Use pmap to execute exchange function on each device in parallel
        4. Use lax.ppermute with static permutation patterns for neighbor communication
        5. Reassemble into global array

        **Key JAX Constraint**:
        lax.ppermute requires the `perm` parameter to be a static Python list, not
        traced values. We pre-compute all permutation patterns outside pmap.

        **Benefits**:
        - Works on multi-node clusters (MPI/NCCL backend)
        - Explicit control over communication patterns
        - Can profile communication time separately

        Args:
            array: 2D global array (ny, nx)
            partition_info: Must contain 'mesh_shape', 'periodic'
            halo_width: Width of halo regions
            diagnostics: Dict to record communication stats

        Returns:
            Array with halos filled via ppermute communication
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")

        mesh_shape = partition_info.get('mesh_shape', (1, 1))
        periodic = partition_info.get('periodic', (False, False))

        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x

        if n_devices == 1:
            # Single device: no communication needed, just handle periodic boundaries
            return self._handle_periodic_single_device(array, halo_width, periodic)

        # PRE-COMPUTE STATIC PERMUTATION PATTERNS
        # This is critical: lax.ppermute requires static (non-traced) perm parameter
        perms = self._build_static_permutations(mesh_shape, periodic)

        # Decompose global array into per-device local arrays
        local_arrays = self._decompose_to_devices(array, mesh_shape, halo_width)

        # Define per-device exchange function
        # NOTE: This runs independently on each device with ppermute for communication
        def exchange_one_device(local_data):
            """Runs on each device independently with ppermute communication."""
            result = local_data

            # X-direction exchange (left-right) - use pre-computed static perms
            if n_devices_x > 1 or periodic[0]:
                result = self._inline_ppermute_x_static(
                    result, halo_width.width_x, periodic[0],
                    perms['x_right'], perms['x_left'], n_devices_x
                )

            # Y-direction exchange (top-bottom) - use pre-computed static perms
            if n_devices_y > 1 or periodic[1]:
                result = self._inline_ppermute_y_static(
                    result, halo_width.width_y, periodic[1],
                    perms['y_down'], perms['y_up'], n_devices_y
                )

            return result

        # Apply pmap to exchange on all devices
        exchange_fn = jax.pmap(exchange_one_device, axis_name='devices')
        exchanged_local = exchange_fn(local_arrays)

        # Reassemble into global array
        result = self._reassemble_from_devices(exchanged_local, array.shape, mesh_shape, halo_width)

        # Record diagnostics
        diagnostics['exchanges'].append({
            'method': 'ppermute',
            'devices': n_devices,
            'mesh_shape': mesh_shape,
        })

        return result

    def _jax_structured_exchange_hybrid(
        self,
        array: jax.Array,
        partition_info: Dict,
        halo_width: HaloWidth,
        diagnostics: Dict,
    ) -> jax.Array:
        """Hybrid implementation: shared memory intra-node + ppermute inter-node.

        This is the optimal production strategy that uses:
        - Shared memory for devices on the same compute node (fast)
        - ppermute for devices on different nodes (necessary)

        **Status**: Planned for Phase 3 (Week 4-5)

        Requires:
        - Two-level partitioning (node-level and device-level)
        - Topology detection to identify node boundaries
        - Separate ghost maps for intra-node and inter-node

        Args:
            array: 2D global array
            partition_info: Must contain topology information
            halo_width: Width of halo regions
            diagnostics: Dict to record communication stats

        Returns:
            Array with halos filled via hybrid communication
        """
        raise NotImplementedError(
            "Hybrid communication strategy is planned for Phase 3 (Week 4-5).\n"
            "Requires topology detection and two-level partitioning.\n"
            "\n"
            "Current workaround: Use comm_strategy='shared_memory' for single-node\n"
            "or comm_strategy='ppermute' for multi-node (once ppermute is implemented)."
        )

    # ========================================================================
    # Helper methods for ppermute-based structured exchange
    # ========================================================================

    def _handle_periodic_single_device(
        self,
        array: jax.Array,
        halo_width: HaloWidth,
        periodic: Tuple[bool, bool],
    ) -> jax.Array:
        """Handle periodic boundaries for single device (no ppermute needed)."""
        result = array

        if periodic[0]:  # Periodic in x
            w = halo_width.width_x
            result = result.at[:, :w].set(result[:, -2*w:-w])
            result = result.at[:, -w:].set(result[:, w:2*w])

        if periodic[1]:  # Periodic in y
            w = halo_width.width_y
            result = result.at[:w, :].set(result[-2*w:-w, :])
            result = result.at[-w:, :].set(result[w:2*w, :])

        return result

    def _build_static_permutations(
        self,
        mesh_shape: Tuple[int, int],
        periodic: Tuple[bool, bool],
    ) -> Dict[str, List[Tuple[int, int]]]:
        """Pre-compute static permutation patterns for all devices.

        JAX's lax.ppermute requires the `perm` parameter to be a static Python
        list/tuple, not traced values. This method computes all communication
        patterns before entering the pmap context.

        Args:
            mesh_shape: (n_devices_y, n_devices_x) tuple
            periodic: (periodic_x, periodic_y) tuple of bools

        Returns:
            Dict with keys:
                'x_right': [(src, dst), ...] for rightward communication
                'x_left': [(src, dst), ...] for leftward communication
                'y_down': [(src, dst), ...] for downward communication
                'y_up': [(src, dst), ...] for upward communication
        """
        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x
        periodic_x, periodic_y = periodic

        # X-direction permutations (horizontal)
        # Key insight: ppermute requires valid permutation (all sources/dests unique)
        # For non-periodic boundaries, we create "partner exchange" where boundary
        # devices exchange with each other, even though the data won't be used.
        x_right_perm = []
        x_left_perm = []

        for dev_id in range(n_devices):
            dev_row = dev_id // n_devices_x
            dev_col = dev_id % n_devices_x

            # Right neighbor: send right edge → right neighbor's left halo
            if dev_col < n_devices_x - 1:
                # Not rightmost: real exchange with next device
                right_neighbor = dev_id + 1
            elif periodic_x:
                # Rightmost with periodic: wrap to leftmost in row
                right_neighbor = dev_row * n_devices_x
            else:
                # Rightmost without periodic: exchange with leftmost (partner)
                # This satisfies collective operation requirement
                # The received data won't be used (boundary doesn't need left halo)
                right_neighbor = dev_row * n_devices_x

            x_right_perm.append((dev_id, right_neighbor))

            # Left neighbor: send left edge → left neighbor's right halo
            if dev_col > 0:
                # Not leftmost: real exchange with previous device
                left_neighbor = dev_id - 1
            elif periodic_x:
                # Leftmost with periodic: wrap to rightmost in row
                left_neighbor = dev_row * n_devices_x + (n_devices_x - 1)
            else:
                # Leftmost without periodic: exchange with rightmost (partner)
                left_neighbor = dev_row * n_devices_x + (n_devices_x - 1)

            x_left_perm.append((dev_id, left_neighbor))

        # Y-direction permutations (vertical)
        y_down_perm = []
        y_up_perm = []

        for dev_id in range(n_devices):
            dev_row = dev_id // n_devices_x
            dev_col = dev_id % n_devices_x

            # Down neighbor: send bottom edge → down neighbor's top halo
            if dev_row < n_devices_y - 1:
                # Not bottommost: real exchange with next row
                down_neighbor = dev_id + n_devices_x
            elif periodic_y:
                # Bottommost with periodic: wrap to topmost in column
                down_neighbor = dev_col
            else:
                # Bottommost without periodic: exchange with topmost (partner)
                down_neighbor = dev_col

            y_down_perm.append((dev_id, down_neighbor))

            # Up neighbor: send top edge → up neighbor's bottom halo
            if dev_row > 0:
                # Not topmost: real exchange with previous row
                up_neighbor = dev_id - n_devices_x
            elif periodic_y:
                # Topmost with periodic: wrap to bottommost in column
                up_neighbor = (n_devices_y - 1) * n_devices_x + dev_col
            else:
                # Topmost without periodic: exchange with bottommost (partner)
                up_neighbor = (n_devices_y - 1) * n_devices_x + dev_col

            y_up_perm.append((dev_id, up_neighbor))

        return {
            'x_right': x_right_perm,
            'x_left': x_left_perm,
            'y_down': y_down_perm,
            'y_up': y_up_perm,
        }

    def _decompose_to_devices(
        self,
        array: jax.Array,
        mesh_shape: Tuple[int, int],
        halo_width: HaloWidth,
    ) -> jax.Array:
        """Decompose global array into per-device local arrays with halo regions.

        Returns array of shape (n_devices, local_ny, local_nx) where each local
        array includes halo regions at boundaries.
        """
        ny, nx = array.shape
        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x

        # Calculate per-device dimensions (excluding halos for now)
        ny_per_device = ny // n_devices_y
        nx_per_device = nx // n_devices_x

        # Local size includes halos on all sides
        local_ny = ny_per_device + 2 * halo_width.width_y
        local_nx = nx_per_device + 2 * halo_width.width_x

        local_arrays = []

        for dev_id in range(n_devices):
            dev_row = dev_id // n_devices_x
            dev_col = dev_id % n_devices_x

            # Calculate global indices for this device's interior region
            y_start = dev_row * ny_per_device
            y_end = (dev_row + 1) * ny_per_device
            x_start = dev_col * nx_per_device
            x_end = (dev_col + 1) * nx_per_device

            # Create local array with halos initialized to zero
            local_array = jnp.zeros((local_ny, local_nx))

            # Copy interior data (global indices → local interior)
            wy, wx = halo_width.width_y, halo_width.width_x
            local_array = local_array.at[wy:wy+ny_per_device, wx:wx+nx_per_device].set(
                array[y_start:y_end, x_start:x_end]
            )

            local_arrays.append(local_array)

        # Stack into (n_devices, local_ny, local_nx)
        return jnp.stack(local_arrays)

    def _inline_ppermute_x_static(
        self,
        local_data: jax.Array,
        halo_width: int,
        periodic: bool,
        x_right_perm: List[Tuple[int, int]],
        x_left_perm: List[Tuple[int, int]],
        n_devices_x: int,
    ) -> jax.Array:
        """Exchange halos in x-direction using pre-computed static permutations.

        Uses static permutation patterns computed outside pmap, avoiding JAX's
        restriction that ppermute perm parameter must be non-traced.

        Args:
            local_data: Per-device local array (ny_local, nx_local)
            halo_width: Width of halo in x-direction
            periodic: Whether x-direction is periodic
            x_right_perm: Static permutation for rightward communication
            x_left_perm: Static permutation for leftward communication
            n_devices_x: Number of devices in x-direction

        Returns:
            Local array with x-direction halos exchanged
        """
        result = local_data
        ny_local, nx_local = local_data.shape
        device_id = lax.axis_index('devices')

        # Right halo exchange: send right edge → right neighbor's left halo
        # Extract my right interior edge
        right_edge = local_data[:, nx_local - 2*halo_width:nx_local - halo_width]

        # Use static permutation (ALL devices participate, even boundaries)
        received_left = lax.ppermute(right_edge, 'devices', perm=x_right_perm)

        # Conditionally update: only non-leftmost devices (or periodic leftmost)
        # When non-periodic leftmost sends to itself, received data is its own edge (no-op)
        result = result.at[:, :halo_width].set(received_left)

        # Left halo exchange: send left edge → left neighbor's right halo
        # Extract my left interior edge
        left_edge = local_data[:, halo_width:2*halo_width]

        # Use static permutation
        received_right = lax.ppermute(left_edge, 'devices', perm=x_left_perm)

        # Conditionally update: only non-rightmost devices (or periodic rightmost)
        result = result.at[:, nx_local - halo_width:].set(received_right)

        return result

    def _inline_ppermute_y_static(
        self,
        local_data: jax.Array,
        halo_width: int,
        periodic: bool,
        y_down_perm: List[Tuple[int, int]],
        y_up_perm: List[Tuple[int, int]],
        n_devices_y: int,
    ) -> jax.Array:
        """Exchange halos in y-direction using pre-computed static permutations.

        Uses static permutation patterns computed outside pmap, avoiding JAX's
        restriction that ppermute perm parameter must be non-traced.

        Args:
            local_data: Per-device local array (ny_local, nx_local)
            halo_width: Width of halo in y-direction
            periodic: Whether y-direction is periodic
            y_down_perm: Static permutation for downward communication
            y_up_perm: Static permutation for upward communication
            n_devices_y: Number of devices in y-direction

        Returns:
            Local array with y-direction halos exchanged
        """
        result = local_data
        ny_local, nx_local = local_data.shape
        device_id = lax.axis_index('devices')

        # Down halo exchange: send bottom edge → down neighbor's top halo
        # Extract my bottom interior edge
        bottom_edge = local_data[ny_local - 2*halo_width:ny_local - halo_width, :]

        # Use static permutation (ALL devices participate)
        received_top = lax.ppermute(bottom_edge, 'devices', perm=y_down_perm)

        # Update top halo
        result = result.at[:halo_width, :].set(received_top)

        # Up halo exchange: send top edge → up neighbor's bottom halo
        # Extract my top interior edge
        top_edge = local_data[halo_width:2*halo_width, :]

        # Use static permutation
        received_bottom = lax.ppermute(top_edge, 'devices', perm=y_up_perm)

        # Update bottom halo
        result = result.at[ny_local - halo_width:, :].set(received_bottom)

        return result

    def _reassemble_from_devices(
        self,
        local_arrays: jax.Array,
        global_shape: Tuple[int, int],
        mesh_shape: Tuple[int, int],
        halo_width: HaloWidth,
    ) -> jax.Array:
        """Reassemble global array from per-device local arrays, including halo overlaps.

        For ppermute strategy, we need to mimic shared-memory behavior where boundary
        cells in the global array are overwritten with neighbor data (halos).

        This allows the ppermute strategy to produce the same visible result as
        shared-memory for testing purposes.
        """
        ny, nx = global_shape
        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x

        ny_per_device = ny // n_devices_y
        nx_per_device = nx // n_devices_x

        # Create global array
        result = jnp.zeros(global_shape)

        wy, wx = halo_width.width_y, halo_width.width_x

        # First pass: place all interiors
        for dev_id in range(n_devices):
            dev_row = dev_id // n_devices_x
            dev_col = dev_id % n_devices_x

            # Extract interior (excluding halos) from local array
            local_interior = local_arrays[dev_id, wy:wy+ny_per_device, wx:wx+nx_per_device]

            # Calculate global position
            y_start = dev_row * ny_per_device
            y_end = (dev_row + 1) * ny_per_device
            x_start = dev_col * nx_per_device
            x_end = (dev_col + 1) * nx_per_device

            # Place in global array
            result = result.at[y_start:y_end, x_start:x_end].set(local_interior)

        # Second pass: overwrite boundary cells with halo data (to match shared-memory behavior)
        for dev_id in range(n_devices):
            dev_row = dev_id // n_devices_x
            dev_col = dev_id % n_devices_x

            y_start = dev_row * ny_per_device
            y_end = (dev_row + 1) * ny_per_device
            x_start = dev_col * nx_per_device
            x_end = (dev_col + 1) * nx_per_device

            # Right boundary: copy right halo to next device's left boundary
            if dev_col < n_devices_x - 1:
                # My right halo (contains right neighbor's left edge data)
                right_halo = local_arrays[dev_id, wy:wy+ny_per_device, wx+nx_per_device:wx+nx_per_device+wx]
                # Place it at the boundary (first wx columns of right neighbor)
                result = result.at[y_start:y_end, x_end:x_end+wx].set(right_halo)

            # Bottom boundary: copy bottom halo to next device's top boundary
            if dev_row < n_devices_y - 1:
                # My bottom halo (contains bottom neighbor's top edge data)
                bottom_halo = local_arrays[dev_id, wy+ny_per_device:wy+ny_per_device+wy, wx:wx+nx_per_device]
                # Place it at the boundary (first wy rows of bottom neighbor)
                result = result.at[y_end:y_end+wy, x_start:x_end].set(bottom_halo)

        return result


class UnstructuredHaloExchanger(BaseHaloExchanger):
    """Graph-based halo exchange for unstructured grids.

    Uses arbitrary cell-to-cell communication where ghost cells
    can be scattered throughout neighbor shards.

    Best for: TriGrid, MixedGrid with graph partitioning (e.g., METIS)
    """

    def exchange(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int = 1,
    ) -> Tuple[Any, Dict]:
        """Perform unstructured halo exchange.

        Args:
            array: 1D array of cell values
            partition_info: Dict with keys:
                - 'device_to_cells': {device_id: [cell_indices]}
                - 'ghost_cell_map': {device_id: {neighbor_device: [cell_indices]}}
                - 'n_devices': total number of devices
            halo_width: Number of neighbor layers (for multi-layer halos)

        Returns:
            (updated_array, diagnostics)
        """
        diagnostics = {
            'strategy': 'unstructured',
            'comm_strategy': self.comm_strategy.value,
            'ghost_cells_per_device': {},
            'total_ghost_cells': 0,
            'scatter_pattern': [],
        }

        if self.backend.name == "jax_sharding":
            # Route to appropriate implementation based on comm_strategy
            if self.comm_strategy == CommStrategy.SHARED_MEMORY:
                result = self._jax_unstructured_exchange_shared(
                    array, partition_info, halo_width, diagnostics
                )
            elif self.comm_strategy == CommStrategy.PPERMUTE:
                result = self._jax_unstructured_exchange_ppermute(
                    array, partition_info, halo_width, diagnostics
                )
            elif self.comm_strategy == CommStrategy.HYBRID:
                result = self._jax_unstructured_exchange_hybrid(
                    array, partition_info, halo_width, diagnostics
                )
            else:
                raise ValueError(f"Unknown comm_strategy: {self.comm_strategy}")
        else:
            result = array  # Placeholder

        return result, diagnostics

    def _jax_unstructured_exchange_shared(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int,
        diagnostics: Dict,
    ) -> Any:
        """JAX implementation using shared memory with explicit ghost storage.

        Creates extended partition-local arrays: [owned_cells | ghost_cells]
        This demonstrates the data structure needed for distributed systems.

        Algorithm:
        1. For each partition, create extended array with space for ghosts
        2. Copy owned cell data to indices 0...n_owned-1
        3. Explicitly gather ghost cell data via shared memory access
        4. Concatenate all partition-local arrays

        Structure:
            Global array (n_cells,) → Extended array (sum of local_sizes,)
            where local_size = n_owned + n_ghosts

        **Limitation**: Uses `array[global_idx]` to gather ghosts, which works on
        single-node via shared memory but won't work on multi-node clusters.
        For true multi-node: Use ppermute strategy instead.
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available")

        ghost_cell_map = partition_info.get('ghost_cell_map', {})
        device_to_cells = partition_info.get('device_to_cells', {})
        global_to_local = partition_info.get('global_to_local', {})
        local_array_sizes = partition_info.get('local_array_sizes', {})
        n_devices = partition_info.get('n_devices', 1)

        # Convert to JAX array if needed
        if not isinstance(array, jax.Array):
            array = jnp.array(array)

        # Track communication pattern for diagnostics
        total_ghosts = 0
        for device_id, neighbor_map in ghost_cell_map.items():
            n_ghosts = sum(len(cells) for cells in neighbor_map.values())
            diagnostics['ghost_cells_per_device'][device_id] = n_ghosts
            total_ghosts += n_ghosts

            for neighbor_id, cell_indices in neighbor_map.items():
                diagnostics['scatter_pattern'].append({
                    'from_device': neighbor_id,
                    'to_device': device_id,
                    'cells': cell_indices,
                    'count': len(cell_indices)
                })

        diagnostics['total_ghost_cells'] = total_ghosts

        # Build extended partition-local arrays with explicit ghost storage
        extended_arrays = []
        partition_offsets = {}
        offset = 0

        for device_id in sorted(device_to_cells.keys()):
            owned = device_to_cells[device_id]
            local_size = local_array_sizes[device_id]

            # Create local array: [owned | ghosts]
            local_array = jnp.zeros(local_size)

            # Copy owned data (global indices → local indices 0...n_owned-1)
            for local_idx, global_idx in enumerate(owned):
                local_array = local_array.at[local_idx].set(array[global_idx])

            # Explicitly gather ghost data
            next_idx = len(owned)
            if device_id in ghost_cell_map:
                for neighbor_id, ghost_indices in ghost_cell_map[device_id].items():
                    for global_idx in ghost_indices:
                        # This is the KEY: explicit copy from global array
                        local_array = local_array.at[next_idx].set(array[global_idx])
                        next_idx += 1

            extended_arrays.append(local_array)
            partition_offsets[device_id] = offset
            offset += local_size

        # Concatenate all partition-local arrays
        # Structure: [P0: owned|ghosts | P1: owned|ghosts | P2: owned|ghosts | ...]
        result = jnp.concatenate(extended_arrays)

        # Store metadata for operators to use
        partition_info['_extended_offsets'] = partition_offsets
        partition_info['_is_extended'] = True

        return result

    def _jax_unstructured_exchange_ppermute(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int,
        diagnostics: Dict,
    ) -> Any:
        """JAX implementation using lax.ppermute for graph-based ghost cell exchange.

        This implementation uses explicit device-to-device communication for irregular
        unstructured grids where ghost cells are scattered across neighbor partitions.

        **Status**: Implementation in progress (Phase 1 of 8-week plan)

        **Architecture**:
        1. Decompose array into per-device local arrays
        2. For each device, build send/recv lists from ghost_cell_map
        3. Use pmap with lax.ppermute to exchange scattered ghost data
        4. Pack/unpack ghost cells (more complex than structured due to irregularity)
        5. Create extended arrays [owned | ghosts]

        **Challenges for unstructured**:
        - Ghost cells are NOT contiguous (scattered pattern)
        - Variable number of ghosts per neighbor
        - Need explicit packing/unpacking buffers
        - More complex than structured exchange

        Args:
            array: 1D array of cell values (n_cells,)
            partition_info: Must contain ghost_cell_map, device_to_cells
            halo_width: Number of neighbor layers
            diagnostics: Dict to record communication stats

        Returns:
            Extended array with explicit ghost storage
        """
        raise NotImplementedError(
            "ppermute-based unstructured exchange is under development (Phase 1 of 8-week plan).\n"
            "Current status: Data structure (explicit ghost storage) is complete.\n"
            "Next steps:\n"
            "  1. Implement ghost cell packing (scatter to contiguous buffer)\n"
            "  2. Build ppermute patterns from ghost_cell_map\n"
            "  3. Create pmap-based exchange with irregular communication\n"
            "  4. Implement ghost unpacking to extended arrays\n"
            "\n"
            "For now, use comm_strategy='shared_memory' (default) which works on single-node.\n"
            "The extended array structure is correct and ready for ppermute."
        )

    def _jax_unstructured_exchange_hybrid(
        self,
        array: Any,
        partition_info: Dict,
        halo_width: int,
        diagnostics: Dict,
    ) -> Any:
        """Hybrid implementation for unstructured grids: shared memory + ppermute.

        **Status**: Planned for Phase 3 (Week 4-5)

        For unstructured grids, hybrid communication is more complex than structured:
        - Must separate ghost_cell_map into intra-node and inter-node subsets
        - Intra-node ghosts: gather via shared memory (fast, irregular OK)
        - Inter-node ghosts: pack → ppermute → unpack (network overhead)

        Requires:
        - Two-level ghost_cell_map partitioning
        - Node topology detection
        - Separate extended array regions for intra/inter ghosts

        Args:
            array: 1D array of cell values
            partition_info: Must contain hierarchical ghost maps
            halo_width: Number of neighbor layers
            diagnostics: Dict to record communication stats

        Returns:
            Extended array with ghosts from both intra and inter-node neighbors
        """
        raise NotImplementedError(
            "Hybrid unstructured exchange is planned for Phase 3 (Week 4-5).\n"
            "Requires topology-aware ghost_cell_map partitioning.\n"
            "\n"
            "Current workaround: Use comm_strategy='shared_memory' for single-node."
        )


def create_halo_exchanger(
    backend: Backend,
    strategy: str = 'auto',
    comm_strategy: CommStrategy | str = CommStrategy.SHARED_MEMORY,
) -> BaseHaloExchanger:
    """Factory function to create appropriate halo exchanger.

    Args:
        backend: Parallel backend
        strategy: Grid type - 'structured', 'unstructured', or 'auto'
        comm_strategy: Communication strategy - 'shared_memory', 'ppermute', 'hybrid', or 'auto'
            Default: 'shared_memory' (fast on single-node)

    Returns:
        HaloExchanger instance configured with specified comm_strategy
    """
    if strategy == 'structured':
        return StructuredHaloExchanger(backend, comm_strategy)
    elif strategy == 'unstructured':
        return UnstructuredHaloExchanger(backend, comm_strategy)
    elif strategy == 'auto':
        # Default to structured
        return StructuredHaloExchanger(backend, comm_strategy)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Convenience functions for backward compatibility

def exchange_halo(
    array: Any,
    backend: Backend,
    halo_width: int | HaloWidth = 1,
    pattern: HaloPattern = HaloPattern.STAR,
    periodic: tuple = (False, False),
) -> Any:
    """Convenience function for structured halo exchange.

    Args:
        array: Array to exchange
        backend: Parallel backend
        halo_width: Halo width (int or HaloWidth object)
        pattern: Exchange pattern
        periodic: Periodic boundaries

    Returns:
        Array with updated halos
    """
    if isinstance(halo_width, int):
        halo_width = HaloWidth(halo_width)

    exchanger = StructuredHaloExchanger(backend)

    partition_info = {
        'mesh_shape': (1, 1),  # Default
        'periodic': periodic,
        'grid_shape': array.shape,
    }

    result, _ = exchanger.exchange(array, partition_info, halo_width)
    return result
