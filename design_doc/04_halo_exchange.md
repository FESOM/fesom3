# Halo Exchange

## Problem Statement

Stencil operations (e.g., computing divergence, gradient, Laplacian) require data from neighboring cells. When a grid is distributed across multiple devices/nodes, some neighbors reside on different partitions. **Halo exchange** synchronizes these boundary cells before stencil computation.

```
Device 0                    Device 1
┌──────────────┬──┐    ┌──┬──────────────┐
│              │H │    │H │              │
│   Interior   │A │←──→│A │   Interior   │
│              │L │    │L │              │
│              │O │    │O │              │
└──────────────┴──┘    └──┴──────────────┘
      Owned    Ghost   Ghost    Owned
```

## Halo Width Configuration

**File**: `fesomx/core/halo_patterns.py`

### HaloWidth Class

Provides flexible halo width specification:

**1. Uniform (int)**:
```python
HaloWidth(2)  # 2 cells in all directions
```

**2. Asymmetric (tuple)**:
```python
HaloWidth((2, 3))  # x=2, y=3
```

**3. Directional (dict)**:
```python
HaloWidth({'left': 1, 'right': 2, 'top': 3, 'bottom': 1})
```

### HaloPattern Enum

| Pattern | Neighbors | Use Case |
|---------|-----------|----------|
| `STAR` | 4 (N, S, E, W) | Standard 5-point stencil |
| `BOX` | 8 (+ diagonals) | Higher-order stencils |
| `EXTENDED` | Multi-layer | Variable-width halos |

### Helper Functions

| Function | Purpose |
|----------|---------|
| `get_halo_indices_2d(shape, halo_width, direction)` | Slice indices for halo region |
| `get_interior_indices_2d(shape, halo_width)` | Slice indices for interior (non-halo) |
| `get_neighbor_directions(pattern)` | List of neighbor directions |
| `compute_halo_buffer_size(shape, halo_width, direction)` | Buffer size in elements |

## Communication Strategies

**File**: `fesomx/core/halo_exchange.py`

### CommStrategy Enum

| Strategy | Mechanism | Best For | Status |
|----------|-----------|----------|--------|
| `SHARED_MEMORY` | JAX cross-device slicing | Single-node | Implemented |
| `PPERMUTE` | `jax.lax.ppermute` with pmap | Multi-node | Implemented (structured) |
| `HYBRID` | Auto-detect intra/inter-node | Production HPC | Planned |
| `AUTO` | Alias for HYBRID | Future | Planned |

## Structured Halo Exchange

**Class**: `StructuredHaloExchanger`

Optimized for regular quadrilateral grids (QuadGrid) with rectangular device mesh.

### Data Structure

Grid partitioned into rectangular blocks:
```
┌─────┬─────┬─────┬─────┐
│ D0  │ D1  │ D2  │ D3  │   4 devices in a row
└─────┴─────┴─────┴─────┘
```

### Shared Memory Strategy

Used for single-node systems. JAX handles cross-device communication automatically via slicing.

**Algorithm**:
1. For each device pair, identify shared boundary
2. Device A sends right edge → Device B's left halo
3. Device B sends left edge → Device A's right halo
4. Handle periodic boundaries by wrapping

```python
# X-direction exchange (conceptual)
for i in range(n_devices - 1):
    device_i_right_edge = array[..., device_i_end-halo:device_i_end]
    device_j_left_halo = device_i_right_edge  # JAX handles cross-device
```

### PPERMUTE Strategy

Used for multi-node clusters with explicit peer-to-peer communication.

**Key JAX Constraint**: `lax.ppermute` requires static (non-traced) permutation patterns.

**Algorithm**:
1. **Pre-compute static permutations** outside pmap:
   - `x_right`: rightward communication pattern
   - `x_left`: leftward communication pattern
   - `y_down`, `y_up`: vertical patterns

2. **Decompose to per-device arrays** with halo regions:
   ```
   global: (ny, nx) → local: (n_devices, local_ny, local_nx)
   ```

3. **Use pmap with ppermute**:
   ```python
   @jax.pmap
   def exchange_one_device(local_array):
       # Send right edge → receive from left
       left_halo = lax.ppermute(right_edge, perm=x_left_perm)
       # Send left edge → receive from right
       right_halo = lax.ppermute(left_edge, perm=x_right_perm)
       return updated_local_array
   ```

4. **Reassemble** global array from device-local arrays

### Periodic Boundaries

- Shared memory: wrap indices at domain edges
- PPERMUTE: include wrap-around in permutation patterns

## Unstructured Halo Exchange

**Class**: `UnstructuredHaloExchanger`

Optimized for unstructured grids (TriGrid, MixedGrid) with scattered ghost cells.

### Data Structure

Extended arrays with explicit ghost storage:
```
Global: [cell_0, cell_1, ..., cell_n]
        ↓
Extended per device:
  Device 0: [owned_cells | ghost_cells_from_device_1 | ghost_cells_from_device_2]
  Device 1: [owned_cells | ghost_cells_from_device_0 | ghost_cells_from_device_2]
```

Partition info contains:
- `device_to_cells`: owned cells per device
- `ghost_cell_map`: which cells each device needs from others
- `global_to_local`: global → local index mapping

### Shared Memory Strategy

**Algorithm**:
1. Create extended array for each device
2. Copy owned cells to local positions
3. **Gather ghost cells** via shared memory:
   ```python
   for global_idx in ghost_indices:
       local_array[local_idx] = global_array[global_idx]
   ```
4. Concatenate all extended arrays

**Output Format**:
```
result = [P0_owned | P0_ghosts | P1_owned | P1_ghosts | ...]
```

### PPERMUTE Strategy (Under Development)

Challenges:
- Ghost cells are scattered (not contiguous)
- Variable number of ghosts per neighbor
- Requires explicit pack/unpack buffers

## Usage

### Via Grid Class

```python
grid = QuadGrid("test", nx=64, ny=64)
grid.create_mesh()
grid.compute_neighbors(halo_width=2)
grid.add_field("temperature", temp_data)

# Perform halo exchange
grid.halo_exchange("temperature", halo_width=2)

# Get result
result = grid.get_field_numpy("temperature")
```

### Via Factory Function

```python
from fesomx.core.halo_exchange import create_halo_exchanger

exchanger = create_halo_exchanger(
    backend=backend,
    strategy='structured',
    comm_strategy='ppermute'
)

result, diagnostics = exchanger.exchange(array, partition_info, halo_width=1)
```

### Convenience Function

```python
from fesomx.core.halo_exchange import exchange_halo

result = exchange_halo(
    array=array,
    backend=backend,
    halo_width=1,
    pattern=HaloPattern.STAR,
    periodic=(True, False)
)
```

## Diagnostics

Both exchangers return diagnostic information:

```python
result, diagnostics = exchanger.exchange(array, partition_info, halo_width)

# Structured diagnostics
{
    'strategy': 'structured',
    'comm_strategy': 'ppermute',
    'exchanges': [
        {'direction': 'x', 'cells': 128, 'method': 'ppermute'},
        {'direction': 'y', 'cells': 64, 'method': 'ppermute'}
    ],
    'cells_communicated': 192
}

# Unstructured diagnostics
{
    'strategy': 'unstructured',
    'ghost_cells_per_device': {0: 25, 1: 30},
    'total_ghost_cells': 55
}
```

## Why Flexible Halo Width?

Higher-order operators require wider halos:

| Operator Order | Stencil Points | Required Halo Width |
|----------------|----------------|---------------------|
| 2nd order | 5-point | 1 |
| 4th order | 9-point | 2 |
| 6th order | 13-point | 3 |

FESOMx supports configurable halo width to enable:
- Higher-order accuracy
- Future operator extensions
- Mixed-order schemes

## Implementation Status

| Configuration | Structured | Unstructured |
|---------------|------------|--------------|
| Shared Memory | Implemented | Implemented |
| PPERMUTE | Implemented | In Progress |
| Hybrid | Planned | Planned |

## Related Documents

- [01_grids_and_staggering.md](01_grids_and_staggering.md) - Grid types
- [02_operators_and_their_dispatch.md](02_operators_and_their_dispatch.md) - How operators use halo exchange
- [03_backend_abstraction.md](03_backend_abstraction.md) - Backend integration
