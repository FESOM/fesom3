# Core Modules

Core grid infrastructure for JAX-based distributed mesh computations.

## Architecture Overview

```
BaseGrid (abstract)
├── QuadGrid      - Structured quadrilateral grids
├── TriGrid       - Unstructured triangular grids
└── MixedGrid     - Mixed quad/triangle grids

Backend (abstract)
└── JAXShardingBackend - JAX sharding implementation

HaloExchanger (abstract)
├── StructuredHaloExchanger   - Edge-based exchange
└── UnstructuredHaloExchanger - Graph-based exchange
```

## Module Reference

### `base_grid.py`
**Abstract base class for all grids**

- `BaseGrid` - Defines common interface for grids
  - `create_mesh()` - Build mesh geometry
  - `get_cell_centers()` - Compute cell centroids
  - `get_cell_volumes()` - Compute cell areas/volumes
  - `compute_neighbors()` - Build neighbor connectivity

### `quad_grid.py`
**Structured quadrilateral (rectangular) grids**

- `QuadGrid` - Regular 2D grid with i,j indexing
  - `create_mesh(nx, ny, domain)` - Create rectangular mesh
  - `create_partition_info(mesh_shape)` - Structured partitioning
  - Returns partition with directional neighbors (left/right/top/bottom)
  - Best for: Regular domains, aligned boundaries

**Example:**
```python
grid = QuadGrid("my_grid", nx=64, ny=64, stagger_type=StaggerType.A)
grid.create_mesh(domain=(0, 1, 0, 1))
partition_info = grid.create_partition_info(mesh_shape=(2, 2))  # 2×2 devices
```

### `tri_grid.py`
**Unstructured triangular grids**

- `TriGrid` - Arbitrary triangular mesh
  - `create_mesh(vertices, triangles)` - Build from connectivity
  - `create_partition_info(n_partitions, method)` - Graph partitioning
  - Supports METIS or geometric partitioning
  - Returns ghost cell maps for irregular communication
  - Best for: Complex domains, adaptive refinement

**Example:**
```python
grid = TriGrid("tri_mesh", stagger_type=StaggerType.A)
grid.create_mesh(vertices=verts, triangles=tris)
partition_info = grid.create_partition_info(n_partitions=4, method='metis')
```

### `mixed_grid.py`
**Mixed quadrilateral and triangular grids**

- `MixedGrid` - Hybrid meshes with both quads and triangles
  - `create_mesh(vertices, quads, triangles)` - Build from mixed connectivity
  - Uses ragged arrays for variable cell types
  - Best for: Transitional regions, boundary layers

### `staggering.py`
**Arakawa grid staggering schemes**

- `StaggerType` - Enumeration: A, B, C
  - **A-grid**: All variables at cell centers
  - **B-grid**: Velocities at cell corners
  - **C-grid**: Velocities at cell faces (edges)
- `get_stagger_config(type)` - Returns variable location map

**Example:**
```python
from core import StaggerType, get_stagger_config

config = get_stagger_config(StaggerType.C)
# {'h': 'center', 'u': 'face_x', 'v': 'face_y'}
```

### `backend.py`
**Abstract backend interface**

- `Backend` - Base class for parallel backends
  - `initialize(**kwargs)` - Setup parallel environment
  - `halo_exchange()` - Exchange ghost cells
  - `gather()` / `scatter()` - Collective operations
- `get_backend(name, **kwargs)` - Factory function

**Available backends:**
- `"jax_sharding"` - JAX with NamedSharding (default)
- Future: `"jax_mpi"`, `"numpy_mpi"`

### `jax_sharding_backend.py`
**JAX sharding implementation**

- `JAXShardingBackend` - Distributed arrays via JAX
  - Uses `jax.sharding.Mesh` for device layout
  - Automatic cross-device communication
  - Supports multi-CPU and multi-GPU

**Example:**
```python
backend = get_backend('jax_sharding', mesh_shape=(2, 4))
# Creates 2×4 device mesh (8 devices)
```

### `halo_exchange.py`
**Ghost cell synchronization**

**Two strategies:**

1. **StructuredHaloExchanger** - For QuadGrid
   - Edge-based communication (contiguous halos)
   - Directional exchanges (x, y)
   - Efficient for rectangular partitions

2. **UnstructuredHaloExchanger** - For TriGrid/MixedGrid
   - Graph-based communication (scattered ghosts)
   - Uses ghost_cell_map from partitioning
   - Handles irregular neighbor patterns

**Key methods:**
- `exchange(array, partition_info, halo_width)` - Perform exchange
- Returns `(updated_array, diagnostics)`

**Example:**
```python
from core.halo_exchange import StructuredHaloExchanger

exchanger = StructuredHaloExchanger(backend)
data_exchanged, diagnostics = exchanger.exchange(
    array, partition_info, halo_width=2
)
print(f"Cells communicated: {diagnostics['cells_communicated']}")
```

### `halo_patterns.py`
**Halo region specifications**

- `HaloWidth` - Halo dimensions (can differ in x/y)
- `HaloPattern` - Communication patterns (STAR, BOX, CUSTOM)
- `get_halo_indices_2d()` - Compute halo regions
- `get_neighbor_directions()` - Determine neighbor layout

## Partition Information Structure

### QuadGrid partition_info:
```python
{
    'mesh_shape': (n_y, n_x),           # Device mesh dimensions
    'periodic': (periodic_x, periodic_y),
    'grid_shape': (ny, nx),              # Global grid size
    'device_to_cells': {
        device_id: (j_start, j_end, i_start, i_end)
    },
    'neighbor_map': {
        device_id: {'left': id, 'right': id, 'top': id, 'bottom': id}
    },
    'strategy': 'structured'
}
```

### TriGrid partition_info:
```python
{
    'n_devices': n_partitions,
    'device_to_cells': {
        device_id: [cell_indices]        # List of owned cells
    },
    'ghost_cell_map': {
        device_id: {
            neighbor_id: [ghost_cell_indices]  # Cells to receive
        }
    },
    'partition': np.ndarray,             # Partition ID per cell
    'strategy': 'unstructured',
    'info': {...}                        # Partitioning statistics
}
```

## Typical Workflow

```python
import jax.numpy as jnp
from core import QuadGrid, StaggerType
from core.backend import get_backend
from core.halo_exchange import StructuredHaloExchanger

# 1. Create grid
grid = QuadGrid("sim", nx=128, ny=128, stagger_type=StaggerType.A)
grid.create_mesh(domain=(0, 1, 0, 1))

# 2. Setup backend
backend = get_backend('jax_sharding', mesh_shape=(4, 8))  # 32 devices

# 3. Create partition
partition_info = grid.create_partition_info(mesh_shape=(4, 8))

# 4. Initialize data (sharded)
data = jnp.zeros((128, 128))

# 5. Perform halo exchange
exchanger = StructuredHaloExchanger(backend)
data, diagnostics = exchanger.exchange(data, partition_info, halo_width=2)

# 6. Use data for computation
# ... stencil operations, advection, etc.
```

## Performance Notes

- **QuadGrid**: O(1) neighbor lookup, aligned memory access
- **TriGrid**: O(k) neighbor lookup (k = avg neighbors), irregular access
- **Halo width**: Communication cost ∝ perimeter × halo_width
- **Load balance**: QuadGrid perfect, TriGrid depends on partitioner (METIS ≈ 1-5% imbalance)

## Dependencies

- **Required**: JAX, NumPy
- **Optional**: pymetis (for TriGrid METIS partitioning)
- **Future**: mpi4py (for JAX MPI backend)

## See Also

- `../utilities/` - Mesh generation, visualization, I/O
- `../examples/` - Usage examples and tutorials
- `../tests/` - Unit tests and validation
