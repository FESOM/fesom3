# User Interaction and Visualization

## Overview

FESOMx provides utilities for visualizing grids, fields, partitioning, and halo exchanges. The current implementation focuses on development and debugging support using matplotlib.

**File**: `fesomx/utilities/visualization.py`

## Visualization Functions

### Grid Visualization

```python
from fesomx.utilities.visualization import plot_grid

# Basic grid plot
ax = plot_grid(
    vertices=grid.vertices,
    cells=grid.cells,
    show_vertices=True,
    show_cells=True,
    title="My Grid"
)
```

**Options**:
- `show_vertices`: Plot vertex points
- `show_cells`: Plot cell edges
- Works with uniform and mixed connectivity

### Field Visualization

```python
from fesomx.utilities.visualization import plot_field

# Cell-centered field
ax = plot_field(
    vertices=grid.vertices,
    cells=grid.cells,
    field_data=temperature,
    title="Temperature",
    cmap='viridis',
    vmin=0, vmax=100
)
```

**Features**:
- Automatic detection of cell-centered vs vertex-centered data
- Colorbar with field values
- Supports custom colormaps and value ranges

### Halo Region Visualization

```python
from fesomx.utilities.visualization import plot_halo_regions

# Show halo cells on structured grid
ax = plot_halo_regions(
    vertices=grid.vertices,
    cells=grid.cells,
    halo_width=2,
    shape=(ny, nx),
    title="Halo Regions"
)
```

**Use Case**: Debugging halo exchange correctness

### Partition Visualization

```python
from fesomx.utilities.visualization import plot_partition_boundaries

# Show mesh partitioning
ax = plot_partition_boundaries(
    vertices=grid.vertices,
    cells=grid.cells,
    partition=partition_array,
    show_boundaries=True,
    title="Mesh Partitioning"
)
```

**Features**:
- Colors cells by partition ID
- Highlights partition boundary edges
- Useful for checking METIS output

### Ghost Cell Visualization

```python
from fesomx.utilities.visualization import plot_ghost_cells

# Show ghost cells for a specific device
ax = plot_ghost_cells(
    vertices=grid.vertices,
    cells=grid.cells,
    partition=partition_array,
    device_id=0,
    ghost_cell_map=ghost_map,
    title="Ghost Cells for Device 0"
)
```

**Shows**:
- Owned cells (yellow)
- Ghost cells needed (green)
- Other devices' cells (red)

### Communication Graph

```python
from fesomx.utilities.visualization import plot_communication_graph

# Show device communication pattern
ax = plot_communication_graph(
    partition_info=partition_info,
    layout='grid',  # or 'spring', 'circular'
    title="Device Communication"
)
```

**Shows**:
- Nodes = devices
- Edges = communication links
- Edge labels = ghost cell counts (if available)

## Examples

FESOMx provides 17 example scripts demonstrating different features:

### Grid and Halo Exchange (01-12)

| Example | Description |
|---------|-------------|
| 01_simple_quad_halo.py | Basic quadrilateral grid with halo exchange |
| 02_triangle_halo.py | Unstructured triangular mesh |
| 03_mixed_grid_halo.py | Mixed quad/triangle grid |
| 04_staggered_grid.py | Arakawa A, B, C staggering |
| 05_multi_width_halo.py | Variable halo widths |
| 06_parallel_scaling.py | Multi-device parallelism |
| 07_halo_visualization.py | Partitioning visualization |
| 08_halo_correctness.py | Partition validation |
| 09_partition_visualization.py | Partition strategies |
| 10_data_movement.py | Cross-device data transfer |
| 11_partition_and_save.py | METIS partitioning |
| 12_load_and_exchange.py | Distributed halo exchange |

### Mathematical Operators (13-17)

| Example | Description |
|---------|-------------|
| 13_divergence_halo.py | Divergence with halo exchange |
| 14_laplacian_diffusion.py | Heat diffusion simulation |
| 15_gradient_pressure.py | Pressure gradient force |
| 16_triangle_divergence.py | Triangle grid operators |
| 17_triangle_divergence_halo.py | Distributed triangle computation |

### Running Examples

```bash
# Run directly
python examples/01_simple_quad_halo.py

# With GPU
JAX_PLATFORMS=gpu python examples/01_simple_quad_halo.py

# Via script
bash scripts/run_example.sh examples/01_simple_quad_halo.py
```

## Common Patterns

### Basic Workflow

```python
from fesomx import QuadGrid, StaggerType, divergence
import jax.numpy as jnp

# 1. Create grid
grid = QuadGrid("my_grid", nx=32, ny=32, stagger_type=StaggerType.C)
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))

# 2. Create fields
u = jnp.ones((32, 32))
v = jnp.zeros((32, 32))

# 3. Compute operator
div = divergence(u, v, grid)

# 4. Visualize
from fesomx.utilities.visualization import plot_field
plot_field(grid.vertices, grid.cells, div.flatten(), title="Divergence")
```

### Distributed Workflow

```python
from fesomx import QuadGrid, get_backend
from jax.sharding import Mesh, PartitionSpec as P
import jax

# 1. Setup device mesh
devices = jax.devices()
mesh = Mesh(devices.reshape(2, 2), axis_names=('x', 'y'))

# 2. Create sharded data
with mesh:
    data_sharded = jax.device_put(data, NamedSharding(mesh, P('x', 'y')))

# 3. Create partition info
partition_info = grid.create_partition_info(mesh_shape=(2, 2))

# 4. Compute with halo exchange
result = divergence(u, v, grid, partition_info=partition_info)
```

## Saving Plots

```python
from fesomx.utilities.visualization import save_plot

# Create visualization
ax = plot_field(vertices, cells, field)

# Save to file
save_plot("output/field.png", dpi=300)
```

## Limitations

| Limitation | Impact | Planned Solution |
|------------|--------|------------------|
| matplotlib only | No interactive plots | Add plotly/bokeh |
| 2D only | No 3D visualization | Add 3D support |
| Basic colormaps | Limited aesthetics | Custom colormap library |
| No animation | Static snapshots only | Add animation support |

## Future Extensions

- **Interactive visualization**: Plotly, Bokeh backends
- **3D support**: Volume rendering, isosurfaces
- **Animation**: Time-evolving fields
- **Dashboard**: Real-time monitoring during simulation
- **VTK export**: Integration with ParaView

## Related Documents

- [01_grids_and_staggering.md](01_grids_and_staggering.md) - Grid structures to visualize
- [04_halo_exchange.md](04_halo_exchange.md) - Partition info for ghost cell plots
- [05_IO.md](05_IO.md) - Saving visualization output
