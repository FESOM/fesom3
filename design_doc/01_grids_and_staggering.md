# Grids and Staggering

## Rationale for Grid Hierarchy

FESOMx needs to support both structured and unstructured grids to serve different modeling scenarios:

- **Structured grids** (QuadGrid): Support both finite volume (FV) and optionally finite difference (FD) operators
- **Unstructured grids** (TriGrid, MixedGrid): Provide flexible geometry for complex domains using finite volume (FV) methods

FV methods work uniformly across all grid types, while FD is an optional optimization for structured grids where array slicing enables efficient stencil operations.

A common abstract interface (`BaseGrid`) enables operator polymorphism across grid types. The `is_structured()` method serves as the key dispatch criterion for algorithm selection.

## BaseGrid Abstract Contract

**File**: `fesomx/core/base_grid.py`

BaseGrid defines the universal interface that all grid types must implement:

### Abstract Methods (Must Implement)

| Method | Returns | Purpose |
|--------|---------|---------|
| `create_mesh(*args)` | None | Create grid topology and geometry |
| `get_cell_centers()` | (n_cells, 2) array | Cell centroid coordinates |
| `get_cell_volumes()` | (n_cells,) array | Cell areas |
| `compute_neighbors(halo_width)` | None | Build neighbor connectivity |
| `is_structured()` | bool | **Key dispatch method** |

### Concrete Methods (Inherited)

| Method | Purpose |
|--------|---------|
| `add_field(name, data, location)` | Store field with stagger metadata |
| `get_field(name)` | Retrieve backend-specific array |
| `get_field_numpy(name)` | Convert to NumPy array |
| `halo_exchange(name, halo_width)` | Perform ghost cell exchange |

### Field Storage

Fields are stored in `self.fields` dictionary with metadata:

```python
self.fields[name] = {
    'data': backend_array,           # JAX array
    'stagger': StaggeredField(...),  # Location metadata
    'sharding': sharding_spec,       # Distributed layout
}
```

## Grid Types

| Grid | is_structured() | Cell Type | Indexing | Use Case |
|------|-----------------|-----------|----------|----------|
| QuadGrid | `True` | Quad only | Logical (i,j) | Regular domains, FV (+ optional FD) |
| TriGrid | `False` | Triangle only | Cell ID list | Complex geometry, FESOM-like |
| MixedGrid | `False` | Tri + Quad | Cell ID list | Transition regions |

### QuadGrid (Structured)

**File**: `fesomx/core/quad_grid.py`

Represents logically rectangular grids that may be curvilinear in physical space.

**Key Features**:
- Cells arranged in (ny, nx) logical structure
- Efficient array slicing for stencil operations
- Directional neighbors: left, right, top, bottom

**Index Conversion**:
```python
# Linear index to logical (i, j)
i, j = grid.get_logical_indices(cell_id)

# Logical (i, j) to linear index
cell_id = grid.get_linear_index(i, j)
```

**Mesh Creation**:
```python
grid = QuadGrid("my_grid", nx=32, ny=32, stagger_type=StaggerType.C)
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
```

### TriGrid (Unstructured Triangular)

**File**: `fesomx/core/tri_grid.py`

Represents completely triangular unstructured meshes, similar to FESOM2.

**Key Features**:
- Edge-based connectivity via `_build_edges()` and `_build_cell_neighbors()`
- Neighbor list: `cell_neighbors[cell_id]` = list of adjacent cell IDs
- Periodic boundary conditions via edge midpoint matching
- Uses graph partitioning (METIS) for domain decomposition

**Mesh Creation**:
```python
from fesomx.utilities.mesh_generation import create_triangular_mesh

mesh_data = create_triangular_mesh(vertices, method='delaunay')
grid = TriGrid("unstructured")
grid.create_mesh(mesh_data['vertices'], mesh_data['triangles'])
```

### MixedGrid (Unstructured Mixed)

**File**: `fesomx/core/mixed_grid.py`

Supports both triangular and quadrilateral cells in the same mesh.

**Key Features**:
- `cell_types` array: 3 = triangle, 4 = quad
- Variable-length connectivity in `cells` list
- Dual implementation pattern: dispatches correct algorithm per cell type

**Cell Type Queries**:
```python
tri_indices = grid.get_triangular_cells()    # Returns triangle cell IDs
quad_indices = grid.get_quadrilateral_cells() # Returns quad cell IDs
```

**Mesh Creation**:
```python
from fesomx.utilities.mesh_generation import create_mixed_mesh

mesh_data = create_mixed_mesh(nx=16, ny=16, tri_fraction=0.3)
grid = MixedGrid("mixed")
grid.create_mesh(mesh_data['vertices'], mesh_data['cells'])
```

## Staggering Schemes

**File**: `fesomx/core/staggering.py`

### Why Staggering Matters

Different staggering schemes have different numerical properties:

| Scheme | Pros | Cons | Used By |
|--------|------|------|---------|
| A-grid | Simple implementation | Checkerboard pressure oscillations | Simple models |
| B-grid | Good for vorticity | Requires averaging for fluxes | FESOM2, atmospheric models |
| C-grid | Optimal wave propagation | More complex indexing | MOM, NEMO, MITgcm |

### StaggerType Enum

```python
class StaggerType(Enum):
    A = "A"  # All variables at cell centers
    B = "B"  # Velocities at corners, scalars at centers
    C = "C"  # Velocities on faces, scalars at centers
```

### StaggerLocation Enum

Defines possible variable locations within a cell:

| Location | Position | Offset from Center |
|----------|----------|-------------------|
| `CENTER` | Cell center | (0.0, 0.0) |
| `CORNER` | Cell vertex | (0.5, 0.5) |
| `FACE_X` | Right edge midpoint | (0.5, 0.0) |
| `FACE_Y` | Top edge midpoint | (0.0, 0.5) |
| `EDGE_X` | x-aligned edge | (0.5, 0.0) |
| `EDGE_Y` | y-aligned edge | (0.0, 0.5) |

### StaggeredField

A NamedTuple that associates a field with its stagger metadata:

```python
from fesomx.core.staggering import StaggeredField, StaggerLocation

field = StaggeredField(
    name='u',
    location=StaggerLocation.FACE_X,
    stagger_type=StaggerType.C
)

# Get offset for interpolation
offset_x, offset_y = field.get_offset()  # (0.5, 0.0)
```

### Stagger Configuration

`get_stagger_config(stagger_type)` returns standard variable locations:

**A-grid**:
```python
{'u': CENTER, 'v': CENTER, 'p': CENTER, 'T': CENTER}
```

**B-grid**:
```python
{'u': CORNER, 'v': CORNER, 'p': CENTER, 'T': CENTER}
```

**C-grid**:
```python
{'u': FACE_X, 'v': FACE_Y, 'p': CENTER, 'T': CENTER}
```

### Visual Representation

```
A-grid:                    B-grid:                    C-grid:
+-------+-------+          u,v-----u,v-----u,v        +---v---+---v---+
|       |       |          |       |       |          |       |       |
| u,v,p | u,v,p |          |  p,T  |  p,T  |          u  p,T  u  p,T  u
|       |       |          |       |       |          |       |       |
+-------+-------+          u,v-----u,v-----u,v        +---v---+---v---+
|       |       |          |       |       |          |       |       |
| u,v,p | u,v,p |          |  p,T  |  p,T  |          u  p,T  u  p,T  u
|       |       |          |       |       |          |       |       |
+-------+-------+          u,v-----u,v-----u,v        +---v---+---v---+
```

## Mesh Creation Utilities

**File**: `fesomx/utilities/mesh_generation.py`

| Function | Output Grid | Description |
|----------|-------------|-------------|
| `create_rectangular_mesh(nx, ny, domain)` | QuadGrid | Regular quadrilateral mesh |
| `create_triangular_mesh(vertices)` | TriGrid | Delaunay triangulation of points |
| `create_structured_triangular_mesh(nx, ny)` | TriGrid | Regular triangles (each quad split into 2) |
| `create_mixed_mesh(nx, ny, tri_fraction)` | MixedGrid | Random mix of triangles and quads |

### Current Test Grids vs Future Production Grids

Current tests generate unstructured grids from regular grids (e.g., splitting quads into triangles) for validation and development. This approach ensures correctness testing is straightforward.

**Future base grids planned**:
- **FESOM2 grids**: Real ocean model meshes with variable resolution
- **HEALPix grids**: Hierarchical equal-area isolatitude pixelization (common in climate/atmospheric models)

These will be loaded from external mesh files rather than generated programmatically.

## Key Design Pattern: is_structured() Dispatch

The `is_structured()` method enables runtime algorithm selection:

```python
def divergence(u, v, grid, **kwargs):
    if grid.is_structured():
        # QuadGrid: use finite difference (efficient array slicing)
        return fd_divergence(u, v, grid, **kwargs)
    else:
        # TriGrid/MixedGrid: use finite volume (edge-based fluxes)
        return fv_divergence(u, v, grid, **kwargs)
```

This pattern allows:
- Same user API for all grid types
- Optimal algorithm selection per grid topology
- Easy extension for new grid types

## Related Documents

- [00_goals_and_architecture_overview.md](00_goals_and_architecture_overview.md) - Project goals
- [02_operators_and_their_dispatch.md](02_operators_and_their_dispatch.md) - How operators use grid dispatch
- [04_halo_exchange.md](04_halo_exchange.md) - Grid-aware halo exchange
