# Operators Module

Mathematical differential operators for structured and unstructured grids with automatic method dispatch.

## Overview

The operators module provides a **unified API** for computing differential operators on both structured (QuadGrid) and unstructured (TriGrid, MixedGrid) grids. The same function calls automatically dispatch to the appropriate numerical method based on grid type.

**Key Features**:
- Single API works for all grid types
- Automatic method selection via `grid.is_structured()`
- JAX-native implementation (JIT-compilable, GPU-accelerated)
- Support for distributed computing with halo exchange
- Modular architecture for easy extension

## Supported Operators

### 1. Divergence: ∇·**u**

Computes the divergence of a 2D vector field.

**Mathematical Definition**:
```
∇·u = ∂u/∂x + ∂v/∂y
```

**Usage**:
```python
from operators import divergence

# Works on any grid type!
div = divergence(u, v, grid)
```

**Methods**:
- **Structured grids** (QuadGrid): Finite difference with A/B/C-grid staggering
  - A-grid: Central differences at cell centers
  - C-grid: Natural divergence at cell centers (velocity at faces)
- **Unstructured grids** (TriGrid): Finite volume with edge flux summation

### 2. Gradient: ∇φ

Computes the gradient of a scalar field.

**Mathematical Definition**:
```
∇φ = (∂φ/∂x, ∂φ/∂y)
```

**Usage**:
```python
from operators import gradient

dphidx, dphidy = gradient(phi, grid)
```

**Methods**:
- **Structured grids**: 2nd order central differences
- **Unstructured grids**: Green-Gauss reconstruction (finite volume)

### 3. Laplacian: ∇²φ

Computes the Laplacian of a scalar field.

**Mathematical Definition**:
```
∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
```

**Usage**:
```python
from operators import laplacian

# 2nd order (default)
lapl = laplacian(phi, grid)

# 4th order (structured grids only)
lapl = laplacian(phi, grid, order=4)
```

**Methods**:
- **Structured grids**: 2nd or 4th order finite difference stencils
- **Unstructured grids**: ∇·(∇φ) using finite volume (two-step process)

## Architecture

### Modular Structure

```
operators/
├── __init__.py              # Public API with automatic dispatch
├── base.py                  # Abstract operator interface
├── structured/              # Finite difference implementations
│   ├── divergence_fd.py    # Structured divergence
│   ├── gradient_fd.py      # Structured gradient
│   └── laplacian_fd.py     # Structured Laplacian
└── unstructured/            # Finite volume implementations
    ├── divergence_fv.py    # Unstructured divergence
    ├── gradient_fv.py      # Unstructured gradient (Green-Gauss)
    └── laplacian_fv.py     # Unstructured Laplacian
```

### Dispatch Mechanism

The unified API in `operators/__init__.py` automatically selects the appropriate implementation:

```python
def divergence(u, v, grid, **kwargs):
    """Compute divergence with automatic method selection."""
    if grid.is_structured():
        return divergence_fd(u, v, grid, **kwargs)  # Finite difference
    else:
        return divergence_fv(u, v, grid, **kwargs)  # Finite volume
```

This pattern is repeated for `gradient()` and `laplacian()`.

## Usage Examples

### QuadGrid (Structured)

```python
import jax.numpy as jnp
from core import QuadGrid, StaggerType
from operators import divergence, gradient, laplacian

# Create structured grid
grid = QuadGrid(name="quad", nx=64, ny=64, stagger_type=StaggerType.A)
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
grid.dx = 1.0 / (grid.nx - 1)
grid.dy = 1.0 / (grid.ny - 1)

# Create fields (2D arrays)
u = jnp.ones((64, 64))
v = jnp.zeros((64, 64))
phi = jnp.random.rand(64, 64)

# Compute operators
div = divergence(u, v, grid)              # Shape: (64, 64)
dpdx, dpdy = gradient(phi, grid)          # Each: (64, 64)
lapl = laplacian(phi, grid, order=2)      # Shape: (64, 64)
```

### TriGrid (Unstructured)

```python
from core import TriGrid, StaggerType
from operators import divergence, gradient, laplacian

# Create triangular grid
vertices = [[0, 0], [1, 0], [0.5, 0.5], [1, 1], [0, 1]]
triangles = [[0, 1, 2], [1, 3, 2], [2, 3, 4], [0, 2, 4]]

grid = TriGrid(name="tri", stagger_type=StaggerType.A)
grid.create_mesh(vertices=vertices, triangles=triangles)
grid.compute_neighbors(halo_width=1)

# Create fields (1D arrays at cell centers)
n_cells = len(triangles)
u = jnp.ones(n_cells)
v = jnp.zeros(n_cells)
phi = jnp.random.rand(n_cells)

# Same API, different implementation!
div = divergence(u, v, grid)              # Shape: (n_cells,)
dpdx, dpdy = gradient(phi, grid)          # Each: (n_cells,)
lapl = laplacian(phi, grid)               # Shape: (n_cells,)
```

### With Halo Exchange (Distributed Computing)

```python
from jax.sharding import Mesh, PartitionSpec as P
import jax

# Create device mesh
mesh = Mesh(jax.devices(), axis_names=('x', 'y'))

# Shard arrays
with mesh:
    u_sharded = jax.device_put(u, jax.sharding.NamedSharding(mesh, P('x', 'y')))
    v_sharded = jax.device_put(v, jax.sharding.NamedSharding(mesh, P('x', 'y')))

# Create partition info
partition_info = {
    'mesh': mesh,
    'partition_spec': P('x', 'y'),
    'shape': u.shape,
    'n_partitions': (2, 2)
}

# Compute with automatic halo exchange
div = divergence(
    u_sharded, v_sharded, grid,
    partition_info=partition_info,
    perform_halo_exchange=True  # Automatic neighbor communication
)
```

## Numerical Methods

### Finite Difference (Structured Grids)

**Advantages**:
- High accuracy on regular grids
- Simple stencil operations
- Efficient array slicing

**A-Grid Divergence** (2nd order central differences):
```python
div[i, j] = (u[i+1, j] - u[i-1, j]) / (2*dx) +
            (v[i, j+1] - v[i, j-1]) / (2*dy)
```

**Laplacian** (4th order option):
```python
lapl[i, j] = (-u[i+2,j] + 16*u[i+1,j] - 30*u[i,j] + 16*u[i-1,j] - u[i-2,j]) / (12*dx²) +
             (-u[i,j+2] + 16*u[i,j+1] - 30*u[i,j] + 16*u[i,j-1] - u[i,j-2]) / (12*dy²)
```

### Finite Volume (Unstructured Grids)

**Advantages**:
- Works on arbitrary polygonal cells
- Conserves quantities exactly
- Natural for flux-based problems

**Divergence** (edge flux summation):
```
∇·u = (1/A) Σ_edges (u·n̂) * edge_length
```

Where:
- A = cell area
- u·n̂ = velocity dotted with outward normal
- Sum over all edges of the cell

**Green-Gauss Gradient**:
```
∇φ = (1/A) Σ_edges φ_edge * n̂ * edge_length
```

Where φ_edge is interpolated from cell centers to edge midpoints.

## Testing

The module includes comprehensive parametrized tests that verify correctness on both grid types:

```bash
# Run operator tests
pytest tests/test_operators.py -v              # 114 structured grid tests
pytest tests/test_operators_parametrized.py -v # 13 cross-grid tests

# Run all tests
pytest tests/ -v                                # 127 total tests
```

**Test Coverage**:
- Divergence of divergence-free fields (should be zero)
- Gradient of constant fields (should be zero)
- Laplacian of constant fields (should be zero)
- Gradient of Gaussian fields (compared to analytical solution)
- Laplacian of Gaussian fields (compared to analytical solution)
- Shape correctness for all operators
- Cross-grid comparison (quad vs triangle)

## Performance Considerations

### Structured Grids (Finite Difference)

**Pros**:
- Very fast (vectorized array operations)
- Excellent memory locality
- High accuracy (2nd/4th order)

**Typical Performance**:
- 64×64 grid: ~0.1 ms per operation
- 512×512 grid: ~10 ms per operation

### Unstructured Grids (Finite Volume)

**Pros**:
- Handles arbitrary geometries
- Exact conservation
- Flexible mesh refinement

**Cons**:
- Slower than structured (edge loops)
- Lower accuracy (1st order)
- More complex implementation

**Typical Performance**:
- 4096 triangles: ~5 ms per operation
- 32768 triangles: ~40 ms per operation

**Optimization**: Use JAX JIT compilation for significant speedup:
```python
from jax import jit

# JIT-compile operator call
divergence_jit = jit(lambda u, v: divergence(u, v, grid))
div = divergence_jit(u, v)  # Much faster on repeated calls!
```

## Advanced Usage

### Custom Stencils (Structured Grids)

For B-grid and C-grid staggering:

```python
# C-grid: natural divergence (velocities on faces)
grid_c = QuadGrid(name="c_grid", nx=64, ny=64, stagger_type=StaggerType.C)
div = divergence(u_on_x_faces, v_on_y_faces, grid_c)
```

### Edge-Based Operations (Unstructured Grids)

Access edge information for custom operators:

```python
# Get edge data from triangle grid
edges = grid.get_edges()
edge_normals = grid.get_edge_normals()
edge_lengths = grid.get_edge_lengths()

# Implement custom flux calculations
# (see operators/unstructured/divergence_fv.py for examples)
```

### Boundary Conditions

Operators respect grid boundaries:
- **Structured grids**: Stencils near boundaries use available neighbors
- **Unstructured grids**: Boundary edges use one-sided values (no neighbor)

For periodic boundaries:
```python
grid = QuadGrid(name="periodic", nx=64, ny=64, periodic=(True, False))
# Divergence will wrap around in x-direction
```

## Examples

See the examples directory for detailed demonstrations:

- **examples/13_divergence_halo.py**: Divergence with halo exchange
- **examples/14_laplacian_diffusion.py**: Diffusion equation using Laplacian
- **examples/15_gradient_pressure.py**: Pressure gradient forces
- **examples/16_triangle_divergence.py**: **Side-by-side quad vs triangle comparison**

## Extension Guide

### Adding a New Operator

1. **Create structured implementation** in `operators/structured/`:
   ```python
   # operators/structured/new_operator_fd.py
   def new_operator_fd(phi, grid, **kwargs):
       # Finite difference implementation
       pass
   ```

2. **Create unstructured implementation** in `operators/unstructured/`:
   ```python
   # operators/unstructured/new_operator_fv.py
   def new_operator_fv(phi, grid, **kwargs):
       # Finite volume implementation
       pass
   ```

3. **Add dispatch function** in `operators/__init__.py`:
   ```python
   def new_operator(phi, grid, **kwargs):
       if grid.is_structured():
           return new_operator_fd(phi, grid, **kwargs)
       else:
           return new_operator_fv(phi, grid, **kwargs)
   ```

4. **Add tests** in `tests/test_operators_parametrized.py`

5. **Update this README** with usage examples

## References

**Numerical Methods**:
- Finite Difference: LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*
- Finite Volume: Moukalled, F., et al. (2016). *The Finite Volume Method in Computational Fluid Dynamics*

**Arakawa Grids**:
- Arakawa, A., & Lamb, V. R. (1977). *Computational design of the basic dynamical processes of the UCLA general circulation model*

## Version History

- **v0.2.0** (Current): Modular architecture with automatic dispatch
  - Added finite volume operators for unstructured grids
  - Unified API for all grid types
  - Parametrized testing framework

- **v0.1.0**: Initial implementation
  - Finite difference operators for structured grids only
  - A/B/C-grid staggering support

## Contributing

When adding new operators:
1. Implement both structured (FD) and unstructured (FV) versions
2. Add comprehensive tests with analytical solutions
3. Document numerical method and accuracy
4. Provide usage examples
5. Update this README

## Contact

For questions or issues related to operators:
- Open an issue on the project repository
- See main README.md for contact information
