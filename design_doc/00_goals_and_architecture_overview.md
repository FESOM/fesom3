# Goals and Architecture Overview

## Project Vision

FESOMx aims to build an unstructured grid ocean model inspired by FESOM2, with key differences in flexibility and target platforms:

- **Flexible and performant** on different hardware (distributed CPU and GPU nodes)
- **Suitable for contemporary ML-based applications** (differentiable, JAX-native)
- **Performance on distributed CPU and GPU nodes** is a primary metric

## Why FESOMx vs FESOM2

| Aspect | FESOM2 | FESOMx |
|--------|--------|--------|
| Language | Fortran | Python |
| Grid Types | Triangular only | Triangular, Quadrilateral, Mixed |
| Staggering | Arakawa B-grid | Arakawa A, B, C |
| Parallelism | MPI | JAX sharding (GPU/TPU native) |
| ML Integration | Limited | First-class (JAX ecosystem) |

## Technology Stack

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Examples, Visualization)                   │
├─────────────────────────────────────────────────────────┤
│                    Operators Layer                       │
│         (divergence, gradient, laplacian)               │
│              ┌──────────┬──────────┐                    │
│              │ Finite   │ Finite   │                    │
│              │Difference│ Volume   │                    │
│              └──────────┴──────────┘                    │
├─────────────────────────────────────────────────────────┤
│                     Grid Layer                           │
│        (BaseGrid, QuadGrid, TriGrid, MixedGrid)         │
│              + Staggering (A, B, C)                     │
├─────────────────────────────────────────────────────────┤
│                   Backend Layer                          │
│            (JAX Sharding, future: NumPy)                │
│              + Halo Exchange                            │
├─────────────────────────────────────────────────────────┤
│                  Hardware Layer                          │
│           (CPU, GPU, TPU, Multi-node)                   │
└─────────────────────────────────────────────────────────┘
```

### Primary Backend: JAX

- **JAX arrays** on CPU and GPU as primary data backend
- **NumPy arrays** as secondary backend (lower priority)
- **Future consideration**: Numba, PyTorch array backends
- **Precision control**: Single (float32) vs double (float64) precision managed at backend level

### Why JAX?

1. **Hardware portability**: Same code runs on CPU, GPU, TPU
2. **Automatic differentiation**: Enables ML/adjoint applications
3. **JIT compilation**: Performance comparable to compiled languages
4. **Sharding API**: Native multi-device parallelism without MPI boilerplate

## Grid Support

FESOMx supports three grid types to cover different use cases:

| Grid Type | Geometry | Use Case |
|-----------|----------|----------|
| **Triangular** | Unstructured triangles | Complex coastlines, variable resolution (FESOM-like) |
| **Quadrilateral** | Structured quads | Regular domains, efficient finite difference |
| **Mixed** | Triangles + Quads | Transition regions, adaptive refinement |

## Staggering Support

Three Arakawa staggering schemes are supported:

| Scheme | Scalar Location | Velocity Location | Common Usage |
|--------|-----------------|-------------------|--------------|
| **A-grid** | Cell center | Cell center | Simple, but checkerboard issues |
| **B-grid** | Cell center | Cell corners | FESOM2, atmospheric models |
| **C-grid** | Cell center | Cell faces | MOM, NEMO, most ocean models |

## Modular Design Principles

### 1. Operator Modularity

Common mathematical operators are kept modular and dispatched based on grid type:

- Same API: `divergence(u, v, grid)`, `gradient(phi, grid)`, `laplacian(phi, grid)`
- Automatic dispatch: structured grids use FD, unstructured use FV

### 2. Backend Abstraction

Parallel computing backend is abstracted to enable future extensions:

- Current: JAX sharding backend
- Future: JAX MPI, NumPy+MPI, PyTorch

### 3. Halo Exchange Flexibility

Halo exchanges are flexible with respect to halo width:

- Support for higher-order operators (wider stencils)
- Configurable per-direction widths
- Tested for correctness across grid types

## Package Structure

```
fesomx/
├── __init__.py              # Public API exports
├── core/                    # Grid and parallel infrastructure
│   ├── base_grid.py         # Abstract grid interface
│   ├── quad_grid.py         # Structured quadrilateral grid
│   ├── tri_grid.py          # Unstructured triangular grid
│   ├── mixed_grid.py        # Mixed quad/triangle grid
│   ├── staggering.py        # Arakawa A, B, C schemes
│   ├── backend.py           # Backend abstraction
│   ├── jax_sharding_backend.py
│   ├── halo_exchange.py     # Distributed communication
│   └── halo_patterns.py     # Halo width configurations
├── operators/               # Mathematical operators
│   ├── divergence.py        # Unified divergence API
│   ├── gradient.py          # Unified gradient API
│   ├── laplacian.py         # Unified Laplacian API
│   ├── structured/          # Finite difference implementations
│   └── unstructured/        # Finite volume implementations
└── utilities/               # Helper functions
    ├── mesh_generation.py   # Mesh creation
    ├── visualization.py     # Plotting
    └── partitioning.py      # Domain decomposition
```

## Current Limitations

| Component | Current State | Future Plans |
|-----------|---------------|--------------|
| **IO** | npz, pkl formats | NetCDF, Zarr, UGRID |
| **Backends** | JAX sharding only | NumPy, Numba, PyTorch |
| **Visualization** | Basic matplotlib | Interactive, 3D support |
| **Partitioning** | Basic METIS | Advanced load balancing |

## Related Documents

- [01_grids_and_staggering.md](01_grids_and_staggering.md) - Grid hierarchy and staggering details
- [02_operators_and_their_dispatch.md](02_operators_and_their_dispatch.md) - Operator implementation
- [03_backend_abstraction.md](03_backend_abstraction.md) - Backend design pattern
- [04_halo_exchange.md](04_halo_exchange.md) - Distributed communication
- [05_IO.md](05_IO.md) - Input/output utilities
- [06_user_interaction_and_visualization.md](06_user_interaction_and_visualization.md) - User-facing tools
