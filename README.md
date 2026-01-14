# FESOMx: Parallel Grid Computations with Halo Exchange

A JAX-based framework for parallel halo exchange operations on structured and unstructured grids with different staggering schemes for ocean/climate modeling.

## Overview

FESOMx provides infrastructure for:
- Testing JAX sharding for halo exchanges on different grid types
- Supporting quadrilateral, triangular, and mixed grids
- Implementing Arakawa A, B, C staggering schemes
- Benchmarking halo exchange patterns and widths
- Extensible backend architecture (JAX sharding → JAX MPI → NumPy+MPI)

**Target Application**: Ocean/climate modeling with structured and unstructured grids.

## Features

- **Multiple Grid Types**:
  - Structured quadrilateral grids (`QuadGrid`)
  - Unstructured triangular grids (`TriGrid`)
  - Mixed quadrilateral/triangular grids (`MixedGrid`)

- **Staggering Schemes**:
  - A-grid: All variables at cell centers
  - B-grid: Velocities at corners, scalars at centers
  - C-grid: Velocities at faces, scalars at centers

- **Mathematical Operators**:
  - **Divergence**: ∇·u (velocity divergence)
  - **Gradient**: ∇φ (scalar gradient)
  - **Laplacian**: ∇²φ (scalar Laplacian)
  - **Automatic dispatch**: Same API for all grid types
    - Structured grids → Finite difference (2nd/4th order)
    - Unstructured grids → Finite volume (edge-based fluxes)
  - **JAX-native**: JIT-compilable, GPU-accelerated

- **Flexible Halo Exchange**:
  - Uniform, asymmetric, and directional halo widths
  - Multiple halo patterns (star, box, extended)
  - Periodic boundary conditions

## Installation

### Requirements

- Python >= 3.10
- JAX >= 0.8.1
- CUDA 13 (for GPU support, optional)

### Using Virtual Environment (Recommended)

```bash
# Clone or navigate to the repository
cd fesom3

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install package in editable mode
pip install -e .
```

### Install with Optional Dependencies

```bash
# Install with development tools (pytest, black, ruff, mypy)
pip install -e ".[dev]"

# Install with CUDA support
pip install -e ".[cuda]"

# Install with full dependencies (uxarray, pymetis)
pip install -e ".[full]"

# Install everything
pip install -e ".[dev,cuda,full]"
```

### Using Conda

```bash
# Create conda environment
conda create -n fesomx python=3.12
conda activate fesomx

# Install JAX with CUDA (if using GPU)
pip install jax[cuda13]

# Install package
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Test basic imports
python -c "from fesomx import QuadGrid, divergence; print('OK')"

# Test all imports
python -c "
from fesomx import QuadGrid, TriGrid, MixedGrid, StaggerType
from fesomx import divergence, gradient, laplacian
from fesomx import get_backend
print('All imports successful!')
"
```

## Quick Start

### Basic Usage

```python
from fesomx import QuadGrid, StaggerType, divergence, gradient, laplacian
import jax.numpy as jnp
import numpy as np

# Create a grid
grid = QuadGrid(
    name="my_grid",
    nx=32, ny=32,
    stagger_type=StaggerType.A
)
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
grid.dx = 1.0 / 31
grid.dy = 1.0 / 31

# Compute divergence
u = jnp.ones((32, 32))
v = jnp.zeros((32, 32))
div = divergence(u, v, grid)

# Compute gradient
phi = jnp.sin(jnp.linspace(0, 2*jnp.pi, 32))[:, None] * jnp.ones((1, 32))
grad_x, grad_y = gradient(phi, grid)

# Compute Laplacian
lapl = laplacian(phi, grid)

print(f"Divergence shape: {div.shape}")
print(f"Gradient shapes: {grad_x.shape}, {grad_y.shape}")
print(f"Laplacian shape: {lapl.shape}")
```

### Running Examples

```bash
# Run example directly
python examples/01_simple_quad_halo.py

# Run with GPU
JAX_PLATFORMS=gpu python examples/01_simple_quad_halo.py

# Available examples (01-17)
python examples/16_triangle_divergence.py   # Triangle grid operators
python examples/17_triangle_divergence_halo.py  # Distributed triangle grid
```

## Testing

### Run All Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with verbose output
pytest tests/ -v --tb=short
```

### Run Specific Tests

```bash
# Run specific test file
pytest tests/test_grids.py -v

# Run specific test class
pytest tests/test_grids.py::TestQuadGrid -v

# Run tests matching pattern
pytest tests/ -v -k "divergence"
pytest tests/ -v -k "quad"
```

### Run with Coverage

```bash
# Generate coverage report
pytest tests/ -v --cov=fesomx --cov-report=html

# View report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Tests in Parallel

```bash
# Requires pytest-xdist
pip install pytest-xdist
pytest tests/ -v -n auto
```

### Test Markers

```bash
# Skip slow tests
pytest tests/ -v -m "not slow"

# Run only GPU tests
pytest tests/ -v -m "gpu"

# Run only CPU tests
pytest tests/ -v -m "cpu"
```

## Project Structure

```
fesom3/
├── fesomx/                      # Main package
│   ├── __init__.py              # Top-level exports
│   ├── py.typed                 # PEP 561 type marker
│   ├── core/                    # Core grid and parallel computing
│   │   ├── backend.py           # Backend abstraction
│   │   ├── jax_sharding_backend.py
│   │   ├── base_grid.py         # Abstract grid class
│   │   ├── quad_grid.py         # Quadrilateral grid
│   │   ├── tri_grid.py          # Triangular grid
│   │   ├── mixed_grid.py        # Mixed grid
│   │   ├── staggering.py        # Staggering schemes
│   │   ├── halo_patterns.py     # Halo configurations
│   │   └── halo_exchange.py     # Halo exchange operations
│   ├── operators/               # Mathematical operators
│   │   ├── divergence.py        # Divergence operator
│   │   ├── gradient.py          # Gradient operator
│   │   ├── laplacian.py         # Laplacian operator
│   │   ├── structured/          # Finite difference methods
│   │   └── unstructured/        # Finite volume methods
│   └── utilities/               # Helper utilities
│       ├── mesh_generation.py   # Mesh creation
│       ├── visualization.py     # Plotting functions
│       └── partitioning.py      # Grid partitioning
├── tests/                       # Test suite (220+ tests)
├── examples/                    # Example scripts (17 examples)
├── scripts/                     # Bash execution scripts
├── configs/                     # Configuration files
├── pyproject.toml               # Package configuration
├── CLAUDE.md                    # Development guide
└── README.md                    # This file
```

## Examples Overview

### Grid and Halo Exchange (01-12)
1. **01_simple_quad_halo.py**: Basic quadrilateral grid with halo exchange
2. **02_triangle_halo.py**: Unstructured triangular mesh
3. **03_mixed_grid_halo.py**: Mixed quad/triangle grid
4. **04_staggered_grid.py**: Arakawa A, B, C staggering
5. **05_multi_width_halo.py**: Variable halo widths
6. **06_parallel_scaling.py**: Multi-device parallelism
7. **07_halo_visualization.py**: Partitioning visualization
8. **08_halo_correctness.py**: Partition validation
9. **09_partition_visualization.py**: Partition strategies
10. **10_data_movement.py**: Cross-device data transfer
11. **11_partition_and_save.py**: METIS partitioning
12. **12_load_and_exchange.py**: Distributed halo exchange

### Mathematical Operators (13-17)
13. **13_divergence_halo.py**: Divergence with halo exchange
14. **14_laplacian_diffusion.py**: Heat diffusion simulation
15. **15_gradient_pressure.py**: Pressure gradient force
16. **16_triangle_divergence.py**: Triangle grid operators
17. **17_triangle_divergence_halo.py**: Distributed triangle computation

## Dependencies

### Core (installed automatically)
- `jax>=0.8.1`, `jaxlib>=0.8.1` - Parallel computing
- `numpy>=1.24.0` - Array operations
- `scipy>=1.10.0` - Spatial algorithms
- `xarray>=2023.1.0` - Data structures
- `matplotlib>=3.7.0` - Visualization

### Optional Extras

| Extra | Dependencies | Use Case |
|-------|-------------|----------|
| `[cuda]` | `jax[cuda13]` | GPU acceleration |
| `[full]` | `uxarray`, `pymetis` | Unstructured grids, graph partitioning |
| `[dev]` | `pytest`, `black`, `ruff`, `mypy` | Development and testing |

## Performance Tips

- **Grid size**: Use grid sizes divisible by device count for optimal sharding
- **Memory**: Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to avoid GPU memory preallocation
- **Halo width**: Larger halos increase communication overhead
- **JIT compilation**: First run is slower due to compilation; subsequent runs are fast

## HPC Usage

For running on HPC clusters with Slurm:

```bash
# Edit scripts/submit_slurm.sh for your cluster
sbatch scripts/submit_slurm.sh examples/01_simple_quad_halo.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run `pytest tests/ -v` to ensure all tests pass
5. Submit a pull request

## License

MIT

## Contact

AWI Team - Alfred Wegener Institute

## Citation

If you use this code in your research, please cite:

```
TBD
```
