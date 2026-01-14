# Examples

Comprehensive examples demonstrating JAX grids functionality.

## Quick Reference

| Example | Topic | Grid Type | Key Features |
|---------|-------|-----------|--------------|
| 01 | Basic grids | Quad/Tri/Mixed | Mesh creation, visualization |
| 02 | Staggering | Quad | A/B/C-grid variable placement |
| 03 | I/O operations | Quad/Tri | Save/load grids and fields |
| 04 | Field operations | Quad | Gradients, interpolation, stencils |
| 05 | Unstructured | Tri | Triangle meshes, connectivity |
| 06 | Parallel scaling | Quad | Multi-CPU, sharding, performance |
| 07 | Halo visualization | Quad/Tri | Educational partition concepts |
| 08 | Correctness tests | Quad/Tri | Partition validation |
| 09 | Partition viz | Quad/Tri | Ghost cells, communication graph |
| 10 | Data movement | Quad | Real cross-device transfer |
| 11 | Partition & save | Quad/Tri | **METIS partitioning, save to disk** |
| 12 | Load & exchange | Quad/Tri | **Load partitions, distributed halo** |
| **13** | **Divergence operator** | **Quad** | **∇·u with halo exchange** ✨ |
| **14** | **Laplacian operator** | **Quad** | **∇²φ for diffusion** ✨ |
| **15** | **Gradient operator** | **Quad** | **∇φ for pressure forces** ✨ |
| **16** | **Triangle divergence** | **Quad+Tri** | **Modular architecture demo** ✨ |

## Running Examples

```bash
# Run single example
python examples/01_basic_grids.py

# Run all examples
for i in {01..10}; do python examples/${i}_*.py; done

# Using bash script
./scripts/run_example.sh 06
```

## Detailed Descriptions

### 01_basic_grids.py
**Introduction to grid types**

Creates and visualizes the three grid types:
- QuadGrid: 20×20 rectangular mesh
- TriGrid: Structured triangular mesh
- MixedGrid: 70% quads + 30% triangles

**Output:**
- `output/quad_grid.png`
- `output/tri_grid.png`
- `output/mixed_grid.png`

**Learning goals:**
- Grid creation API
- Basic visualization
- Mesh properties (cells, vertices, edges)

---

### 02_staggering.py
**Arakawa staggering schemes**

Demonstrates variable placement on A, B, and C grids:
- A-grid: h, u, v all at centers
- B-grid: h at centers, u, v at corners
- C-grid: h at centers, u at x-faces, v at y-faces

**Output:**
- `output/stagger_A.png`
- `output/stagger_B.png`
- `output/stagger_C.png`

**Learning goals:**
- Staggering concepts
- Variable location mapping
- Grid indexing for different stagger types

---

### 03_io_operations.py
**Saving and loading grids**

Tests serialization of:
- Grid structure (vertices, cells, connectivity)
- Field data (cell-centered and vertex-centered)
- Multiple fields (temperature, velocity)

**Output:**
- `output/quad_grid.pkl`
- `output/temperature_field.pkl`

**Learning goals:**
- Grid persistence
- Field I/O operations
- Data formats

---

### 04_field_operations.py
**Field computations on grids**

Implements common operations:
- Gradient calculation (∇φ)
- Laplacian (∇²φ)
- Interpolation (cell→vertex, vertex→cell)
- 5-point stencil operations

**Uses:** Gaussian field for testing

**Learning goals:**
- Finite difference stencils
- Field transformations
- Numerical operators

---

### 05_unstructured_grids.py
**Triangular mesh handling**

Advanced unstructured grid features:
- Create triangle mesh from connectivity
- Build edge lists
- Find cell neighbors
- Identify boundary cells
- Compute cell areas and centroids

**Output:**
- `output/triangle_mesh.png`
- `output/triangle_neighbors.png`

**Learning goals:**
- Unstructured data structures
- Topological queries
- Neighbor finding algorithms

---

### 06_parallel_scaling.py
**Multi-device parallelism with JAX**

Demonstrates JAX sharding across CPUs:
- 32 virtual CPU devices
- 4×8 device mesh configuration
- Array sharding and distribution
- Shard size calculation

**Configuration:**
```python
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=32'
os.environ['JAX_PLATFORMS'] = 'cpu'
```

**Output:**
- Device information
- Shard dimensions
- Memory distribution

**Learning goals:**
- JAX device management
- Sharding strategies
- Performance considerations

---

### 07_halo_visualization.py
**Educational explanation of partitioning**

Visual guide to partition concepts:
- QuadGrid: 2×2 rectangular partition
- Neighbor connectivity visualization
- Halo region identification
- ASCII art partition diagrams

**Output:**
- Detailed text explanations
- Neighbor maps
- Overhead calculations

**Learning goals:**
- What is partitioning?
- Why halos are needed
- Structured vs unstructured strategies

---

### 08_halo_correctness.py
**Partition validation tests**

Verifies partitioning correctness:
- QuadGrid: All cells assigned uniquely
- QuadGrid: Neighbor connectivity correct
- TriGrid: METIS/geometric partitioning
- TriGrid: Ghost cell mappings valid
- Load balance analysis

**Output:**
- Pass/fail test results
- Partition statistics
- Imbalance metrics

**Learning goals:**
- Partition completeness
- Ghost cell correctness
- Load balancing

---

### 09_partition_visualization.py
**Visual comparison of partition strategies**

Creates comprehensive partition visualizations:

**QuadGrid (2×2 partition):**
- Partition boundaries (black lines)
- Communication graph (grid layout)
- Device assignment heatmap
- Edge-based exchange explanation

**TriGrid (4 partitions):**
- Irregular partition boundaries
- Communication graph (spring layout)
- Ghost cells for devices 0 and 1
- Scattered ghost cell visualization

**Output:**
- `output/partition_quad_visualization.png` (4-panel)
- `output/partition_tri_visualization.png` (4-panel)

**Plots:**
1. Partition boundaries with colored regions
2. Communication graph showing device connectivity
3. Ghost cells (owned vs needed)
4. Device assignment / statistics

**Learning goals:**
- Visual partition understanding
- Communication patterns
- Ghost cell distribution
- Structured vs unstructured differences

---

### 11_partition_and_save.py
**METIS partitioning and persistence**

Demonstrates graph partitioning and saving partitioned grids:
- METIS library for optimal load balancing
- Geometric fallback for when METIS unavailable
- Save partition metadata to disk
- Partition quality metrics

**Output:**
- Partition files in `output/partitions/`
- Load balance statistics
- Communication volume analysis

**Learning goals:**
- Graph partitioning algorithms
- Partition quality metrics
- Saving distributed data

---

### 12_load_and_exchange.py
**Loading and distributed halo exchange**

Loads saved partitions and performs halo exchange:
- Load partition files from disk
- Reconstruct communication topology
- Perform distributed halo exchange
- Verify correctness

**Learning goals:**
- Restoring partitioned state
- Checkpoint/restart workflows
- Distributed operation validation

---

### 13_divergence_halo.py ✨ NEW
**Divergence operator with halo exchange**

Demonstrates WHY halo exchange is needed for stencil operations:

**Setup:**
- 64×64 QuadGrid, 2×2 partition
- Divergence-free velocity field (∇·u = 0 analytically)
- Sine wave pattern

**Tests:**
1. Divergence WITHOUT halo exchange → errors at boundaries
2. Divergence WITH halo exchange → correct everywhere

**Results:**
- Without halos: large errors at partition boundaries
- With halos: errors at machine precision
- Improvement factor: typically 100-1000×

**Output:**
- `output/example_13_divergence_halo.png`
- Side-by-side comparison with/without halos
- Error visualization

**Learning goals:**
- Why stencil operations need halos
- Impact on boundary accuracy
- Proper distributed computing

---

### 14_laplacian_diffusion.py ✨ NEW
**Laplacian operator for diffusion**

Simulates 2D heat diffusion using the Laplacian operator:

**Equation:**
```
∂φ/∂t = κ∇²φ
```

**Setup:**
- 128×128 QuadGrid
- Initial Gaussian temperature distribution
- Time integration with forward Euler
- Periodic boundary conditions

**Features:**
- 2nd order Laplacian (5-point stencil)
- 4th order Laplacian (9-point stencil) option
- Time evolution visualization
- Energy conservation tracking

**Output:**
- `output/example_14_laplacian_diffusion.png`
- Evolution sequence (t=0, 25, 50, 100 steps)
- Convergence analysis

**Learning goals:**
- Laplacian operator usage
- PDE time integration
- Accuracy vs order trade-offs

---

### 15_gradient_pressure.py ✨ NEW
**Gradient operator for pressure forces**

Demonstrates pressure gradient calculation for fluid dynamics:

**Equation:**
```
F = -∇p  (pressure gradient force)
```

**Setup:**
- 64×64 QuadGrid with A-grid staggering
- Gaussian pressure distribution (high at center)
- Compute gradient → pressure force field
- Compare to analytical solution

**Features:**
- Central difference gradient
- Vector field visualization
- Accuracy validation
- Staggered grid considerations

**Output:**
- `output/example_15_gradient_pressure.png`
- Pressure field, gradient components
- Vector field (quiver plot)
- Error distribution

**Learning goals:**
- Gradient operator usage
- Pressure force computation
- Numerical vs analytical comparison

---

### 16_triangle_divergence.py ✨ NEW
**Divergence on triangular grids**

**MOST IMPORTANT NEW EXAMPLE** - Demonstrates modular operator architecture!

**Key Demonstration:**
- Creates BOTH QuadGrid and TriGrid for same domain
- Uses SAME divergence() API for both
- Compares finite difference (quad) vs finite volume (triangle)
- Side-by-side visualization

**Setup:**
- 32×32 QuadGrid (structured)
- 32×32 TriGrid (2 triangles per quad = 2048 triangles)
- Solid body rotation velocity field (divergence-free)

**Automatic Dispatch:**
```python
# Same function, different implementations!
div_quad = divergence(u_quad, v_quad, quad_grid)  # → divergence_fd()
div_tri = divergence(u_tri, v_tri, tri_grid)      # → divergence_fv()
```

**Results:**
- QuadGrid: max error ~0 (machine precision)
- TriGrid: max error ~1e-14 (also excellent!)
- Both correctly identify divergence-free field

**Output:**
- `output/example_16_triangle_divergence.png`
- 3×3 subplot grid:
  - Row 1: QuadGrid velocity components + vector field
  - Row 2: TriGrid velocity components + mesh structure
  - Row 3: Divergence comparison + error histogram

**Learning goals:**
- **Modular architecture works!**
- Same API for structured and unstructured grids
- Finite difference vs finite volume methods
- Triangle mesh visualization
- Method accuracy comparison

---

### 10_data_movement.py
**Verification of actual data transfer**

**Most important example** - proves data moves between devices!

**Setup:**
- 8 devices in 2×4 mesh
- 64×64 interior grid + 2-cell halos
- Each device's interior initialized with its device ID (0-7)
- Halos initialized to -1 (uninitialized)

**Execution:**
- Perform halo exchange
- Verify ghost cells filled with neighbor IDs
- Show communication diagnostics

**Results:**
- 544 cells communicated (272 x-direction + 272 y-direction)
- Method: cross-device-slicing
- Halos correctly filled with neighbor data

**Output:**
- `output/data_movement_visualization.png` (before/after comparison)
- Detailed verification report

**Learning goals:**
- Proof of cross-device communication
- Halo exchange mechanics
- JAX automatic data transfer
- Communication volume analysis

## Example Progression

**Recommended order:**

1. **Basics** (01-03): Understand grid types, staggering, I/O
2. **Operations** (04-05): Field operations, unstructured grids
3. **Parallelism** (06): Multi-device setup
4. **Partitioning** (07-09): Concepts, validation, visualization
5. **Communication** (10-12): Data movement, partition save/load
6. **Operators** (13-16): Mathematical operators ✨ NEW
   - Start with 13: See why halos matter for operators
   - Try 14-15: Explore gradient and Laplacian
   - **Must see 16**: Triangle grid support (new architecture!)

## Hardware Requirements

- **Examples 01-05**: Single CPU sufficient
- **Examples 06-10**: Multi-core CPU recommended (uses 8-32 virtual devices)
- **GPU**: Optional (set `JAX_PLATFORMS='gpu'` to use)

## Environment Setup

```bash
# Activate conda environment
conda activate jax081

# Install optional dependency for better triangle partitioning
pip install pymetis

# Set environment variables (if needed)
export JAX_PLATFORMS=cpu
export XLA_FLAGS='--xla_force_host_platform_device_count=32'
```

## Troubleshooting

**Import errors:**
- Ensure `PYTHONPATH` includes project root
- Or run from project root: `python examples/XX_*.py`

**JAX detects GPU instead of CPU:**
- Set `os.environ['JAX_PLATFORMS'] = 'cpu'` before importing JAX

**METIS not available:**
- Install: `pip install pymetis`
- Fallback: Examples automatically use geometric partitioning

**Matplotlib errors:**
- Install: `pip install matplotlib`
- Or set `MATPLOTLIB_AVAILABLE = False` to skip plots

## Output Directory

All examples save output to `output/` (auto-created):
- `*.png` - Visualizations
- `*.pkl` - Serialized grids/fields

## Performance Benchmarking

Use example 06 as template for scaling studies:
- Vary `mesh_shape` for strong scaling
- Vary `grid_size` for weak scaling
- Measure communication overhead
- Profile halo exchange cost

## Next Steps

After running examples:
- Explore `tests/` for unit tests
- Read `core/README.md` for API details
- Modify examples for your use case
- Check `utilities/` for helper functions

## Contributing Examples

To add a new example:
1. Use numbering: `11_your_topic.py`
2. Include docstring explaining purpose
3. Add output directory creation
4. Follow structure: Setup → Computation → Visualization → Summary
5. Update this README with entry
