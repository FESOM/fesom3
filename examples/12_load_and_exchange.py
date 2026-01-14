"""
Example 12: Load Partitions and Perform Halo Exchange

This example demonstrates the simulation workflow:
- Load pre-partitioned grid from disk
- Load partition info and field data
- Setup JAX backend with appropriate device mesh
- Perform halo exchange on distributed data
- Verify and visualize results

This simulates what each MPI process would do in a real distributed application.
"""

import os
from pathlib import Path

# ============================================
# Pre-configuration: Check if we need virtual devices
# ============================================

INPUT_DIR = Path("output/partitions")

# Quick check for number of partitions needed
partition_files = list(INPUT_DIR.glob("partition_*_*parts.pkl"))
if partition_files:
    import pickle
    with open(partition_files[0], 'rb') as f:
        partition_data = pickle.load(f)
    N_PARTITIONS = partition_data['n_partitions']

    # Set XLA_FLAGS BEFORE importing JAX
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N_PARTITIONS}'
else:
    os.environ['JAX_PLATFORMS'] = 'cpu'

# Now import JAX (after environment is configured)
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Mesh, PartitionSpec as P
import numpy as np
import pickle

from fesomx import QuadGrid, TriGrid, StaggerType
from fesomx.core.backend import get_backend
from fesomx.core.halo_exchange import StructuredHaloExchanger, UnstructuredHaloExchanger

print("=" * 70)
print("Example 12: Load Partitions and Perform Halo Exchange")
print("=" * 70)
print()

# ============================================
# Configuration
# ============================================

# Auto-detect grid type from saved files
grid_files = list(INPUT_DIR.glob("grid_*.pkl"))
if not grid_files:
    print("ERROR: No saved grid files found!")
    print(f"Please run example 11 first to create partitions in {INPUT_DIR}")
    sys.exit(1)

GRID_TYPE = grid_files[0].stem.split('_')[1]  # Extract 'quad' or 'tri'

print("Configuration:")
print(f"  Input directory: {INPUT_DIR}")
print(f"  Detected grid type: {GRID_TYPE}")
print(f"  Number of partitions: {N_PARTITIONS}")
print()

# ============================================
# Part 1: Load Grid and Partition
# ============================================

print("PART 1: Load Grid and Partition Info")
print("=" * 70)
print()

# Load grid structure
grid_file = INPUT_DIR / f"grid_{GRID_TYPE}.pkl"
print(f"Loading grid from: {grid_file}")

with open(grid_file, 'rb') as f:
    grid_data = pickle.load(f)

print(f"  Grid name: {grid_data['name']}")
print(f"  Grid type: {grid_data['grid_type']}")
print(f"  Stagger type: {grid_data['stagger_type']}")
print(f"  Periodic: {grid_data['periodic']}")
print()

# Reconstruct grid object
if GRID_TYPE == 'quad':
    # For QuadGrid, we need to infer nx, ny from cells
    cells = grid_data['cells']
    # Assuming cells are (ny, nx) in row-major order
    ny = len(cells)
    nx = len(cells[0]) if ny > 0 else 0

    grid = QuadGrid(
        grid_data['name'],
        nx=nx,
        ny=ny,
        stagger_type=StaggerType(grid_data['stagger_type']),
        periodic=grid_data['periodic']
    )
    grid.vertices = grid_data['vertices']
    grid.cells = grid_data['cells']

    print(f"  Reconstructed QuadGrid: {nx}×{ny}")

elif GRID_TYPE == 'tri':
    grid = TriGrid(
        grid_data['name'],
        stagger_type=StaggerType(grid_data['stagger_type'])
    )
    grid.create_mesh(
        vertices=grid_data['vertices'],
        triangles=grid_data['cells']
    )

    print(f"  Reconstructed TriGrid: {grid.n_cells} cells, {grid.n_vertices} vertices")

print()

# Load partition info (N_PARTITIONS already loaded at top)
partition_file = partition_files[0]
print(f"Loading partition from: {partition_file}")

with open(partition_file, 'rb') as f:
    partition_data = pickle.load(f)

partition_info = partition_data['partition_info']
method = partition_data['method']

print(f"  Partitioning method: {method}")
print()

# Load fields
fields_file = INPUT_DIR / f"fields_{GRID_TYPE}_full.pkl"
print(f"Loading fields from: {fields_file}")

with open(fields_file, 'rb') as f:
    fields = pickle.load(f)

field_names = list(fields.keys())
print(f"  Fields loaded: {field_names}")
print()

# ============================================
# Part 2: Setup JAX Backend
# ============================================

print("PART 2: Setup JAX Backend")
print("=" * 70)
print()

# Get JAX devices (already configured at startup)
devices = jax.devices()
n_devices = len(devices)
print(f"JAX devices configured: {n_devices}")
print(f"Device platforms: {[d.platform for d in devices[:min(4, n_devices)]]}{'...' if n_devices > 4 else ''}")
print()

# Create device mesh
if GRID_TYPE == 'quad':
    # Structured mesh - use 2D device mesh
    mesh_shape = partition_info['mesh_shape']
    print(f"Creating 2D device mesh: {mesh_shape}")
    mesh = Mesh(np.array(devices[:N_PARTITIONS]).reshape(mesh_shape), axis_names=('y', 'x'))
    sharding = NamedSharding(mesh, P('y', 'x'))

elif GRID_TYPE == 'tri':
    # Unstructured mesh - use 1D device array
    print(f"Creating 1D device array: {N_PARTITIONS} devices")
    mesh = Mesh(np.array(devices[:N_PARTITIONS]), axis_names=('devices',))
    sharding = NamedSharding(mesh, P('devices',))

print(f"Device mesh created: {mesh}")
print()

# Initialize backend
backend = get_backend('jax_sharding')
if GRID_TYPE == 'quad':
    backend.initialize(mesh_shape=mesh_shape)
else:
    backend.initialize(mesh_shape=(N_PARTITIONS,))

# ============================================
# Part 3: Distribute Data Across Devices
# ============================================

print("PART 3: Distribute Field Data Across Devices")
print("=" * 70)
print()

# Select first field for halo exchange demo
field_name = field_names[0]
field_data = fields[field_name]

print(f"Working with field: '{field_name}'")
print(f"  Original shape: {field_data.shape}")
print(f"  Min: {field_data.min():.4f}, Max: {field_data.max():.4f}, Mean: {field_data.mean():.4f}")
print()

# Create sharded array
print("Creating sharded array...")
field_jax = jax.device_put(field_data, sharding)

print(f"  Sharded array: {field_jax.shape}")
print(f"  Sharding: {field_jax.sharding}")
print()

# Print per-device statistics
print("Per-device data distribution:")
for part_id in range(min(N_PARTITIONS, 8)):  # Show first 8
    if GRID_TYPE == 'quad':
        j_start, j_end, i_start, i_end = partition_info['device_to_cells'][part_id]
        n_cells = (j_end - j_start) * (i_end - i_start)
    else:
        cell_indices = partition_info['device_to_cells'][part_id]
        n_cells = len(cell_indices)

    print(f"  Device {part_id}: {n_cells} cells")

if N_PARTITIONS > 8:
    print(f"  ... ({N_PARTITIONS - 8} more devices)")
print()

# ============================================
# Part 4: Perform Halo Exchange
# ============================================

print("PART 4: Perform Halo Exchange")
print("=" * 70)
print()

# Create appropriate halo exchanger
if GRID_TYPE == 'quad':
    print("Using StructuredHaloExchanger for QuadGrid...")
    exchanger = StructuredHaloExchanger(backend)
    halo_width = 2
elif GRID_TYPE == 'tri':
    print("Using UnstructuredHaloExchanger for TriGrid...")
    exchanger = UnstructuredHaloExchanger(backend)
    halo_width = 1

print(f"Halo width: {halo_width}")
print()

# Perform exchange
print("Executing halo exchange...")
field_exchanged, diagnostics = exchanger.exchange(
    field_jax,
    partition_info,
    halo_width=halo_width
)

print("✓ Halo exchange complete!")
print()

# ============================================
# Part 5: Analyze Results
# ============================================

print("PART 5: Analyze Halo Exchange Results")
print("=" * 70)
print()

print("Communication Diagnostics:")
print(f"  Strategy: {diagnostics['strategy']}")

if diagnostics['strategy'] == 'structured':
    print(f"  Total cells communicated: {diagnostics['cells_communicated']}")
    print(f"  Exchanges performed:")
    for exch in diagnostics['exchanges']:
        print(f"    - {exch['direction']}: {exch['cells']} cells "
              f"via {exch['method']} ({exch.get('devices', 'N/A')} devices)")

elif diagnostics['strategy'] == 'unstructured':
    print(f"  Total ghost cells: {diagnostics['total_ghost_cells']}")
    print(f"  Ghost cells per device:")
    for dev_id, n_ghosts in list(diagnostics['ghost_cells_per_device'].items())[:8]:
        print(f"    Device {dev_id}: {n_ghosts} ghost cells")
    if N_PARTITIONS > 8:
        print(f"    ... ({N_PARTITIONS - 8} more devices)")

print()

# Verify data integrity
field_after = np.array(field_exchanged)
print("Data integrity check:")
print(f"  Before: min={field_data.min():.4f}, max={field_data.max():.4f}, mean={field_data.mean():.4f}")
print(f"  After:  min={field_after.min():.4f}, max={field_after.max():.4f}, mean={field_after.mean():.4f}")

# Check if data changed (should be same for interior, different for halos in real multi-device)
diff = np.abs(field_after - field_data).max()
print(f"  Max difference: {diff:.6e}")

if diff < 1e-10:
    print("  ℹ️  Data unchanged (expected for single-node shared memory)")
else:
    print("  ✓ Data modified (halos updated)")

print()

# ============================================
# Part 6: Visualization
# ============================================

print("PART 6: Create Visualization")
print("=" * 70)
print()

try:
    import matplotlib.pyplot as plt
    from utilities import plot_field, plot_partition_boundaries, save_plot

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Get partition array for visualization
    if GRID_TYPE == 'tri':
        partition_array = partition_info['partition']
    else:
        partition_array = np.zeros(grid.n_cells, dtype=int)
        for part_id in range(N_PARTITIONS):
            j_start, j_end, i_start, i_end = partition_info['device_to_cells'][part_id]
            nx = grid.nx
            for j in range(j_start, j_end):
                for i in range(i_start, i_end):
                    cell_id = j * nx + i
                    partition_array[cell_id] = part_id

    # Plot 1: Partition boundaries
    plot_partition_boundaries(
        grid.vertices,
        grid.cells,
        partition_array,
        ax=axes[0, 0],
        title=f"{GRID_TYPE.upper()}Grid: {N_PARTITIONS} Partitions",
        show_boundaries=True
    )

    # Plot 2: Field before exchange
    plot_field(
        grid.vertices,
        grid.cells,
        field_data,
        ax=axes[0, 1],
        title=f"Field Before Exchange: {field_name}",
        cmap='RdBu_r'
    )

    # Plot 3: Field after exchange
    plot_field(
        grid.vertices,
        grid.cells,
        field_after,
        ax=axes[1, 0],
        title=f"Field After Exchange: {field_name}",
        cmap='RdBu_r'
    )

    # Plot 4: Difference (should show halo modifications)
    diff_field = np.abs(field_after - field_data)
    plot_field(
        grid.vertices,
        grid.cells,
        diff_field,
        ax=axes[1, 1],
        title=f"Absolute Difference (Halo Updates)",
        cmap='hot'
    )

    plt.tight_layout()

    output_file = INPUT_DIR / f"halo_exchange_{GRID_TYPE}_{N_PARTITIONS}parts.png"
    save_plot(output_file, dpi=150)

    print(f"✓ Saved visualization: {output_file}")
    print()

except ImportError as e:
    print(f"Matplotlib not available ({e}) - skipping visualization")
    print()

# ============================================
# Part 7: Load and Compare Per-Partition Data
# ============================================

print("PART 7: Verify Per-Partition Data Loading")
print("=" * 70)
print()

print("Loading per-partition data files...")
partition_data_loaded = []

for part_id in range(min(N_PARTITIONS, 4)):  # Check first 4
    part_file = INPUT_DIR / f"partition_{part_id}" / "data.pkl"

    if part_file.exists():
        with open(part_file, 'rb') as f:
            part_data = pickle.load(f)

        n_cells = part_data['n_cells']
        part_fields = part_data['fields']

        print(f"  ✓ Partition {part_id}: {n_cells} cells, "
              f"fields: {list(part_fields.keys())}")

        partition_data_loaded.append(part_data)
    else:
        print(f"  ✗ Partition {part_id}: file not found")

if N_PARTITIONS > 4:
    print(f"  ... ({N_PARTITIONS - 4} more partitions)")

print()
print("This demonstrates how each MPI rank would load only its partition!")
print()

# ============================================
# Summary
# ============================================

print("=" * 70)
print("Load and Exchange Complete!")
print("=" * 70)
print()

print("Summary:")
print(f"  ✓ Loaded {GRID_TYPE} grid with {grid.n_cells} cells")
print(f"  ✓ Loaded {N_PARTITIONS} partitions ({method} method)")
print(f"  ✓ Loaded {len(fields)} fields")
print(f"  ✓ Distributed data across {len(devices)} JAX devices")
print(f"  ✓ Performed halo exchange ({diagnostics['strategy']} strategy)")
if diagnostics['strategy'] == 'structured':
    print(f"  ✓ Communicated {diagnostics['cells_communicated']} cells")
else:
    print(f"  ✓ Exchanged {diagnostics['total_ghost_cells']} ghost cells")
print()

print("Workflow demonstrated:")
print("  1. ✓ Load pre-partitioned grid (offline partitioning)")
print("  2. ✓ Reconstruct grid object from saved data")
print("  3. ✓ Setup JAX backend with correct device mesh")
print("  4. ✓ Distribute field data across devices")
print("  5. ✓ Perform halo exchange with ghost cell synchronization")
print("  6. ✓ Verify results and visualize")
print()

print("Next steps for HPC deployment:")
print("  • Each MPI rank loads its own partition_N/data.pkl")
print("  • Use mpi4jax for actual inter-node communication")
print("  • Implement compute kernel between halo exchanges")
print("  • Add I/O for checkpointing and output")
print()
