"""
Example 11: Partition Grid and Save to Disk

This example demonstrates the pre-processing workflow:
- Create a mesh (QuadGrid or TriGrid)
- Partition using METIS (or geometric for QuadGrid)
- Initialize field variables
- Save partition info, grid, and fields to disk

This is typical for HPC workflows where partitioning happens offline
and each compute process loads only its partition.
"""

import numpy as np
import pickle
from pathlib import Path

from fesomx import QuadGrid, TriGrid, StaggerType
from fesomx.utilities import create_structured_triangular_mesh

print("=" * 70)
print("Example 11: Partition Grid and Save to Disk")
print("=" * 70)
print()

# ============================================
# Configuration
# ============================================

# Choose grid type: 'quad' or 'tri'
GRID_TYPE = 'tri'  # Change to 'quad' for structured grid
N_PARTITIONS = 4
OUTPUT_DIR = Path("output/partitions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Configuration:")
print(f"  Grid type: {GRID_TYPE}")
print(f"  Number of partitions: {N_PARTITIONS}")
print(f"  Output directory: {OUTPUT_DIR}")
print()

# ============================================
# Part 1: Create Grid
# ============================================

print("PART 1: Create Grid")
print("=" * 70)
print()

if GRID_TYPE == 'quad':
    # Structured quadrilateral grid
    print("Creating QuadGrid (128×128)...")
    nx, ny = 128, 128
    grid = QuadGrid(
        "ocean_domain",
        nx=nx,
        ny=ny,
        stagger_type=StaggerType.C,  # C-grid for ocean models
        periodic=(True, False)  # Periodic in x, not in y
    )
    grid.create_mesh(domain=(0.0, 360.0, -90.0, 90.0))  # Lon-lat domain

    print(f"  Grid: {nx}×{ny} cells")
    print(f"  Domain: [0°E, 360°E] × [-90°N, 90°N]")
    print(f"  Staggering: C-grid")
    print(f"  Periodic: x=True, y=False")

elif GRID_TYPE == 'tri':
    # Unstructured triangular grid
    print("Creating TriGrid (24×24 base → 1152 triangles)...")
    mesh_data = create_structured_triangular_mesh(nx=24, ny=24)

    grid = TriGrid("ocean_domain", stagger_type=StaggerType.A)
    grid.create_mesh(
        vertices=mesh_data['vertices'],
        triangles=mesh_data['triangles']
    )

    n_triangles = len(mesh_data['triangles'])
    n_vertices = len(mesh_data['vertices'])

    print(f"  Grid: {n_triangles} triangles, {n_vertices} vertices")
    print(f"  Staggering: A-grid")

print()

# ============================================
# Part 2: Partition Grid
# ============================================

print("PART 2: Partition Grid with METIS")
print("=" * 70)
print()

print(f"Partitioning into {N_PARTITIONS} partitions...")
print()

if GRID_TYPE == 'quad':
    # Structured partitioning for QuadGrid
    # Find best factorization of N_PARTITIONS
    if N_PARTITIONS == 4:
        mesh_shape = (2, 2)
    elif N_PARTITIONS == 8:
        mesh_shape = (2, 4)
    elif N_PARTITIONS == 16:
        mesh_shape = (4, 4)
    else:
        # Simple factorization
        import math
        ny_parts = int(math.sqrt(N_PARTITIONS))
        nx_parts = N_PARTITIONS // ny_parts
        mesh_shape = (ny_parts, nx_parts)

    print(f"  Mesh shape: {mesh_shape[0]}×{mesh_shape[1]}")
    partition_info = grid.create_partition_info(mesh_shape=mesh_shape)
    method_used = 'structured'

elif GRID_TYPE == 'tri':
    # Graph partitioning for TriGrid
    try:
        partition_info = grid.create_partition_info(
            n_partitions=N_PARTITIONS,
            method='metis'
        )
        method_used = 'METIS'
        print("  Using METIS graph partitioning")
    except Exception as e:
        print(f"  METIS not available ({e}), using geometric partitioning...")
        partition_info = grid.create_partition_info(
            n_partitions=N_PARTITIONS,
            method='geometric'
        )
        method_used = 'geometric'

print()

# Print partition statistics
print("Partition Statistics:")
print("-" * 60)

for part_id in range(N_PARTITIONS):
    n_cells = len(partition_info['device_to_cells'][part_id])

    if GRID_TYPE == 'tri':
        ghost_map = partition_info['ghost_cell_map']
        n_ghosts = sum(len(cells) for cells in ghost_map[part_id].values())
        n_neighbors = len(ghost_map[part_id])
        overhead = n_ghosts / n_cells * 100 if n_cells > 0 else 0

        print(f"  Partition {part_id}: {n_cells:4d} cells, "
              f"{n_ghosts:3d} ghosts ({overhead:4.1f}% overhead), "
              f"{n_neighbors} neighbors")
    else:
        neighbor_map = partition_info['neighbor_map']
        n_neighbors = len(neighbor_map[part_id])
        print(f"  Partition {part_id}: {n_cells:4d} cells, "
              f"{n_neighbors} neighbors")

print()

# ============================================
# Part 3: Initialize Field Variables
# ============================================

print("PART 3: Initialize Field Variables")
print("=" * 70)
print()

print("Initializing fields on full grid...")
print()

if GRID_TYPE == 'quad':
    # Initialize fields for C-grid ocean model
    # h: sea surface height at cell centers
    # u: zonal velocity at x-faces (edges)
    # v: meridional velocity at y-faces (edges)

    # Cell-centered field (h)
    centers = grid.get_cell_centers()
    x_centers = centers[:, 0].reshape(ny, nx)
    y_centers = centers[:, 1].reshape(ny, nx)

    # Create an interesting pattern (e.g., Gaussian bumps)
    h = np.exp(-((x_centers - 180)**2 + (y_centers - 0)**2) / 5000.0)
    h += 0.5 * np.exp(-((x_centers - 90)**2 + (y_centers - 30)**2) / 3000.0)
    h = h.flatten()

    # Initialize velocities (for C-grid, these would be at faces)
    # For simplicity, use cell-centered here
    u = np.sin(2 * np.pi * x_centers / 360.0) * np.cos(np.pi * y_centers / 180.0)
    v = -np.cos(2 * np.pi * x_centers / 360.0) * np.sin(np.pi * y_centers / 180.0)
    u = u.flatten()
    v = v.flatten()

    fields = {
        'h': h,  # Sea surface height
        'u': u,  # Zonal velocity
        'v': v,  # Meridional velocity
    }

    print("  Initialized C-grid fields:")
    print(f"    h: sea surface height ({h.shape})")
    print(f"    u: zonal velocity ({u.shape})")
    print(f"    v: meridional velocity ({v.shape})")

elif GRID_TYPE == 'tri':
    # Initialize fields for A-grid
    # All variables at cell centers

    centers = grid.get_cell_centers()
    x_centers = centers[:, 0]
    y_centers = centers[:, 1]

    # Temperature field with gradient
    temperature = 20.0 + 10.0 * np.sin(2 * np.pi * x_centers) * np.cos(2 * np.pi * y_centers)

    # Velocity components
    u = np.cos(2 * np.pi * x_centers)
    v = np.sin(2 * np.pi * y_centers)

    # Vorticity
    vorticity = u**2 + v**2

    fields = {
        'temperature': temperature,
        'u': u,
        'v': v,
        'vorticity': vorticity,
    }

    print("  Initialized A-grid fields:")
    print(f"    temperature: {temperature.shape}")
    print(f"    u: {u.shape}")
    print(f"    v: {v.shape}")
    print(f"    vorticity: {vorticity.shape}")

print()

# ============================================
# Part 4: Save to Disk
# ============================================

print("PART 4: Save Partition and Fields to Disk")
print("=" * 70)
print()

# Save grid structure
grid_file = OUTPUT_DIR / f"grid_{GRID_TYPE}.pkl"
with open(grid_file, 'wb') as f:
    pickle.dump({
        'grid_type': GRID_TYPE,
        'vertices': grid.vertices,
        'cells': grid.cells,
        'stagger_type': grid.stagger_type.value,
        'periodic': grid.periodic,
        'name': grid.name,
    }, f)
print(f"✓ Saved grid structure: {grid_file}")

# Save partition info
partition_file = OUTPUT_DIR / f"partition_{GRID_TYPE}_{N_PARTITIONS}parts.pkl"
with open(partition_file, 'wb') as f:
    pickle.dump({
        'n_partitions': N_PARTITIONS,
        'partition_info': partition_info,
        'method': method_used,
    }, f)
print(f"✓ Saved partition info: {partition_file}")

# Save full fields
fields_file = OUTPUT_DIR / f"fields_{GRID_TYPE}_full.pkl"
with open(fields_file, 'wb') as f:
    pickle.dump(fields, f)
print(f"✓ Saved full fields: {fields_file}")

# Optionally: Save per-partition data (for true distributed workflow)
print()
print("Saving per-partition data...")
for part_id in range(N_PARTITIONS):
    part_dir = OUTPUT_DIR / f"partition_{part_id}"
    part_dir.mkdir(exist_ok=True)

    # Get cell indices for this partition
    if GRID_TYPE == 'quad':
        j_start, j_end, i_start, i_end = partition_info['device_to_cells'][part_id]
        cell_indices = []
        for j in range(j_start, j_end):
            for i in range(i_start, i_end):
                cell_indices.append(j * nx + i)
        cell_indices = np.array(cell_indices)
    else:
        cell_indices = np.array(partition_info['device_to_cells'][part_id])

    # Extract fields for this partition
    part_fields = {}
    for field_name, field_data in fields.items():
        part_fields[field_name] = field_data[cell_indices]

    # Save partition-specific data
    part_file = part_dir / "data.pkl"
    with open(part_file, 'wb') as f:
        pickle.dump({
            'partition_id': part_id,
            'cell_indices': cell_indices,
            'fields': part_fields,
            'n_cells': len(cell_indices),
        }, f)

    print(f"  ✓ Partition {part_id}: {len(cell_indices)} cells → {part_file}")

print()

# ============================================
# Part 5: Visualization (Optional)
# ============================================

print("PART 5: Create Visualization")
print("=" * 70)
print()

try:
    import matplotlib.pyplot as plt
    from utilities import plot_field, plot_partition_boundaries, save_plot

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Partition boundaries
    if GRID_TYPE == 'tri':
        partition_array = partition_info['partition']
    else:
        partition_array = np.zeros(grid.n_cells, dtype=int)
        for part_id in range(N_PARTITIONS):
            j_start, j_end, i_start, i_end = partition_info['device_to_cells'][part_id]
            for j in range(j_start, j_end):
                for i in range(i_start, i_end):
                    cell_id = j * nx + i
                    partition_array[cell_id] = part_id

    plot_partition_boundaries(
        grid.vertices,
        grid.cells,
        partition_array,
        ax=axes[0],
        title=f"{GRID_TYPE.upper()}Grid: {N_PARTITIONS} Partitions ({method_used})",
        show_boundaries=True
    )

    # Plot 2: First field
    first_field_name = list(fields.keys())[0]
    first_field_data = fields[first_field_name]

    plot_field(
        grid.vertices,
        grid.cells,
        first_field_data,
        ax=axes[1],
        title=f"Initial Field: {first_field_name}",
        cmap='RdBu_r'
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"partition_{GRID_TYPE}_{N_PARTITIONS}parts.png"
    save_plot(output_file, dpi=150)

    print(f"✓ Saved visualization: {output_file}")
    print()

except ImportError:
    print("Matplotlib not available - skipping visualization")
    print()

# ============================================
# Summary
# ============================================

print("=" * 70)
print("Partition and Save Complete!")
print("=" * 70)
print()

print("Summary:")
print(f"  Grid type: {GRID_TYPE}")
print(f"  Partitions: {N_PARTITIONS}")
print(f"  Partitioning method: {method_used}")
print(f"  Total cells: {grid.n_cells}")
print(f"  Fields saved: {list(fields.keys())}")
print()

print("Files created:")
print(f"  📁 {OUTPUT_DIR}/")
print(f"    ├── grid_{GRID_TYPE}.pkl")
print(f"    ├── partition_{GRID_TYPE}_{N_PARTITIONS}parts.pkl")
print(f"    ├── fields_{GRID_TYPE}_full.pkl")
print(f"    └── partition_0/ ... partition_{N_PARTITIONS-1}/")
print(f"          └── data.pkl (per-partition field data)")
print()

print("Next step:")
print(f"  Run: python examples/12_load_and_exchange.py")
print()
