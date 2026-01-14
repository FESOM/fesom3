"""
Example 09: Partition and Ghost Cell Visualization

This example demonstrates:
- Visual representation of mesh partitioning
- Partition boundary highlighting
- Ghost cell identification and visualization
- Communication graph between devices
- Comparison of structured vs unstructured partitioning
"""

import numpy as np
import sys
from pathlib import Path

from fesomx import QuadGrid, TriGrid, StaggerType
from fesomx.utilities import (
    create_structured_triangular_mesh,
    plot_partition_boundaries,
    plot_ghost_cells,
    plot_communication_graph,
    save_plot,
)

print("=" * 70)
print("Example 09: Partition and Ghost Cell Visualization")
print("=" * 70)
print()

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    print("Matplotlib not available - visualization disabled")
    MATPLOTLIB_AVAILABLE = False
    sys.exit(0)

# ============================================
# Part 1: Structured Grid Partitioning
# ============================================

print("PART 1: Structured QuadGrid Partitioning Visualization")
print("=" * 70)
print()

print("1. Creating 32×32 QuadGrid with 2×2 partition...")
print()

nx, ny = 32, 32
grid_quad = QuadGrid(
    "quad_viz",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,
    periodic=(False, False)
)

grid_quad.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))

# Create partition info for 2×2 mesh (4 devices)
partition_info_quad = grid_quad.create_partition_info(mesh_shape=(2, 2))

# Create partition array for visualization
partition_quad = np.zeros(nx * ny, dtype=int)
for device_id, (j_start, j_end, i_start, i_end) in partition_info_quad['device_to_cells'].items():
    for j in range(j_start, j_end):
        for i in range(i_start, i_end):
            cell_id = j * nx + i
            partition_quad[cell_id] = device_id

print("   Creating visualizations...")
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Plot 1: Partition boundaries
plot_partition_boundaries(
    grid_quad.vertices,
    grid_quad.cells,
    partition_quad,
    ax=axes[0, 0],
    title="QuadGrid: 4 Partitions with Boundaries",
    show_boundaries=True
)

# Plot 2: Communication graph
plot_communication_graph(
    partition_info_quad,
    ax=axes[0, 1],
    title="QuadGrid: Communication Graph (2×2 mesh)",
    layout='grid'
)

# Plot 3 & 4: Ghost cells for devices 0 and 3
# Note: QuadGrid doesn't use ghost_cell_map, but we can visualize owned regions
axes[1, 0].text(0.5, 0.5, 'QuadGrid uses\nEdge-based Exchange\n(No scattered ghost cells)',
               ha='center', va='center', fontsize=14,
               transform=axes[1, 0].transAxes)
axes[1, 0].set_title("QuadGrid: Structured Halo Exchange")
axes[1, 0].axis('off')

# Show partition assignment
partition_2d = partition_quad.reshape(ny, nx)
im = axes[1, 1].imshow(partition_2d, cmap='tab10', vmin=0, vmax=3, origin='lower')
axes[1, 1].set_title("QuadGrid: Device Assignment")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y")
plt.colorbar(im, ax=axes[1, 1], label='Device ID')

# Add grid lines showing partition boundaries
axes[1, 1].axvline(nx//2 - 0.5, color='black', linewidth=2)
axes[1, 1].axhline(ny//2 - 0.5, color='black', linewidth=2)

plt.tight_layout()
output_file = "output/partition_quad_visualization.png"
Path("output").mkdir(exist_ok=True)
save_plot(output_file, dpi=150)

print(f"   Saved: {output_file}")
print()

# ============================================
# Part 2: Unstructured Grid Partitioning
# ============================================

print("\nPART 2: Unstructured TriGrid Partitioning Visualization")
print("=" * 70)
print()

print("1. Creating triangular mesh (12×12 quads → 288 triangles)...")
print()

mesh_data = create_structured_triangular_mesh(nx=12, ny=12)
n_triangles = len(mesh_data['triangles'])

grid_tri = TriGrid("tri_viz", stagger_type=StaggerType.A)
grid_tri.create_mesh(
    vertices=mesh_data['vertices'],
    triangles=mesh_data['triangles']
)

print(f"   Triangles: {n_triangles}")
print()

# Create partition info
print("2. Partitioning triangular mesh into 4 partitions...")
print()

try:
    partition_info_tri = grid_tri.create_partition_info(
        n_partitions=4,
        method='metis'
    )
    method_used = 'METIS'
except Exception as e:
    print(f"   METIS not available ({e}), using geometric partitioning...")
    partition_info_tri = grid_tri.create_partition_info(
        n_partitions=4,
        method='geometric'
    )
    method_used = 'Geometric'

print(f"   Method: {method_used}")
print()

partition_tri = partition_info_tri['partition']
ghost_map_tri = partition_info_tri['ghost_cell_map']

print("   Creating visualizations...")
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 16))

# Plot 1: Partition boundaries
plot_partition_boundaries(
    grid_tri.vertices,
    grid_tri.cells,
    partition_tri,
    ax=axes[0, 0],
    title=f"TriGrid: 4 Partitions ({method_used}) with Boundaries",
    show_boundaries=True
)

# Plot 2: Communication graph
plot_communication_graph(
    partition_info_tri,
    ax=axes[0, 1],
    title="TriGrid: Communication Graph",
    layout='spring'
)

# Plot 3 & 4: Ghost cells for devices 0 and 1
plot_ghost_cells(
    grid_tri.vertices,
    grid_tri.cells,
    partition_tri,
    device_id=0,
    ghost_cell_map=ghost_map_tri,
    ax=axes[1, 0],
    title="TriGrid: Ghost Cells for Device 0"
)

plot_ghost_cells(
    grid_tri.vertices,
    grid_tri.cells,
    partition_tri,
    device_id=1,
    ghost_cell_map=ghost_map_tri,
    ax=axes[1, 1],
    title="TriGrid: Ghost Cells for Device 1"
)

plt.tight_layout()
output_file = "output/partition_tri_visualization.png"
save_plot(output_file, dpi=150)

print(f"   Saved: {output_file}")
print()

# ============================================
# Part 3: Comparison Statistics
# ============================================

print("\nPART 3: Comparison Statistics")
print("=" * 70)
print()

print("Structured (QuadGrid) Partitioning:")
print("-" * 40)
print(f"  Method: Regular grid decomposition")
print(f"  Partitions: 4 (2×2 mesh)")
print(f"  Cells per partition: {nx*ny//4} (perfectly balanced)")
print(f"  Communication pattern: Directional (L/R/T/B)")
print(f"  Ghost cells: Contiguous edges")
print()

print("Unstructured (TriGrid) Partitioning:")
print("-" * 40)
print(f"  Method: {method_used}")
print(f"  Partitions: 4")

cells_per_part = [len(partition_info_tri['device_to_cells'][i]) for i in range(4)]
min_cells = min(cells_per_part)
max_cells = max(cells_per_part)
avg_cells = np.mean(cells_per_part)

print(f"  Cells per partition: {min_cells}-{max_cells} (avg: {avg_cells:.1f})")

total_ghosts = sum(
    sum(len(cells) for cells in neighbors.values())
    for neighbors in ghost_map_tri.values()
)
print(f"  Total ghost cells: {total_ghosts}")
print(f"  Avg ghost cells per device: {total_ghosts/4:.1f}")

# Communication volume
print()
print("  Communication requirements per device:")
for device_id in range(4):
    owned = len(partition_info_tri['device_to_cells'][device_id])
    ghosts_needed = sum(len(cells) for cells in ghost_map_tri[device_id].values())
    overhead = ghosts_needed / owned * 100 if owned > 0 else 0

    print(f"    Device {device_id}: {owned} owned, {ghosts_needed} ghosts needed ({overhead:.1f}% overhead)")

print()

# ============================================
# Part 4: Key Insights
# ============================================

print("\nPART 4: Key Visual Insights")
print("=" * 70)
print()

print("From the visualizations you can see:")
print()

print("1. PARTITION BOUNDARIES (Top-left plots):")
print("   • QuadGrid: Clean rectangular boundaries")
print("   • TriGrid: Irregular boundaries following mesh topology")
print("   • Black lines show where different partitions meet")
print()

print("2. COMMUNICATION GRAPH (Top-right plots):")
print("   • QuadGrid: Regular 2×2 grid pattern, each device has 2-3 neighbors")
print("   • TriGrid: Potentially all-to-all communication")
print("   • Edge labels show number of cells to communicate")
print()

print("3. GHOST CELLS (Bottom plots):")
print("   • QuadGrid: Uses edge-based exchange (no scattered ghosts)")
print("   • TriGrid: Green cells = ghost cells needed from other devices")
print("   • Yellow cells = owned by this device")
print("   • Red cells = owned by other devices")
print()

print("4. LOAD BALANCE:")
print(f"   • QuadGrid: Perfect (all devices have exactly {nx*ny//4} cells)")
print(f"   • TriGrid: {method_used} balancing ({min_cells}-{max_cells} cells)")
print()

# ============================================
# Summary
# ============================================

print("=" * 70)
print("Visualization Complete!")
print("=" * 70)
print()

print("Output files created:")
print("  • output/partition_quad_visualization.png - Structured partitioning")
print("  • output/partition_tri_visualization.png - Unstructured partitioning")
print()

print("Key differences visualized:")
print("  ✓ Structured grids have rectangular, aligned partitions")
print("  ✓ Unstructured grids have irregular, topology-based partitions")
print("  ✓ Ghost cells in unstructured grids are scattered (not just edges)")
print("  ✓ Communication patterns differ significantly")
print()

print("Next: Run example 10 to see actual data movement with ppermute!")
print()
