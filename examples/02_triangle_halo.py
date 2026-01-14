"""
Example 02: Triangular Grid with Halo Exchange

This example demonstrates:
- Creating an unstructured triangular grid
- Working with triangle connectivity
- Halo exchange on unstructured grids
- Visualizing triangular meshes
"""

import numpy as np
from pathlib import Path

from fesomx import TriGrid, StaggerType
from fesomx.utilities import create_structured_triangular_mesh, plot_grid, plot_field, save_plot

print("=" * 60)
print("Example 02: Triangular Grid with Halo Exchange")
print("=" * 60)
print()

# ============================================
# 1. Create Triangular Mesh
# ============================================

print("1. Creating triangular grid...")

# Create structured triangular mesh (splits quads into triangles)
mesh_data = create_structured_triangular_mesh(
    nx=6,
    ny=6,
    domain=(0.0, 1.0, 0.0, 1.0)
)

print(f"   Vertices: {len(mesh_data['vertices'])}")
print(f"   Triangles: {len(mesh_data['triangles'])}")
print()

# ============================================
# 2. Create Grid Object
# ============================================

print("2. Creating TriGrid object...")

grid = TriGrid(name="tri_mesh", stagger_type=StaggerType.A)
grid.create_mesh(
    vertices=mesh_data['vertices'],
    triangles=mesh_data['triangles']
)
grid.compute_neighbors(halo_width=1)

print(f"   Grid: {grid}")
print(f"   Number of cells: {grid.n_cells}")
print(f"   Number of vertices: {grid.n_vertices}")
print(f"   Number of edges: {len(grid.edges)}")
print()

# ============================================
# 3. Create Field on Triangular Grid
# ============================================

print("3. Creating field on triangular grid...")

# Create a radial field centered at (0.5, 0.5)
centers = grid.get_cell_centers()
x = centers[:, 0]
y = centers[:, 1]

# Radial distance from center
r = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
field = np.sin(4 * np.pi * r)

print(f"   Field created with {len(field)} values")
print(f"   Field range: [{field.min():.4f}, {field.max():.4f}]")
print()

# Add field to grid
grid.add_field("radial_wave", field)

# ============================================
# 4. Perform Halo Exchange
# ============================================

print("4. Performing halo exchange on unstructured grid...")

grid.halo_exchange("radial_wave", halo_width=1)
result_field = grid.get_field_numpy("radial_wave")

print(f"   Halo exchange completed")
print(f"   Result field shape: {result_field.shape}")
print()

# ============================================
# 5. Analyze Grid Topology
# ============================================

print("5. Analyzing grid topology...")

# Find boundary cells
boundary_cells = grid.get_boundary_cells()
print(f"   Boundary cells: {len(boundary_cells)}")
print(f"   Interior cells: {grid.n_cells - len(boundary_cells)}")

# Check neighbor counts
neighbor_counts = [len(neighbors) for neighbors in grid.cell_neighbors]
print(f"   Avg neighbors per cell: {np.mean(neighbor_counts):.2f}")
print(f"   Min neighbors: {np.min(neighbor_counts)}")
print(f"   Max neighbors: {np.max(neighbor_counts)}")
print()

# ============================================
# 6. Visualization
# ============================================

print("6. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot grid structure
    plot_grid(
        grid.vertices,
        grid.cells,
        ax=axes[0],
        title="Triangular Grid Structure",
        show_vertices=True,
        show_cells=True
    )

    # Highlight boundary cells
    boundary_field = np.zeros(grid.n_cells)
    boundary_field[boundary_cells] = 1.0

    # Plot field
    plot_field(
        grid.vertices,
        grid.cells,
        result_field,
        ax=axes[1],
        title="Radial Wave Field on Triangular Grid",
        cmap='seismic',
        vmin=-1,
        vmax=1
    )

    plt.tight_layout()
    output_file = "output/example_02_tri_grid.png"
    Path("output").mkdir(exist_ok=True)
    save_plot(output_file, dpi=150)

    print(f"   Visualization saved to: {output_file}")
    print()

except ImportError:
    print("   Matplotlib not available, skipping visualization")
    print()

# ============================================
# 7. Cell Area Statistics
# ============================================

print("7. Cell geometry statistics:")

areas = grid.get_cell_volumes()
print(f"   Mean area: {areas.mean():.6f}")
print(f"   Std area:  {areas.std():.6f}")
print(f"   Min area:  {areas.min():.6f}")
print(f"   Max area:  {areas.max():.6f}")
print()

# ============================================
# Summary
# ============================================

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
print()
print("Key takeaways:")
print("  - Created an unstructured triangular grid")
print("  - Analyzed grid topology and neighbor connectivity")
print("  - Applied a radial wave field to the grid")
print("  - Performed halo exchange on unstructured mesh")
print()
