"""
Example 03: Mixed Grid (Quads + Triangles) with Halo Exchange

This example demonstrates:
- Creating a mixed grid with both quads and triangles
- Handling variable cell connectivity
- Halo exchange on mixed grids
- Analyzing different cell types
"""

import numpy as np
from pathlib import Path

from fesomx import MixedGrid, StaggerType
from fesomx.utilities import create_mixed_mesh, plot_grid, plot_field, save_plot

print("=" * 60)
print("Example 03: Mixed Grid with Halo Exchange")
print("=" * 60)
print()

# ============================================
# 1. Create Mixed Mesh
# ============================================

print("1. Creating mixed grid (quads + triangles)...")

# Create mesh with 30% triangles, 70% quads
mesh_data = create_mixed_mesh(
    nx=8,
    ny=8,
    domain=(0.0, 1.0, 0.0, 1.0),
    tri_fraction=0.3,
    seed=42  # For reproducibility
)

print(f"   Total vertices: {len(mesh_data['vertices'])}")
print(f"   Total cells: {len(mesh_data['cells'])}")
print(f"   Triangles: {np.sum(mesh_data['cell_types'] == 3)}")
print(f"   Quadrilaterals: {np.sum(mesh_data['cell_types'] == 4)}")
print()

# ============================================
# 2. Create MixedGrid Object
# ============================================

print("2. Creating MixedGrid object...")

grid = MixedGrid(name="mixed_mesh", stagger_type=StaggerType.A)
grid.create_mesh(
    vertices=mesh_data['vertices'],
    cells=mesh_data['cells']
)
grid.compute_neighbors(halo_width=1)

print(f"   Grid: {grid}")
print(f"   Number of cells: {grid.n_cells}")
print(f"   Number of vertices: {grid.n_vertices}")
print()

# ============================================
# 3. Create Field with Different Values per Cell Type
# ============================================

print("3. Creating field that varies by cell type...")

# Create a field that highlights the difference between cell types
centers = grid.get_cell_centers()
x = centers[:, 0]
y = centers[:, 1]

# Base field: smooth gradient
base_field = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)

# Modify based on cell type
field = base_field.copy()
tri_cells = grid.get_triangular_cells()
quad_cells = grid.get_quadrilateral_cells()

# Add offset to triangle cells to distinguish them
field[tri_cells] += 0.5

print(f"   Field created with {len(field)} values")
print(f"   Triangular cells - mean: {field[tri_cells].mean():.4f}")
print(f"   Quadrilateral cells - mean: {field[quad_cells].mean():.4f}")
print()

# Add field to grid
grid.add_field("mixed_field", field)

# ============================================
# 4. Perform Halo Exchange
# ============================================

print("4. Performing halo exchange on mixed grid...")

grid.halo_exchange("mixed_field", halo_width=1)
result_field = grid.get_field_numpy("mixed_field")

print(f"   Halo exchange completed")
print()

# ============================================
# 5. Analyze Cell Statistics by Type
# ============================================

print("5. Analyzing cell statistics by type...")

areas = grid.get_cell_volumes()

# Triangular cells
tri_areas = areas[tri_cells]
print(f"   Triangular cells:")
print(f"     Count: {len(tri_cells)}")
print(f"     Mean area: {tri_areas.mean():.6f}")
print(f"     Std area: {tri_areas.std():.6f}")
print()

# Quadrilateral cells
quad_areas = areas[quad_cells]
print(f"   Quadrilateral cells:")
print(f"     Count: {len(quad_cells)}")
print(f"     Mean area: {quad_areas.mean():.6f}")
print(f"     Std area: {quad_areas.std():.6f}")
print()

# ============================================
# 6. Neighbor Connectivity Analysis
# ============================================

print("6. Analyzing neighbor connectivity...")

# Check how many neighbors each cell has
neighbor_counts = [len(neighbors) for neighbors in grid.cell_neighbors]
tri_neighbor_counts = [neighbor_counts[i] for i in tri_cells]
quad_neighbor_counts = [neighbor_counts[i] for i in quad_cells]

print(f"   Triangular cells:")
print(f"     Avg neighbors: {np.mean(tri_neighbor_counts):.2f}")
print(f"   Quadrilateral cells:")
print(f"     Avg neighbors: {np.mean(quad_neighbor_counts):.2f}")
print()

# ============================================
# 7. Visualization
# ============================================

print("7. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot grid structure
    plot_grid(
        grid.vertices,
        grid.cells,
        ax=axes[0],
        title="Mixed Grid Structure\n(Quads + Triangles)",
        show_vertices=False,
        show_cells=True
    )

    # Plot cell types
    cell_type_field = grid.cell_types.astype(float)
    plot_field(
        grid.vertices,
        grid.cells,
        cell_type_field,
        ax=axes[1],
        title="Cell Types\n(3=Triangle, 4=Quad)",
        cmap='Set1',
        vmin=3,
        vmax=4
    )

    # Plot actual field
    plot_field(
        grid.vertices,
        grid.cells,
        result_field,
        ax=axes[2],
        title="Field on Mixed Grid\nafter Halo Exchange",
        cmap='viridis'
    )

    plt.tight_layout()
    output_file = "output/example_03_mixed_grid.png"
    Path("output").mkdir(exist_ok=True)
    save_plot(output_file, dpi=150)

    print(f"   Visualization saved to: {output_file}")
    print()

except ImportError:
    print("   Matplotlib not available, skipping visualization")
    print()

# ============================================
# Summary
# ============================================

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
print()
print("Key takeaways:")
print("  - Created a mixed grid with both quadrilaterals and triangles")
print("  - Handled variable cell connectivity (3 and 4 vertices per cell)")
print("  - Analyzed statistics separately for each cell type")
print("  - Performed halo exchange on mixed unstructured grid")
print()
