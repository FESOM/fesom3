"""
Example 01: Simple Quadrilateral Grid with Halo Exchange

This example demonstrates:
- Creating a basic structured quadrilateral grid
- Adding a field to the grid
- Performing halo exchange
- Visualizing the results
"""

import numpy as np
from pathlib import Path

from fesomx import QuadGrid, StaggerType, get_backend
from fesomx.core.halo_patterns import HaloWidth
from fesomx.utilities import plot_grid, plot_field, save_plot

print("=" * 60)
print("Example 01: Simple Quadrilateral Grid with Halo Exchange")
print("=" * 60)
print()

# ============================================
# 1. Create Grid
# ============================================

print("1. Creating quadrilateral grid...")
nx, ny = 8, 8
grid = QuadGrid(
    name="simple_quad",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,
    periodic=(False, False)
)

# Create mesh
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
grid.compute_neighbors(halo_width=1)

print(f"   Grid created: {grid}")
print(f"   Number of cells: {grid.n_cells}")
print(f"   Number of vertices: {grid.n_vertices}")
print()

# ============================================
# 2. Create Test Field
# ============================================

print("2. Creating test field...")

# Create a smooth field (e.g., 2D Gaussian)
centers = grid.get_cell_centers()
x = centers[:, 0].reshape(ny, nx)
y = centers[:, 1].reshape(ny, nx)

# Gaussian centered at (0.5, 0.5)
field = np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.1)

print(f"   Field shape: {field.shape}")
print(f"   Field range: [{field.min():.4f}, {field.max():.4f}]")
print()

# Add field to grid
grid.add_field("gaussian", field)

# ============================================
# 3. Perform Halo Exchange
# ============================================

print("3. Performing halo exchange...")

# Perform halo exchange
halo_width = 1
grid.halo_exchange("gaussian", halo_width=halo_width)

print(f"   Halo exchange completed (width={halo_width})")
print()

# ============================================
# 4. Retrieve and Check Results
# ============================================

print("4. Checking results...")

result_field = grid.get_field_numpy("gaussian")
print(f"   Result shape: {result_field.shape}")
print(f"   Result range: [{result_field.min():.4f}, {result_field.max():.4f}]")
print()

# ============================================
# 5. Visualization
# ============================================

print("5. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot grid structure
    plot_grid(
        grid.vertices,
        grid.cells,
        ax=axes[0],
        title="Grid Structure",
        show_vertices=True,
        show_cells=True
    )

    # Plot field
    plot_field(
        grid.vertices,
        grid.cells,
        result_field.ravel(),
        ax=axes[1],
        title="Gaussian Field after Halo Exchange",
        cmap='plasma'
    )

    plt.tight_layout()
    output_file = "output/example_01_quad_grid.png"
    Path("output").mkdir(exist_ok=True)
    save_plot(output_file, dpi=150)

    print(f"   Visualization saved to: {output_file}")
    print()

except ImportError:
    print("   Matplotlib not available, skipping visualization")
    print()

# ============================================
# 6. Compute Statistics
# ============================================

print("6. Field statistics:")
print(f"   Mean: {result_field.mean():.6f}")
print(f"   Std:  {result_field.std():.6f}")
print(f"   Min:  {result_field.min():.6f}")
print(f"   Max:  {result_field.max():.6f}")
print()

# ============================================
# Summary
# ============================================

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
print()
print("Key takeaways:")
print("  - Created a simple structured quadrilateral grid")
print("  - Added a 2D Gaussian field to the grid")
print("  - Performed halo exchange with width=1")
print("  - Visualized the grid and field")
print()
