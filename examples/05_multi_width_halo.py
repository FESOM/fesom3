"""
Example 05: Variable Halo Width Exchange

This example demonstrates:
- Using different halo widths
- Asymmetric halo widths (different per direction)
- Performance implications of halo width
- Visualizing halo regions
"""

import numpy as np
import time
from pathlib import Path

from fesomx import QuadGrid, StaggerType
from fesomx.core.halo_patterns import HaloWidth, get_halo_indices_2d, get_interior_indices_2d
from fesomx.utilities import plot_field, plot_halo_regions, save_plot

print("=" * 60)
print("Example 05: Variable Halo Width Exchange")
print("=" * 60)
print()

# ============================================
# 1. Create Grid
# ============================================

print("1. Creating grid...")

nx, ny = 16, 16
grid = QuadGrid(
    name="multi_halo",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,
    periodic=(False, False)
)

grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))

print(f"   Grid created: {nx}x{ny} cells")
print()

# ============================================
# 2. Test Different Halo Widths
# ============================================

print("2. Testing different halo widths...")
print()

halo_widths = [1, 2, 3, 4]

# Create a test field
centers = grid.get_cell_centers()
x = centers[:, 0].reshape(ny, nx)
y = centers[:, 1].reshape(ny, nx)
base_field = np.sin(4*np.pi*x) * np.cos(4*np.pi*y)

for width in halo_widths:
    print(f"   Halo width = {width}:")

    # Compute neighbors with this halo width
    grid.compute_neighbors(halo_width=width)

    # Add field
    field_name = f"field_w{width}"
    grid.add_field(field_name, base_field.copy())

    # Time the halo exchange
    start = time.time()
    grid.halo_exchange(field_name, halo_width=width)
    elapsed = time.time() - start

    result = grid.get_field_numpy(field_name)

    # Calculate halo region size
    hw = HaloWidth(width)
    interior_slice = get_interior_indices_2d((ny, nx), hw)
    interior_size = (interior_slice[0].stop - interior_slice[0].start) * \
                   (interior_slice[1].stop - interior_slice[1].start)
    halo_size = ny * nx - interior_size

    print(f"     Interior cells: {interior_size}")
    print(f"     Halo cells: {halo_size}")
    print(f"     Halo fraction: {halo_size / (ny*nx) * 100:.1f}%")
    print(f"     Exchange time: {elapsed*1000:.2f} ms")
    print()

# ============================================
# 3. Asymmetric Halo Widths
# ============================================

print("3. Testing asymmetric halo widths...")
print()

# Different widths in x and y directions
hw_asymmetric = HaloWidth((2, 3))  # width_x=2, width_y=3

print(f"   Asymmetric halo: x={hw_asymmetric.width_x}, y={hw_asymmetric.width_y}")
print(f"   Direction widths:")
for direction in ['left', 'right', 'top', 'bottom']:
    w = hw_asymmetric.get_direction_width(direction)
    print(f"     {direction}: {w}")
print()

# ============================================
# 4. Directional Halo Widths
# ============================================

print("4. Testing directional halo widths...")
print()

# Different width for each direction
hw_directional = HaloWidth({
    'left': 1,
    'right': 2,
    'top': 3,
    'bottom': 1
})

print(f"   Directional halo:")
for direction in ['left', 'right', 'top', 'bottom']:
    w = hw_directional.get_direction_width(direction)
    print(f"     {direction}: {w}")
print()

# ============================================
# 5. Visualize Halo Regions
# ============================================

print("5. Visualizing halo regions...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes = axes.ravel()

    for i, width in enumerate([1, 2, 3, 4]):
        # Create field indicating halo vs interior
        halo_field = np.zeros(ny * nx)
        hw = HaloWidth(width)

        # Mark cells as halo (1) or interior (0)
        for j in range(ny):
            for k in range(nx):
                cell_id = j * nx + k
                # Cell is in halo if within width of boundary
                if k < width or k >= nx - width or j < width or j >= ny - width:
                    halo_field[cell_id] = 1

        plot_field(
            grid.vertices,
            grid.cells,
            halo_field,
            ax=axes[i],
            title=f"Halo Regions (width={width})\nBlue=Interior, Red=Halo",
            cmap='RdYlBu',
            vmin=0,
            vmax=1
        )

    plt.tight_layout()
    output_file = "output/example_05_halo_widths.png"
    Path("output").mkdir(exist_ok=True)
    save_plot(output_file, dpi=150)

    print(f"   Visualization saved to: {output_file}")
    print()

except ImportError:
    print("   Matplotlib not available, skipping visualization")
    print()

# ============================================
# 6. Performance Analysis
# ============================================

print("6. Performance analysis:")
print()

print(f"   Grid size: {nx}x{ny} = {nx*ny} cells")
print()
print("   Halo overhead by width:")
print("   Width | Interior | Halo | Overhead")
print("   ------|----------|------|----------")

for width in [1, 2, 3, 4, 5]:
    hw = HaloWidth(width)
    interior_slice = get_interior_indices_2d((ny, nx), hw)

    if interior_slice[0].stop > interior_slice[0].start and \
       interior_slice[1].stop > interior_slice[1].start:
        interior_size = (interior_slice[0].stop - interior_slice[0].start) * \
                       (interior_slice[1].stop - interior_slice[1].start)
        halo_size = ny * nx - interior_size
        overhead = halo_size / interior_size * 100
        print(f"     {width:2d}   |  {interior_size:5d}   | {halo_size:4d} |  {overhead:5.1f}%")
    else:
        print(f"     {width:2d}   |   N/A    | N/A  |   N/A")

print()

# ============================================
# 7. Recommendations
# ============================================

print("7. Recommendations:")
print()
print("   - Smaller halo widths (1-2) minimize communication overhead")
print("   - Larger halo widths (3-5) needed for high-order stencils")
print("   - For weak scaling, halo overhead decreases with grid size")
print("   - Asymmetric halos useful for anisotropic domains")
print()

# ============================================
# Summary
# ============================================

print("=" * 60)
print("Example completed successfully!")
print("=" * 60)
print()
print("Key takeaways:")
print("  - Tested halo widths from 1 to 4 cells")
print("  - Demonstrated uniform, asymmetric, and directional halo widths")
print("  - Analyzed halo overhead vs grid size")
print("  - Visualized halo regions for different widths")
print("  - Halo width choice depends on stencil requirements and scaling")
print()
