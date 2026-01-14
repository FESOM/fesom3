"""
Example 04: Staggered Grids (Arakawa A, B, C)

This example demonstrates:
- Creating grids with different staggering schemes (A, B, C)
- Understanding variable locations on staggered grids
- Working with velocity and pressure fields
- Comparing staggering types
"""

import numpy as np
from pathlib import Path

from fesomx import QuadGrid, StaggerType, StaggerLocation
from fesomx.core.staggering import get_stagger_config, create_staggered_field
from fesomx.utilities import plot_field, save_plot

print("=" * 60)
print("Example 04: Staggered Grids (Arakawa A, B, C)")
print("=" * 60)
print()

# ============================================
# 1. Create Grids with Different Staggering
# ============================================

print("1. Creating grids with different staggering types...")

nx, ny = 8, 8
domain = (0.0, 1.0, 0.0, 1.0)

# Create three grids with different staggering
grid_A = QuadGrid("A-grid", nx=nx, ny=ny, stagger_type=StaggerType.A)
grid_A.create_mesh(domain=domain)
grid_A.compute_neighbors(halo_width=1)

grid_B = QuadGrid("B-grid", nx=nx, ny=ny, stagger_type=StaggerType.B)
grid_B.create_mesh(domain=domain)
grid_B.compute_neighbors(halo_width=1)

grid_C = QuadGrid("C-grid", nx=nx, ny=ny, stagger_type=StaggerType.C)
grid_C.create_mesh(domain=domain)
grid_C.compute_neighbors(halo_width=1)

print(f"   A-grid created: {grid_A}")
print(f"   B-grid created: {grid_B}")
print(f"   C-grid created: {grid_C}")
print()

# ============================================
# 2. Show Stagger Configurations
# ============================================

print("2. Stagger configurations:")
print()

for stagger_type in [StaggerType.A, StaggerType.B, StaggerType.C]:
    config = get_stagger_config(stagger_type)
    print(f"   {stagger_type.value}-grid:")
    for var, loc in config.items():
        print(f"     {var}: {loc.value}")
    print()

# ============================================
# 3. Create Fields on Each Grid Type
# ============================================

print("3. Creating velocity and pressure fields...")

def create_flow_field(grid, field_type):
    """Create a simple flow field for testing."""
    centers = grid.get_cell_centers()
    x = centers[:, 0].reshape(ny, nx)
    y = centers[:, 1].reshape(ny, nx)

    if field_type == 'u':
        # U-velocity: flow in x-direction
        return -np.sin(np.pi * y) * np.cos(np.pi * x)
    elif field_type == 'v':
        # V-velocity: flow in y-direction
        return np.sin(np.pi * x) * np.cos(np.pi * y)
    elif field_type == 'p':
        # Pressure: related to divergence
        return -0.25 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    else:
        raise ValueError(f"Unknown field type: {field_type}")

# Add fields to each grid
for grid, grid_name in [(grid_A, "A-grid"), (grid_B, "B-grid"), (grid_C, "C-grid")]:
    u_field = create_flow_field(grid, 'u')
    v_field = create_flow_field(grid, 'v')
    p_field = create_flow_field(grid, 'p')

    grid.add_field('u', u_field)
    grid.add_field('v', v_field)
    grid.add_field('p', p_field)

    print(f"   {grid_name} fields added: u, v, p")

print()

# ============================================
# 4. Perform Halo Exchange on All Grids
# ============================================

print("4. Performing halo exchange on all grids...")

for grid, grid_name in [(grid_A, "A-grid"), (grid_B, "B-grid"), (grid_C, "C-grid")]:
    grid.halo_exchange('u', halo_width=1)
    grid.halo_exchange('v', halo_width=1)
    grid.halo_exchange('p', halo_width=1)
    print(f"   {grid_name}: halo exchange completed")

print()

# ============================================
# 5. Compare Field Values
# ============================================

print("5. Comparing field statistics across grids:")
print()

fields = ['u', 'v', 'p']
grids = [(grid_A, "A-grid"), (grid_B, "B-grid"), (grid_C, "C-grid")]

for field_name in fields:
    print(f"   {field_name}-field:")
    for grid, grid_name in grids:
        field_data = grid.get_field_numpy(field_name)
        print(f"     {grid_name}: mean={field_data.mean():7.4f}, "
              f"std={field_data.std():7.4f}, "
              f"range=[{field_data.min():7.4f}, {field_data.max():7.4f}]")
    print()

# ============================================
# 6. Analyze Stagger Offsets
# ============================================

print("6. Variable location offsets:")
print()

for stagger_type in [StaggerType.A, StaggerType.B, StaggerType.C]:
    print(f"   {stagger_type.value}-grid:")
    for var_name in ['u', 'v', 'p']:
        field = create_staggered_field(var_name, stagger_type)
        offset = field.get_offset()
        print(f"     {var_name}: offset={offset}, location={field.location.value}")
    print()

# ============================================
# 7. Visualization
# ============================================

print("7. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    grids_data = [
        (grid_A, "A-grid"),
        (grid_B, "B-grid"),
        (grid_C, "C-grid")
    ]
    fields_to_plot = ['u', 'v', 'p']

    for i, (grid, grid_name) in enumerate(grids_data):
        for j, field_name in enumerate(fields_to_plot):
            field_data = grid.get_field_numpy(field_name).ravel()

            plot_field(
                grid.vertices,
                grid.cells,
                field_data,
                ax=axes[i, j],
                title=f"{grid_name}: {field_name}-field",
                cmap='RdBu_r',
                vmin=-1,
                vmax=1
            )

    plt.tight_layout()
    output_file = "output/example_04_staggered_grids.png"
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
print("  - Created grids with three staggering types (A, B, C)")
print("  - A-grid: all variables at cell centers")
print("  - B-grid: velocities at corners, scalars at centers")
print("  - C-grid: velocities at faces, scalars at centers")
print("  - Performed halo exchange on staggered fields")
print("  - Compared field statistics across staggering types")
print()
