"""
Example 07: Halo Exchange Visualization

This example demonstrates:
- Structured partitioning of QuadGrid across devices
- Visualization of device boundaries and halos
- Communication patterns for halo exchange
- Comparison of different staggering schemes (A, B, C)
- Educational step-by-step view of data movement
"""

import numpy as np

from fesomx import QuadGrid, StaggerType
from fesomx.core.staggering import get_stagger_config, StaggerLocation

print("=" * 70)
print("Example 07: Halo Exchange Visualization and Explanation")
print("=" * 70)
print()

# ============================================
# 1. Create Grid with Multiple Devices
# ============================================

print("1. Creating 128×128 grid for 32-device (4×8 mesh) partitioning...")
print()

nx, ny = 128, 128
grid = QuadGrid(
    "partition_demo",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,
    periodic=(True, True)  # Periodic for complete halo exchange
)

grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))

print(f"   Grid: {grid}")
print(f"   Total cells: {grid.n_cells}")
print(f"   Domain: [0, 1] × [0, 1]")
print()

# ============================================
# 2. Create Partition Information
# ============================================

print("2. Creating partition information for 4×8 device mesh...")
print()

mesh_shape = (4, 8)  # 4 rows, 8 columns = 32 devices
partition_info = grid.create_partition_info(mesh_shape=mesh_shape)

print(f"   Mesh shape: {partition_info['mesh_shape']}")
print(f"   Strategy: {partition_info['strategy']}")
print(f"   Periodic boundaries: {partition_info['periodic']}")
print()

# Show partition details for a few devices
print("   Sample device partitions:")
print("   " + "=" * 60)
print(f"   {'Device':<10} {'Position':<15} {'Cells (j,i)':<25}")
print("   " + "=" * 60)

sample_devices = [0, 7, 24, 31]  # Corners and representative devices
for device_id in sample_devices:
    j_start, j_end, i_start, i_end = partition_info['device_to_cells'][device_id]
    dev_j = device_id // 8
    dev_i = device_id % 8
    cells_range = f"[{j_start}:{j_end}, {i_start}:{i_end}]"
    print(f"   {device_id:<10} ({dev_j}, {dev_i})<10  {cells_range:<25}")

print()

# ============================================
# 3. Analyze Neighbor Connectivity
# ============================================

print("3. Analyzing neighbor connectivity...")
print()

# Show neighbors for a central device
device_id = 13  # Middle device (row=1, col=5)
neighbors = partition_info['neighbor_map'][device_id]

print(f"   Device {device_id} neighbors:")
for direction, neighbor_id in neighbors.items():
    print(f"     {direction:6s} → Device {neighbor_id}")

print()

# ============================================
# 4. Calculate Halo Regions
# ============================================

print("4. Calculating halo regions for different widths...")
print()

halo_widths = [1, 2, 3]

print("   " + "=" * 70)
print(f"   {'Halo':<8} {'Interior':<20} {'Halo Cells':<15} {'Overhead':<10}")
print(f"   {'Width':<8} {'Cells':<20} {'(Total)':<15} {'%':<10}")
print("   " + "=" * 70)

for width in halo_widths:
    # Calculate interior and halo sizes
    interior_y = ny - 2 * width
    interior_x = nx - 2 * width
    interior_cells = interior_y * interior_x
    total_cells = ny * nx
    halo_cells = total_cells - interior_cells
    overhead = (halo_cells / interior_cells) * 100

    print(f"   {width:<8} {interior_y}×{interior_x} = {interior_cells:<8}   "
          f"{halo_cells:<15} {overhead:6.2f}%")

print()

# ============================================
# 5. Staggering Schemes Comparison
# ============================================

print("5. Variable locations for different staggering schemes...")
print()

print("   Arakawa Grid Staggering:")
print("   " + "-" * 70)

for stagger_type in [StaggerType.A, StaggerType.B, StaggerType.C]:
    config = get_stagger_config(stagger_type)
    print(f"\n   {stagger_type.value}-grid:")
    for var, location in config.items():
        print(f"     {var:3s}: {location.value:10s}", end="")
        if location == StaggerLocation.CENTER:
            print("  (cell centers)")
        elif location == StaggerLocation.CORNER:
            print("  (cell corners/vertices)")
        elif location == StaggerLocation.FACE_X:
            print("  (faces normal to x)")
        elif location == StaggerLocation.FACE_Y:
            print("  (faces normal to y)")

print()
print()

# ============================================
# 6. Communication Pattern Analysis
# ============================================

print("6. Halo exchange communication pattern...")
print()

# Calculate communication volume for one device
dev_id = 13
j_start, j_end, i_start, i_end = partition_info['device_to_cells'][dev_id]
local_ny = j_end - j_start
local_nx = i_end - i_start

halo_width = 1

print(f"   Device {dev_id} (local grid: {local_ny}×{local_nx}):")
print(f"   Halo width: {halo_width}")
print()

print("   Direction-wise communication:")
print("   " + "-" * 60)
print(f"   {'Direction':<12} {'Send Cells':<15} {'Recv Cells':<15}")
print("   " + "-" * 60)

# Left/Right: exchange columns
lr_cells = local_ny * halo_width
print(f"   {'Left':<12} {lr_cells:<15} {lr_cells:<15}")
print(f"   {'Right':<12} {lr_cells:<15} {lr_cells:<15}")

# Top/Bottom: exchange rows
tb_cells = local_nx * halo_width
print(f"   {'Top':<12} {tb_cells:<15} {tb_cells:<15}")
print(f"   {'Bottom':<12} {tb_cells:<15} {tb_cells:<15}")

total_comm = 2 * (lr_cells + tb_cells)
print("   " + "-" * 60)
print(f"   {'TOTAL':<12} {total_comm:<15} cells exchanged per device")

print()

# ============================================
# 7. Educational Explanation
# ============================================

print("7. How Structured Halo Exchange Works:")
print("=" * 70)
print()

print("   Step 1: PARTITION")
print("   -----------------")
print("   • Grid divided into rectangular shards")
print("   • Each device owns a contiguous region")
print("   • 4×8 mesh → 32 devices, each gets ~512 cells (128²/32)")
print()

print("   Step 2: IDENTIFY BOUNDARIES")
print("   ---------------------------")
print("   • Each device identifies its edge cells")
print("   • Left edge: leftmost column(s)")
print("   • Right edge: rightmost column(s)")
print("   • Top/Bottom edges: similarly")
print()

print("   Step 3: PACK DATA")
print("   -----------------")
print("   • Extract edge cells to send")
print("   • Example: Device 13's right edge → send to Device 14")
print("   • No scatter needed - edges are contiguous!")
print()

print("   Step 4: COMMUNICATE (jax.lax.ppermute)")
print("   --------------------------------------")
print("   • Simple permutation pattern:")
print("   •   Send right edge → right neighbor")
print("   •   Receive from left neighbor → left halo")
print("   • Directional: 4 exchanges (L, R, T, B)")
print()

print("   Step 5: UNPACK DATA")
print("   -------------------")
print("   • Insert received data into halo regions")
print("   • Halos at shard boundaries, contiguous")
print("   • Ready for computation!")
print()

# ============================================
# 8. Visual ASCII Diagram
# ============================================

print("8. Visual representation of 2×2 partition (simplified):")
print("=" * 70)
print()

print("   BEFORE Halo Exchange:")
print("   " + "-" * 66)
print("   Device 0          │  Device 1")
print("   ┌────────────┐    │  ┌────────────┐")
print("   │ ?  ?  ?  ? │    │  │ ?  ?  ?  ? │    ? = uninitialized halo")
print("   │ ?  A  A  A │    │  │ ?  B  B  B │    A,B,C,D = interior data")
print("   │ ?  A  A  A │    │  │ ?  B  B  B │")
print("   │ ?  A  A  A │    │  │ ?  B  B  B │")
print("   └────────────┘    │  └────────────┘")
print("   ─────────────────────────────────────────────────────")
print("   Device 2          │  Device 3")
print("   ┌────────────┐    │  ┌────────────┐")
print("   │ ?  C  C  C │    │  │ ?  D  D  D │")
print("   │ ?  C  C  C │    │  │ ?  D  D  D │")
print("   │ ?  C  C  C │    │  │ ?  D  D  D │")
print("   │ ?  ?  ?  ? │    │  │ ?  ?  ?  ? │")
print("   └────────────┘    │  └────────────┘")
print()

print("   AFTER Halo Exchange:")
print("   " + "-" * 66)
print("   Device 0          │  Device 1")
print("   ┌────────────┐    │  ┌────────────┐")
print("   │ C  C  C  C │←──┼─ Top neighbor (periodic)")
print("   │ B  A  A  A │←──┼─ Left neighbor  ")
print("   │ B  A  A  A │    │  │ A  B  B  B │")
print("   │ B  A  A  A │    │  │ A  B  B  B │")
print("   └────────────┘    │  └────────────┘")
print("                          ↑ filled halos")
print()

# ============================================
# 9. Performance Insights
# ============================================

print("9. Performance characteristics:")
print("=" * 70)
print()

print("   Communication volume per device:")
print(f"     • 4 directional sends/receives")
print(f"     • Volume = O(√(N/P)) where N=total cells, P=devices")
print(f"     • For 128²/32: ~{(2*32 + 2*16):.0f} cells per device")
print()

print("   Scaling properties:")
print("     • Strong scaling: Fixed N, increase P → less per-device work")
print("     • Weak scaling: N∝P → constant per-device work")
print("     • Halo overhead ∝ 1/√P (decreases with more devices)")
print()

# ============================================
# Summary
# ============================================

print("=" * 70)
print("Example completed!")
print("=" * 70)
print()

print("Key Takeaways:")
print("  1. Structured grids use simple edge-based halo exchange")
print("  2. Communication is directional (L/R/T/B)")
print("  3. Halos are contiguous - no scatter/gather needed")
print("  4. Periodic boundaries handled by wrapping edges")
print("  5. Overhead decreases with grid size")
print()

print("Next steps:")
print("  • Run example 08 to verify correctness with real data")
print("  • Compare with unstructured exchange (triangular grids)")
print("  • Test performance scaling with example 09")
print()
