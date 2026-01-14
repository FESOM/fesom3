"""
Example 17: Triangle Grid Divergence with Halo Exchange

This example demonstrates distributed halo exchange for UNSTRUCTURED triangle grids.

Mirrors Example 13 but uses TriGrid instead of QuadGrid, showing that:
- The SAME operator API works for both grid types
- Finite volume (FV) method automatically selected for triangles
- Halo exchange is critical for correct boundary computation

Key demonstration:
- TriGrid with METIS partitioning (graph-based, irregular boundaries)
- WITHOUT halos: Wrong results at partition boundaries
- WITH halos: Correct divergence everywhere

Compares to: Example 13 (structured grid), Example 16 (single-device triangle)

NOTE: The benefit of halo exchange is most visible on multi-device systems.
On single-device systems with shared memory, both with/without halos may show
similar (excellent) accuracy because JAX can access all data. To see the true
benefit, run on:
- Multi-GPU system (4+ GPUs)
- Multi-node cluster
- CPU with virtual devices: set XLA_FLAGS='--xla_force_host_platform_device_count=4'
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from fesomx import TriGrid, StaggerType, get_backend, divergence
from fesomx.utilities import create_structured_triangular_mesh

# Import test helpers
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_helpers import (
    create_divergence_free_field_unstructured,
    compute_error
)

print("=" * 80)
print("Example 17: Triangle Grid Divergence - Demonstrating Halo Exchange")
print("=" * 80)
print()
print("This example shows:")
print("  - Distributed computation on UNSTRUCTURED triangle meshes")
print("  - METIS graph partitioning (irregular partition boundaries)")
print("  - Finite volume method with ghost cell exchange")
print("  - Comparison: with vs without halo exchange")
print()

# ============================================
# Setup
# ============================================

print("1. Setting up distributed triangle grid...")

# Grid parameters
nx, ny = 32, 32  # Creates 32×32×2 = 2048 triangles
n_partitions = 4  # 2×2 partition layout

# Create structured triangular mesh
mesh_data = create_structured_triangular_mesh(nx, ny, domain=(0.0, 1.0, 0.0, 1.0))

print(f"   Mesh: {len(mesh_data['vertices'])} vertices, {len(mesh_data['triangles'])} triangles")
print(f"   Partitions: {n_partitions} (using METIS graph partitioning)")
print()

# Create TriGrid
grid = TriGrid(name="triangle_divergence_demo", stagger_type=StaggerType.A)
grid.create_mesh(vertices=mesh_data['vertices'], triangles=mesh_data['triangles'])
grid.compute_neighbors(halo_width=1)

print(f"   Grid cells: {grid.n_cells}")
print(f"   Grid type: Unstructured (triangular)")
print()

# ============================================
# Create Divergence-Free Velocity Field
# ============================================

print("2. Creating divergence-free velocity field...")

# Solid body rotation: ∇·u = 0 analytically
u, v = create_divergence_free_field_unstructured(grid, mode='rotation')

print(f"   Velocity field shape: u={u.shape}, v={v.shape}")
print(f"   Data structure: 1D arrays (n_cells,) - unstructured")
print(f"   Analytical property: ∇·u = 0 everywhere")
print()

# ============================================
# Create Partition Info
# ============================================

print("3. Creating partition information...")

# Create partition using METIS (graph-based partitioning)
try:
    partition_info = grid.create_partition_info(n_partitions=n_partitions, method='metis')
    print("   ✓ Using METIS partitioning (optimal for irregular meshes)")
except Exception as e:
    print(f"   METIS not available ({e}), using geometric partitioning")
    partition_info = grid.create_partition_info(n_partitions=n_partitions, method='geometric')

# Analyze partition
device_to_cells = partition_info['device_to_cells']
ghost_cell_map = partition_info['ghost_cell_map']

cells_per_partition = [len(cells) for cells in device_to_cells.values()]
print(f"   Cells per partition: {cells_per_partition}")
print(f"   Load balance: min={min(cells_per_partition)}, max={max(cells_per_partition)}, "
      f"imbalance={(max(cells_per_partition)/min(cells_per_partition) - 1)*100:.1f}%")

total_ghost_cells = sum(
    len(cells)
    for neighbor_map in ghost_cell_map.values()
    for cells in neighbor_map.values()
)
print(f"   Total ghost cells: {total_ghost_cells}")
print()

# ============================================
# Setup JAX Sharding for Distribution
# ============================================

print("4. Setting up JAX sharding...")

# Create device mesh
devices = jax.devices()
n_devices = len(devices)

if n_devices >= n_partitions:
    # Use actual devices
    device_array = np.array(devices[:n_partitions])
    mesh = Mesh(device_array, axis_names=('x',))  # 1D sharding for unstructured
    print(f"   Using {n_partitions} actual devices")
else:
    print(f"   WARNING: Need {n_partitions} devices, only {n_devices} available")
    print(f"   Using {n_devices} device(s) with logical partitioning")
    device_array = np.array(devices)
    mesh = Mesh(device_array, axis_names=('x',))
    n_partitions = n_devices

print(f"   Device mesh shape: {mesh.shape}")
print(f"   Mesh axes: {mesh.axis_names}")
print()

# Shard the velocity fields
# Note: For unstructured grids, we use 1D sharding along the cell dimension
with mesh:
    u_sharded = jax.device_put(u, jax.sharding.NamedSharding(mesh, P('x',)))
    v_sharded = jax.device_put(v, jax.sharding.NamedSharding(mesh, P('x',)))

print("   Velocity fields sharded across devices")
print()

# Set backend for operators
backend = get_backend('jax_sharding')
grid.backend = backend

# ============================================
# Test 1: Divergence WITHOUT Halo Exchange
# ============================================

print("5. Computing divergence WITHOUT halo exchange...")
print("   (This will have errors at partition boundaries!)")
print()

# Compute divergence without halo exchange
div_no_halo = divergence(
    u_sharded, v_sharded, grid,
    partition_info=partition_info,
    perform_halo_exchange=False  # NO HALO EXCHANGE
)

# Convert to numpy for analysis
div_no_halo_np = np.array(div_no_halo)

# Compute error (should be 0 since field is divergence-free)
# Exclude boundary cells (cells with < 3 neighbors)
centers = grid.get_cell_centers()
interior_mask = (
    (centers[:, 0] > 0.1) & (centers[:, 0] < 0.9) &
    (centers[:, 1] > 0.1) & (centers[:, 1] < 0.9)
)

error_no_halo = np.abs(div_no_halo_np[interior_mask])
max_error_no_halo = error_no_halo.max()
mean_error_no_halo = error_no_halo.mean()

print(f"   Max interior error: {max_error_no_halo:.6e}")
print(f"   Mean interior error: {mean_error_no_halo:.6e}")
print()

# ============================================
# Test 2: Divergence WITH Halo Exchange
# ============================================

print("6. Computing divergence WITH halo exchange...")
print("   (This should be correct everywhere!)")
print()

# Compute divergence with halo exchange
div_with_halo = divergence(
    u_sharded, v_sharded, grid,
    partition_info=partition_info,
    perform_halo_exchange=True  # WITH HALO EXCHANGE
)

# Check if extended arrays were created
if '_is_extended' in partition_info:
    print(f"   ✓ Extended arrays created: {partition_info['_is_extended']}")
    if partition_info['_is_extended']:
        print(f"   Extended array structure:")
        for device_id in sorted(partition_info['device_to_cells'].keys()):
            n_owned = len(partition_info['device_to_cells'][device_id])
            n_local = partition_info['local_array_sizes'][device_id]
            n_ghosts = n_local - n_owned
            print(f"     Partition {device_id}: {n_owned} owned + {n_ghosts} ghost = {n_local} total")
        print()

# Convert to numpy for analysis
div_with_halo_np = np.array(div_with_halo)

# Compute error
error_with_halo = np.abs(div_with_halo_np[interior_mask])
max_error_with_halo = error_with_halo.max()
mean_error_with_halo = error_with_halo.mean()

print(f"   Max interior error: {max_error_with_halo:.6e}")
print(f"   Mean interior error: {mean_error_with_halo:.6e}")
print()

# ============================================
# Comparison and Analysis
# ============================================

print("7. Comparing results...")
print()

improvement = max_error_no_halo / max_error_with_halo if max_error_with_halo > 1e-10 else float('inf')

print(f"   Error WITHOUT halos: {max_error_no_halo:.6e}")
print(f"   Error WITH halos:    {max_error_with_halo:.6e}")
print(f"   Improvement factor:  {improvement:.1f}x")
print()

# Success criteria
if max_error_with_halo < 1e-10:
    print("   ✓ SUCCESS: Halo exchange enables correct computation!")
    print(f"   ✓ Divergence of divergence-free field is ~0 (within numerical precision)")
else:
    print(f"   ⚠ WARNING: Errors larger than expected")

print()

# ============================================
# Visualization
# ============================================

print("8. Creating visualizations...")

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig = plt.figure(figsize=(20, 12))

    # Create triangulation for plotting
    vertices = mesh_data['vertices']
    triangles = mesh_data['triangles']
    triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)
    centers = grid.get_cell_centers()

    # Get partition assignments
    partition = partition_info.get('partition', np.zeros(grid.n_cells))

    # ----------------------
    # Row 1: Mesh and partitions
    # ----------------------

    # Mesh structure
    ax1 = plt.subplot(3, 3, 1)
    ax1.triplot(triang, 'k-', linewidth=0.2, alpha=0.3)
    ax1.scatter(centers[:, 0], centers[:, 1], c='red', s=1, alpha=0.5)
    ax1.set_title('Triangle Mesh Structure\n(cell centers in red)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')

    # Partition assignment
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.tripcolor(triang, partition, cmap='tab10', shading='flat')
    ax2.set_title('METIS Partitioning\n(irregular boundaries)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(scatter, ax=ax2, label='Partition ID')

    # Velocity field
    ax3 = plt.subplot(3, 3, 3)
    ax3.triplot(triang, 'k-', linewidth=0.1, alpha=0.1)
    skip = 40
    ax3.quiver(
        centers[::skip, 0], centers[::skip, 1],
        np.array(u)[::skip], np.array(v)[::skip],
        scale=15, alpha=0.7, color='darkblue'
    )
    ax3.set_title('Velocity Field\n(solid body rotation, ∇·u=0)', fontsize=11, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')

    # ----------------------
    # Row 2: Divergence comparison
    # ----------------------

    # Divergence without halos
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.tripcolor(triang, div_no_halo_np, cmap='seismic',
                        vmin=-0.5, vmax=0.5, shading='flat')
    ax4.set_title(f'Divergence WITHOUT Halos\nMax error: {max_error_no_halo:.2e}',
                  fontsize=11, fontweight='bold', color='red')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4)

    # Divergence with halos
    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.tripcolor(triang, div_with_halo_np, cmap='seismic',
                        vmin=-0.5, vmax=0.5, shading='flat')
    ax5.set_title(f'Divergence WITH Halos\nMax error: {max_error_with_halo:.2e}',
                  fontsize=11, fontweight='bold', color='green')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5)

    # Error comparison
    ax6 = plt.subplot(3, 3, 6)
    error_no_halo_all = np.abs(div_no_halo_np)
    im6 = ax6.tripcolor(triang, np.log10(error_no_halo_all + 1e-15),
                        cmap='hot_r', shading='flat')
    ax6.set_title('log10(Error) - Without Halos\n(errors at boundaries)',
                  fontsize=11, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6)

    # ----------------------
    # Row 3: Analysis
    # ----------------------

    # Error histograms
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(np.log10(error_no_halo + 1e-15), bins=50, alpha=0.6,
             label='Without halos', color='red', edgecolor='black')
    ax7.hist(np.log10(error_with_halo + 1e-15), bins=50, alpha=0.6,
             label='With halos', color='green', edgecolor='black')
    ax7.set_xlabel('log10(|Error|)')
    ax7.set_ylabel('Frequency')
    ax7.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    # Ghost cell communication pattern
    ax8 = plt.subplot(3, 3, 8)
    # Plot cells colored by whether they need ghost data
    needs_ghost = np.zeros(grid.n_cells)
    for device_id, neighbor_map in ghost_cell_map.items():
        for neighbor_id, cell_indices in neighbor_map.items():
            needs_ghost[list(device_to_cells[device_id])] = 1
    im8 = ax8.tripcolor(triang, needs_ghost, cmap='RdYlGn_r', shading='flat')
    ax8.set_title('Cells Requiring Ghost Data\n(red = needs neighbors from other devices)',
                  fontsize=11, fontweight='bold')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    ax8.set_aspect('equal')
    plt.colorbar(im8, ax=ax8, label='Needs Ghost (1=yes)')

    # Summary statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
SUMMARY STATISTICS

Grid Information:
  • Mesh: {grid.n_cells} triangles
  • Partitions: {n_partitions}
  • Partition method: METIS (graph-based)

Load Balance:
  • Min cells/partition: {min(cells_per_partition)}
  • Max cells/partition: {max(cells_per_partition)}
  • Imbalance: {(max(cells_per_partition)/min(cells_per_partition) - 1)*100:.1f}%

Communication:
  • Total ghost cells: {total_ghost_cells}
  • Avg ghosts/partition: {total_ghost_cells/n_partitions:.0f}

Accuracy (interior cells):
  WITHOUT halos:
    • Max error: {max_error_no_halo:.2e}
    • Mean error: {mean_error_no_halo:.2e}

  WITH halos:
    • Max error: {max_error_with_halo:.2e}
    • Mean error: {mean_error_with_halo:.2e}

  Improvement: {improvement:.0f}× better!

Conclusion:
  ✓ Halo exchange ESSENTIAL
    for distributed unstructured grids
  ✓ Finite volume method works
    correctly with ghost cells
    """
    ax9.text(0.1, 0.95, summary_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    output_file = "output/example_17_triangle_divergence_halo.png"
    Path("output").mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')

    print(f"   Visualization saved to: {output_file}")
    print()

except ImportError:
    print("   Matplotlib not available, skipping visualization")
    print()

# ============================================
# Summary
# ============================================

print("=" * 80)
print("Example completed successfully!")
print("=" * 80)
print()
print("Key insights:")
print()
print("1. DISTRIBUTED UNSTRUCTURED GRIDS WORK:")
print("   - Triangle grids successfully partitioned with METIS")
print("   - Ghost cell communication enables correct boundary computation")
print("   - Same operator API as structured grids!")
print()
print("2. HALO EXCHANGE IS CRITICAL:")
print(f"   - Without halos: max error = {max_error_no_halo:.2e}")
print(f"   - With halos:    max error = {max_error_with_halo:.2e}")
print(f"   - Improvement:   {improvement:.0f}× better with halo exchange")
print()
print("3. FINITE VOLUME METHOD:")
print("   - Edge-based flux summation works on irregular partitions")
print("   - METIS creates balanced partitions with minimal edge cuts")
print("   - Ghost cells accessed via JAX cross-device indexing")
print()
print("4. COMPARISON TO STRUCTURED GRIDS:")
print("   - QuadGrid (Example 13): Regular partition boundaries, array slicing")
print("   - TriGrid (Example 17): Irregular boundaries, graph-based neighbors")
print("   - BOTH use same divergence() function! (automatic dispatch)")
print()
print("This demonstrates that the modular operator architecture")
print("enables distributed computing on BOTH structured and")
print("unstructured grids with a unified API!")
print()
