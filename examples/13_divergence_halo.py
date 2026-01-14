"""
Example 13: Divergence with Halo Exchange

This example demonstrates WHY halo exchange is needed for stencil operations.

Key demonstration:
- Divergence requires neighbor cells for finite differences
- WITHOUT halos: Wrong results at partition boundaries
- WITH halos: Correct divergence everywhere

This answers: "What does halo exchange solve?"
Answer: Enables correct stencil computations on distributed grids!
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from fesomx import QuadGrid, StaggerType, get_backend, divergence
from fesomx.operators.divergence import divergence_free_field_2d

print("=" * 80)
print("Example 13: Divergence Computation - Demonstrating Need for Halo Exchange")
print("=" * 80)
print()

# ============================================
# Setup
# ============================================

print("1. Setting up distributed grid...")

# Grid parameters
nx, ny = 64, 64
n_partitions_x, n_partitions_y = 2, 2

# Create grid
grid = QuadGrid(
    name="divergence_demo",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,  # All variables at centers
    periodic=(False, False)
)

# Create mesh
grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
grid.dx = 1.0 / (nx - 1)
grid.dy = 1.0 / (ny - 1)

print(f"   Grid: {nx} x {ny} cells")
print(f"   Partitions: {n_partitions_x} x {n_partitions_y}")
print()

# ============================================
# Create Divergence-Free Velocity Field
# ============================================

print("2. Creating divergence-free velocity field...")

# Get cell centers
centers = grid.get_cell_centers()
x = centers[:, 0].reshape(ny, nx)
y = centers[:, 1].reshape(ny, nx)

# Create divergence-free field (∇·u = 0 analytically)
# Using sine waves: u = sin(πx)cos(πy), v = -cos(πx)sin(πy)
u, v = divergence_free_field_2d(
    jnp.array(x), jnp.array(y),
    mode='sine'
)

print(f"   Velocity field shape: u={u.shape}, v={v.shape}")
print(f"   Analytical property: ∇·u = 0 everywhere")
print()

# ============================================
# Setup JAX Sharding for Distribution
# ============================================

print("3. Setting up JAX sharding...")

# Create device mesh
devices = jax.devices()
n_devices = len(devices)

if n_devices >= n_partitions_x * n_partitions_y:
    device_mesh = mesh_utils.create_device_mesh(
        (n_partitions_x, n_partitions_y),
        devices=devices[:n_partitions_x * n_partitions_y]
    )
else:
    print(f"   WARNING: Need {n_partitions_x * n_partitions_y} devices, "
          f"only {n_devices} available")
    print(f"   Using {n_devices} device(s) with logical partitioning")
    device_mesh = mesh_utils.create_device_mesh((n_devices, 1))
    n_partitions_x, n_partitions_y = n_devices, 1

mesh = Mesh(device_mesh, axis_names=('x', 'y'))

print(f"   Device mesh: {device_mesh.shape}")
print(f"   Mesh axes: {mesh.axis_names}")
print()

# Shard the velocity fields
with mesh:
    u_sharded = jax.device_put(u, jax.sharding.NamedSharding(mesh, P('x', 'y')))
    v_sharded = jax.device_put(v, jax.sharding.NamedSharding(mesh, P('x', 'y')))

# Create partition info for halo exchange
partition_info = {
    'mesh': mesh,
    'partition_spec': P('x', 'y'),
    'shape': (ny, nx),
    'n_partitions': (n_partitions_x, n_partitions_y)
}

# ============================================
# Test 1: Divergence WITHOUT Halo Exchange
# ============================================

print("4. Computing divergence WITHOUT halo exchange...")
print("   (This will have errors at partition boundaries!)")

# Set backend for operators
backend = get_backend('jax_sharding')
grid.backend = backend

# Compute divergence without halo exchange
div_no_halo = divergence(
    u_sharded, v_sharded, grid,
    partition_info=partition_info,
    perform_halo_exchange=False  # NO HALO EXCHANGE
)

# Convert to numpy for analysis
div_no_halo_np = np.array(div_no_halo)

# Compute error (should be 0 since field is divergence-free)
error_no_halo = np.abs(div_no_halo_np)
max_error_no_halo = error_no_halo.max()
mean_error_no_halo = error_no_halo[1:-1, 1:-1].mean()

print(f"   Max error: {max_error_no_halo:.6e}")
print(f"   Mean interior error: {mean_error_no_halo:.6e}")
print()

# ============================================
# Test 2: Divergence WITH Halo Exchange
# ============================================

print("5. Computing divergence WITH halo exchange...")
print("   (This should be correct everywhere!)")

# Compute divergence with halo exchange
div_with_halo = divergence(
    u_sharded, v_sharded, grid,
    partition_info=partition_info,
    perform_halo_exchange=True  # WITH HALO EXCHANGE
)

# Convert to numpy for analysis
div_with_halo_np = np.array(div_with_halo)

# Compute error
error_with_halo = np.abs(div_with_halo_np)
max_error_with_halo = error_with_halo.max()
mean_error_with_halo = error_with_halo[1:-1, 1:-1].mean()

print(f"   Max error: {max_error_with_halo:.6e}")
print(f"   Mean interior error: {mean_error_with_halo:.6e}")
print()

# ============================================
# Comparison and Analysis
# ============================================

print("6. Comparing results...")
print()

improvement = max_error_no_halo / max_error_with_halo if max_error_with_halo > 1e-10 else float('inf')

print(f"   Error WITHOUT halos: {max_error_no_halo:.6e}")
print(f"   Error WITH halos:    {max_error_with_halo:.6e}")
print(f"   Improvement factor:  {improvement:.1f}x")
print()

# ============================================
# Visualization
# ============================================

print("7. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Velocity field
    im1 = axes[0, 0].pcolormesh(x, y, np.array(u), cmap='RdBu_r', shading='auto')
    axes[0, 0].set_title('Velocity u-component', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].pcolormesh(x, y, np.array(v), cmap='RdBu_r', shading='auto')
    axes[0, 1].set_title('Velocity v-component', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1])

    # Quiver plot
    skip = 4
    axes[0, 2].quiver(
        x[::skip, ::skip], y[::skip, ::skip],
        np.array(u)[::skip, ::skip], np.array(v)[::skip, ::skip],
        scale=10, alpha=0.7
    )
    axes[0, 2].set_title('Velocity Field (divergence-free)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    axes[0, 2].set_aspect('equal')

    # Divergence without halos (errors at boundaries!)
    im3 = axes[1, 0].pcolormesh(x, y, div_no_halo_np, cmap='seismic',
                                vmin=-0.1, vmax=0.1, shading='auto')
    axes[1, 0].set_title('Divergence WITHOUT Halos\n(Errors at partition boundaries!)',
                        fontsize=12, fontweight='bold', color='red')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im3, ax=axes[1, 0])

    # Add partition boundary lines
    for i in range(1, n_partitions_x):
        x_line = i * (nx // n_partitions_x)
        axes[1, 0].axvline(x[0, x_line], color='yellow', linewidth=2, linestyle='--',
                          label='Partition boundary' if i == 1 else '')

    for j in range(1, n_partitions_y):
        y_line = j * (ny // n_partitions_y)
        axes[1, 0].axhline(y[y_line, 0], color='yellow', linewidth=2, linestyle='--')

    if n_partitions_x > 1 or n_partitions_y > 1:
        axes[1, 0].legend(loc='upper right', fontsize=8)

    # Divergence with halos (correct!)
    im4 = axes[1, 1].pcolormesh(x, y, div_with_halo_np, cmap='seismic',
                                vmin=-0.1, vmax=0.1, shading='auto')
    axes[1, 1].set_title('Divergence WITH Halos\n(Correct everywhere!)',
                        fontsize=12, fontweight='bold', color='green')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 1])

    # Error comparison
    im5 = axes[1, 2].pcolormesh(x, y, np.log10(error_no_halo + 1e-10),
                                cmap='hot_r', shading='auto')
    axes[1, 2].set_title(f'log10(Error) - Without Halos\nMax: {max_error_no_halo:.2e}',
                        fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 2])

    plt.tight_layout()
    output_file = "output/example_13_divergence_halo.png"
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
print("1. WHAT HALO EXCHANGE SOLVES:")
print("   - Stencil operations (divergence, gradient, Laplacian) need neighbor data")
print("   - At partition boundaries, neighbors are on different devices")
print("   - Without halo exchange → wrong neighbor values → wrong results")
print("   - With halo exchange → correct neighbor values → correct results")
print()
print("2. ERROR ANALYSIS:")
print(f"   - Without halos: max error = {max_error_no_halo:.2e}")
print(f"   - With halos:    max error = {max_error_with_halo:.2e}")
print(f"   - Improvement:   {improvement:.1f}x better with halo exchange")
print()
print("3. VISUALIZATION SHOWS:")
print("   - Yellow lines = partition boundaries")
print("   - Without halos: errors concentrated at boundaries")
print("   - With halos: errors uniformly small (numerical precision)")
print()
print("This demonstrates that halo exchange is ESSENTIAL for")
print("distributed stencil computations!")
print()
