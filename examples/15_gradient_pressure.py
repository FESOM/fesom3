"""
Example 15: Gradient Operator for Pressure Force

This example demonstrates:
- Gradient operator on different grid types (A, B, C)
- Computing pressure gradient force (critical in ocean/atmosphere models)
- How C-grid naturally aligns gradients with velocity locations
- Comparison of accuracy across grid types

Physical application: Pressure gradient force
    F = -∇p/ρ

where p is pressure and ρ is density.
In geophysical flows, this drives much of the motion!
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from fesomx import QuadGrid, StaggerType, get_backend, gradient
from fesomx.operators.gradient import smooth_gaussian_field_2d, analytical_gradient_gaussian

print("=" * 80)
print("Example 15: Gradient Operator for Pressure Gradient Force")
print("=" * 80)
print()

# ============================================
# Setup
# ============================================

print("1. Setting up grids (A, B, and C)...")

# Grid parameters
nx, ny = 64, 64
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

# Create grids with different staggerings
grids = {}
for stagger in [StaggerType.A, StaggerType.B, StaggerType.C]:
    grid = QuadGrid(
        name=f"pressure_grad_{stagger.name}",
        nx=nx,
        ny=ny,
        stagger_type=stagger,
        periodic=(False, False)
    )
    grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
    grid.dx = dx
    grid.dy = dy
    grids[stagger] = grid

print(f"   Grid: {nx} x {ny} cells")
print(f"   Grid spacing: dx = dy = {dx:.6f}")
print(f"   Created grids for A, B, and C staggering")
print()

# ============================================
# Setup JAX Sharding
# ============================================

print("2. Setting up JAX sharding...")

devices = jax.devices()
n_devices = len(devices)
device_mesh = mesh_utils.create_device_mesh((min(2, n_devices), 1))
mesh = Mesh(device_mesh, axis_names=('x', 'y'))

print(f"   Devices: {n_devices}")
print(f"   Device mesh: {device_mesh.shape}")
print()

backend = get_backend('jax_sharding')
for grid in grids.values():
    grid.backend = backend

# ============================================
# Create Pressure Field
# ============================================

print("3. Creating pressure field...")

# Get cell centers for pressure (same for all grid types)
centers = grids[StaggerType.A].get_cell_centers()
x = centers[:, 0].reshape(ny, nx)
y = centers[:, 1].reshape(ny, nx)

# Create pressure field: High pressure system at center
# Using smooth Gaussian
p_high = 1013.25  # High pressure (hPa)
p_ambient = 1000.0  # Ambient pressure (hPa)
sigma = 0.15

pressure = p_ambient + (p_high - p_ambient) * smooth_gaussian_field_2d(
    jnp.array(x), jnp.array(y),
    x0=0.5, y0=0.5, sigma=sigma, amplitude=1.0
)

print(f"   Pressure range: [{pressure.min():.2f}, {pressure.max():.2f}] hPa")
print(f"   High pressure center at (0.5, 0.5)")
print()

# Shard pressure field
with mesh:
    pressure_sharded = jax.device_put(
        pressure,
        jax.sharding.NamedSharding(mesh, P('x', 'y'))
    )

# Create partition info
partition_info = {
    'mesh': mesh,
    'partition_spec': P('x', 'y'),
    'shape': (ny, nx),
    'n_partitions': (min(2, n_devices), 1)
}

# ============================================
# Compute Pressure Gradient on All Grid Types
# ============================================

print("4. Computing pressure gradients on A, B, and C grids...")

gradients = {}
for stagger_type, grid in grids.items():
    print(f"\n   {stagger_type.name}-grid:")

    # Compute gradient
    dpdx, dpdy = gradient(
        pressure_sharded, grid,
        partition_info=partition_info,
        stagger_type=stagger_type,
        return_at_centers=True,  # Return at centers for easy comparison
        perform_halo_exchange=True
    )

    gradients[stagger_type] = (np.array(dpdx), np.array(dpdy))

    print(f"      dP/dx range: [{dpdx.min():.3f}, {dpdx.max():.3f}]")
    print(f"      dP/dy range: [{dpdy.min():.3f}, {dpdy.max():.3f}]")

print()

# ============================================
# Analytical Gradient for Validation
# ============================================

print("5. Computing analytical gradient for validation...")

# Analytical gradient of Gaussian pressure field
factor = (p_high - p_ambient)
dpdx_exact, dpdy_exact = analytical_gradient_gaussian(
    jnp.array(x), jnp.array(y),
    x0=0.5, y0=0.5, sigma=sigma, amplitude=factor
)

dpdx_exact_np = np.array(dpdx_exact)
dpdy_exact_np = np.array(dpdy_exact)

print(f"   Analytical dP/dx range: [{dpdx_exact_np.min():.3f}, {dpdx_exact_np.max():.3f}]")
print(f"   Analytical dP/dy range: [{dpdy_exact_np.min():.3f}, {dpdy_exact_np.max():.3f}]")
print()

# ============================================
# Error Analysis
# ============================================

print("6. Comparing numerical vs analytical gradients...")
print()

for stagger_type, (dpdx_num, dpdy_num) in gradients.items():
    # Compute errors (excluding boundaries)
    margin = 2
    error_x = np.abs(dpdx_num[margin:-margin, margin:-margin] -
                     dpdx_exact_np[margin:-margin, margin:-margin])
    error_y = np.abs(dpdy_num[margin:-margin, margin:-margin] -
                     dpdy_exact_np[margin:-margin, margin:-margin])

    max_error_x = error_x.max()
    max_error_y = error_y.max()
    mean_error_x = error_x.mean()
    mean_error_y = error_y.mean()

    print(f"   {stagger_type.name}-grid:")
    print(f"      Max error in dP/dx:  {max_error_x:.4f}")
    print(f"      Max error in dP/dy:  {max_error_y:.4f}")
    print(f"      Mean error in dP/dx: {mean_error_x:.4f}")
    print(f"      Mean error in dP/dy: {mean_error_y:.4f}")
    print()

# ============================================
# Compute Pressure Gradient Force
# ============================================

print("7. Computing pressure gradient force...")

# Typical density for atmosphere
rho = 1.225  # kg/m³

forces = {}
for stagger_type, (dpdx, dpdy) in gradients.items():
    # Convert pressure gradient to force per unit mass
    # F = -∇p/ρ (units: (hPa/m) / (kg/m³) → m/s² after unit conversion)
    # 1 hPa = 100 Pa = 100 N/m²

    F_x = -dpdx * 100.0 / rho  # m/s²
    F_y = -dpdy * 100.0 / rho  # m/s²

    forces[stagger_type] = (F_x, F_y)

print(f"   Density: ρ = {rho} kg/m³")
print(f"   Force: F = -∇p/ρ")
print()

for stagger_type, (F_x, F_y) in forces.items():
    F_magnitude = np.sqrt(F_x**2 + F_y**2)
    print(f"   {stagger_type.name}-grid force magnitude: max = {F_magnitude.max():.3f} m/s²")

print()

# ============================================
# Visualization
# ============================================

print("8. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    # Figure 1: Pressure and analytical gradient
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))

    # Pressure field
    im1 = axes1[0].pcolormesh(x, y, np.array(pressure), cmap='RdYlBu_r', shading='auto')
    axes1[0].set_title('Pressure Field (hPa)', fontsize=12, fontweight='bold')
    axes1[0].set_xlabel('x')
    axes1[0].set_ylabel('y')
    axes1[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes1[0], label='Pressure (hPa)')

    # Analytical dP/dx
    im2 = axes1[1].pcolormesh(x, y, dpdx_exact_np, cmap='RdBu_r',
                              vmin=-dpdx_exact_np.max(), vmax=dpdx_exact_np.max(),
                              shading='auto')
    axes1[1].set_title('Analytical ∂P/∂x', fontsize=12, fontweight='bold')
    axes1[1].set_xlabel('x')
    axes1[1].set_ylabel('y')
    axes1[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes1[1], label='hPa/m')

    # Analytical dP/dy
    im3 = axes1[2].pcolormesh(x, y, dpdy_exact_np, cmap='RdBu_r',
                              vmin=-dpdy_exact_np.max(), vmax=dpdy_exact_np.max(),
                              shading='auto')
    axes1[2].set_title('Analytical ∂P/∂y', fontsize=12, fontweight='bold')
    axes1[2].set_xlabel('x')
    axes1[2].set_ylabel('y')
    axes1[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes1[2], label='hPa/m')

    plt.tight_layout()
    output_file1 = "output/example_15_pressure_analytical.png"
    Path("output").mkdir(exist_ok=True)
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"   Analytical gradient saved to: {output_file1}")

    # Figure 2: Comparison across grid types
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))

    for idx, stagger_type in enumerate([StaggerType.A, StaggerType.B, StaggerType.C]):
        dpdx_num, dpdy_num = gradients[stagger_type]

        # dP/dx numerical
        im_x = axes2[0, idx].pcolormesh(x, y, dpdx_num, cmap='RdBu_r',
                                        vmin=-dpdx_exact_np.max(),
                                        vmax=dpdx_exact_np.max(), shading='auto')
        axes2[0, idx].set_title(f'{stagger_type.name}-grid: ∂P/∂x',
                               fontsize=12, fontweight='bold')
        axes2[0, idx].set_xlabel('x')
        axes2[0, idx].set_ylabel('y')
        axes2[0, idx].set_aspect('equal')
        plt.colorbar(im_x, ax=axes2[0, idx], label='hPa/m')

        # Error in dP/dx
        error_x = np.abs(dpdx_num - dpdx_exact_np)
        im_err = axes2[1, idx].pcolormesh(x, y, error_x, cmap='hot_r', shading='auto')
        axes2[1, idx].set_title(f'{stagger_type.name}-grid: |Error in ∂P/∂x|',
                               fontsize=12, fontweight='bold')
        axes2[1, idx].set_xlabel('x')
        axes2[1, idx].set_ylabel('y')
        axes2[1, idx].set_aspect('equal')
        plt.colorbar(im_err, ax=axes2[1, idx], label='Error (hPa/m)')

    plt.tight_layout()
    output_file2 = "output/example_15_grid_comparison.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"   Grid comparison saved to: {output_file2}")

    # Figure 3: Pressure gradient force vectors
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))

    skip = 4
    for idx, stagger_type in enumerate([StaggerType.A, StaggerType.B, StaggerType.C]):
        F_x, F_y = forces[stagger_type]

        # Background: pressure
        axes3[idx].pcolormesh(x, y, np.array(pressure), cmap='RdYlBu_r',
                             alpha=0.3, shading='auto')

        # Vectors: force
        axes3[idx].quiver(
            x[::skip, ::skip], y[::skip, ::skip],
            F_x[::skip, ::skip], F_y[::skip, ::skip],
            scale=200, width=0.003, alpha=0.8, color='black'
        )

        axes3[idx].set_title(f'{stagger_type.name}-grid: Force F = -∇p/ρ',
                            fontsize=12, fontweight='bold')
        axes3[idx].set_xlabel('x')
        axes3[idx].set_ylabel('y')
        axes3[idx].set_aspect('equal')

    plt.tight_layout()
    output_file3 = "output/example_15_pressure_force.png"
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"   Force vectors saved to: {output_file3}")
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
print("1. GRADIENT OPERATOR:")
print("   - Computes ∇p = (∂p/∂x, ∂p/∂y) for pressure field")
print("   - Essential for pressure gradient force in geophysical flows")
print("   - Works on A, B, and C grids with appropriate staggering")
print()
print("2. GRID TYPE COMPARISON:")
print("   - A-grid: Gradient at cell centers (simple, but not optimal)")
print("   - B-grid: Gradient naturally at corners (good for vorticity)")
print("   - C-grid: Gradient at faces (PERFECT for ocean/atmosphere models!)")
print()
print("3. C-GRID ADVANTAGE:")
print("   - In C-grid: u at x-faces, v at y-faces")
print("   - Gradient gives ∂p/∂x at x-faces, ∂p/∂y at y-faces")
print("   - NO INTERPOLATION needed for pressure force!")
print("   - This is why ocean models (MOM6, MITgcm, NEMO) use C-grid")
print()
print("4. PRESSURE GRADIENT FORCE:")
print("   - F = -∇p/ρ drives atmospheric and oceanic flows")
print("   - High pressure → low pressure flow")
print("   - Vectors point away from high pressure center")
print()
