"""
Example 14: Laplacian and Heat Diffusion

This example demonstrates:
- Laplacian operator for heat equation
- Time evolution of temperature field
- Importance of halo exchange for time-stepping
- Comparison of 2nd vs 4th order accuracy

Physical application: Heat diffusion equation
    ∂T/∂t = α∇²T

where α is thermal diffusivity.
"""

import numpy as np
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental import mesh_utils

from fesomx import QuadGrid, StaggerType, get_backend, laplacian
from fesomx.operators.laplacian import heat_diffusion_step

print("=" * 80)
print("Example 14: Heat Diffusion using Laplacian Operator")
print("=" * 80)
print()

# ============================================
# Setup
# ============================================

print("1. Setting up grid and parameters...")

# Grid parameters
nx, ny = 128, 128
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)

# Physical parameters
alpha = 0.01  # Thermal diffusivity
dt = 0.0001   # Timestep (must satisfy CFL condition: dt < dx²/(4α))
n_steps = 200  # Number of timesteps

# CFL check
dt_max = dx**2 / (4 * alpha)
print(f"   Grid: {nx} x {ny}")
print(f"   Grid spacing: dx = dy = {dx:.6f}")
print(f"   Thermal diffusivity: α = {alpha}")
print(f"   Timestep: dt = {dt:.6f}")
print(f"   Max stable dt (CFL): {dt_max:.6f}")

if dt > dt_max:
    print(f"   WARNING: dt > dt_max! Simulation may be unstable!")
else:
    print(f"   ✓ CFL condition satisfied (stable)")

print(f"   Total time steps: {n_steps}")
print(f"   Final time: t = {n_steps * dt:.6f}")
print()

# Create grid
grid = QuadGrid(
    name="heat_diffusion",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,
    periodic=(False, False)
)

grid.create_mesh(domain=(0.0, 1.0, 0.0, 1.0))
grid.dx = dx
grid.dy = dy

# ============================================
# Initial Condition
# ============================================

print("2. Setting initial temperature distribution...")

# Get cell centers
centers = grid.get_cell_centers()
x = centers[:, 0].reshape(ny, nx)
y = centers[:, 1].reshape(ny, nx)

# Initial condition: Hot spot at center + cool edges
# Gaussian temperature distribution
x0, y0 = 0.5, 0.5
sigma = 0.1
T_hot = 100.0  # Temperature at center
T_cold = 20.0  # Temperature at edges

T_initial = T_cold + (T_hot - T_cold) * jnp.exp(
    -((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)
)

print(f"   Initial temperature range: [{T_initial.min():.1f}, {T_initial.max():.1f}]")
print(f"   Hot spot at ({x0}, {y0}) with σ = {sigma}")
print()

# ============================================
# Setup JAX Sharding
# ============================================

print("3. Setting up JAX sharding...")

devices = jax.devices()
n_devices = len(devices)
n_partitions_x = min(2, n_devices)
n_partitions_y = 1

if n_devices >= 2:
    device_mesh = mesh_utils.create_device_mesh((n_partitions_x, n_partitions_y))
else:
    device_mesh = mesh_utils.create_device_mesh((1, 1))

mesh = Mesh(device_mesh, axis_names=('x', 'y'))

print(f"   Devices: {n_devices}")
print(f"   Device mesh: {device_mesh.shape}")
print()

# Shard temperature field
with mesh:
    T_sharded = jax.device_put(T_initial, jax.sharding.NamedSharding(mesh, P('x', 'y')))

# Create partition info
partition_info = {
    'mesh': mesh,
    'partition_spec': P('x', 'y'),
    'shape': (ny, nx),
    'n_partitions': (n_partitions_x, n_partitions_y)
}

# Set backend
backend = get_backend('jax_sharding')
grid.backend = backend

# ============================================
# Time Evolution (2nd order Laplacian)
# ============================================

print("4. Evolving heat equation (2nd order accurate)...")
print("   ∂T/∂t = α∇²T")

T_current = T_sharded
T_history = [np.array(T_current)]
times = [0.0]

# Record snapshots at specific times
snapshot_times = [0, 50, 100, 150, 200]

for step in range(n_steps):
    # One timestep
    T_current = heat_diffusion_step(
        T_current, grid, alpha, dt,
        partition_info=partition_info
    )

    # Record snapshots
    if step + 1 in snapshot_times:
        T_history.append(np.array(T_current))
        times.append((step + 1) * dt)

    # Progress
    if (step + 1) % 50 == 0:
        T_np = np.array(T_current)
        print(f"   Step {step+1}/{n_steps}: "
              f"T ∈ [{T_np.min():.1f}, {T_np.max():.1f}], "
              f"mean = {T_np.mean():.1f}")

print()

# ============================================
# Time Evolution (4th order Laplacian)
# ============================================

print("5. Comparing with 4th order accurate Laplacian...")

T_current_4th = T_sharded
T_history_4th = [np.array(T_current_4th)]

for step in range(n_steps):
    # Compute Laplacian with 4th order accuracy
    lapl_T = laplacian(
        T_current_4th, grid,
        partition_info=partition_info,
        order=4  # 4th order!
    )

    # Forward Euler step
    T_current_4th = T_current_4th + alpha * dt * lapl_T

    # Record snapshots
    if step + 1 in snapshot_times:
        T_history_4th.append(np.array(T_current_4th))

print(f"   Completed {n_steps} steps with 4th order Laplacian")
print()

# ============================================
# Analysis
# ============================================

print("6. Analyzing results...")

# Compare 2nd vs 4th order
T_final_2nd = T_history[-1]
T_final_4th = T_history_4th[-1]

diff = np.abs(T_final_2nd - T_final_4th)
max_diff = diff.max()
mean_diff = diff.mean()

print(f"   Final temperature (2nd order): mean = {T_final_2nd.mean():.2f}")
print(f"   Final temperature (4th order): mean = {T_final_4th.mean():.2f}")
print(f"   Max difference (2nd vs 4th): {max_diff:.4f}")
print(f"   Mean difference: {mean_diff:.4f}")
print()

# Energy conservation check (total heat should be conserved in periodic BC)
# For non-periodic BC with diffusion, total heat decreases (flux out of boundaries)
initial_heat = T_initial.sum()
final_heat_2nd = T_final_2nd.sum()
final_heat_4th = T_final_4th.sum()

print(f"   Initial total heat: {initial_heat:.1f}")
print(f"   Final heat (2nd order): {final_heat_2nd:.1f} ({100*final_heat_2nd/initial_heat:.1f}%)")
print(f"   Final heat (4th order): {final_heat_4th:.1f} ({100*final_heat_4th/initial_heat:.1f}%)")
print()

# ============================================
# Visualization
# ============================================

print("7. Creating visualizations...")

try:
    import matplotlib.pyplot as plt

    # Temperature evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    vmin, vmax = T_cold, T_hot

    for idx, (T, t) in enumerate(zip(T_history[:6], times[:6])):
        row = idx // 3
        col = idx % 3
        im = axes[row, col].pcolormesh(x, y, T, cmap='hot', vmin=vmin, vmax=vmax,
                                       shading='auto')
        axes[row, col].set_title(f'Temperature at t = {t:.5f}', fontweight='bold')
        axes[row, col].set_xlabel('x')
        axes[row, col].set_ylabel('y')
        axes[row, col].set_aspect('equal')
        plt.colorbar(im, ax=axes[row, col], label='Temperature')

    plt.tight_layout()
    output_file1 = "output/example_14_diffusion_evolution.png"
    Path("output").mkdir(exist_ok=True)
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"   Evolution plot saved to: {output_file1}")

    # Comparison: 2nd vs 4th order
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes2[0].pcolormesh(x, y, T_final_2nd, cmap='hot', vmin=vmin, vmax=vmax,
                             shading='auto')
    axes2[0].set_title('Final T (2nd order Laplacian)', fontweight='bold')
    axes2[0].set_xlabel('x')
    axes2[0].set_ylabel('y')
    axes2[0].set_aspect('equal')
    plt.colorbar(im1, ax=axes2[0], label='Temperature')

    im2 = axes2[1].pcolormesh(x, y, T_final_4th, cmap='hot', vmin=vmin, vmax=vmax,
                             shading='auto')
    axes2[1].set_title('Final T (4th order Laplacian)', fontweight='bold')
    axes2[1].set_xlabel('x')
    axes2[1].set_ylabel('y')
    axes2[1].set_aspect('equal')
    plt.colorbar(im2, ax=axes2[1], label='Temperature')

    im3 = axes2[2].pcolormesh(x, y, diff, cmap='viridis', shading='auto')
    axes2[2].set_title(f'|Difference| (max = {max_diff:.4f})', fontweight='bold')
    axes2[2].set_xlabel('x')
    axes2[2].set_ylabel('y')
    axes2[2].set_aspect('equal')
    plt.colorbar(im3, ax=axes2[2], label='|Diff|')

    plt.tight_layout()
    output_file2 = "output/example_14_order_comparison.png"
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"   Comparison plot saved to: {output_file2}")
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
print("1. LAPLACIAN FOR DIFFUSION:")
print("   - Heat equation: ∂T/∂t = α∇²T")
print("   - Laplacian operator enables physical diffusion simulation")
print("   - Hot spots spread out and cool down over time")
print()
print("2. NUMERICAL ACCURACY:")
print(f"   - 2nd order: Standard 5-point stencil")
print(f"   - 4th order: More accurate 9-point stencil")
print(f"   - Difference in final state: {max_diff:.4f} (small but measurable)")
print()
print("3. STABILITY:")
print(f"   - CFL condition: dt < dx²/(4α) = {dt_max:.6f}")
print(f"   - Used dt = {dt} ✓")
print()
print("4. HALO EXCHANGE:")
print("   - Laplacian requires neighbor data at every timestep")
print("   - Automatic halo exchange ensures correct evolution")
print("   - Without halos: errors accumulate catastrophically over time!")
print()
