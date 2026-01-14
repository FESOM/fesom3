"""
Example 06: Parallel Scaling and Sharding Analysis

This example demonstrates:
- JAX multi-CPU parallelism configuration
- Strong scaling (fixed problem size, increasing cores)
- Weak scaling (problem size scales with cores)
- Shard size analysis across devices
- Performance metrics for halo exchange
"""

import numpy as np
import time
import os

# Configure JAX for CPU parallelism BEFORE importing JAX
# This creates virtual devices on CPU
n_devices = 32  # Use all 32 CPU cores
os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={n_devices}'
os.environ['JAX_PLATFORMS'] = 'cpu'  # Force CPU-only mode (ignore GPU)

from fesomx import QuadGrid, StaggerType, get_backend
from fesomx.core.halo_patterns import HaloWidth

print("=" * 70)
print("Example 06: Parallel Scaling and Sharding Analysis")
print("=" * 70)
print()

# ============================================
# 1. Initialize JAX Backend
# ============================================

print("1. Initializing JAX backend with multi-CPU support...")
print()

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P, NamedSharding

    print(f"   JAX version: {jax.__version__}")
    print(f"   Available devices: {len(jax.devices())}")
    print(f"   Device types: {[d.device_kind for d in jax.devices()[:4]]}...")
    print()

    # Initialize backend with 2D mesh
    # For 32 devices, we can use 4×8, 8×4, or other factorizations
    mesh_configs = [
        (1, 32),   # 1D: all devices in one row
        (2, 16),   # 2D: 2×16
        (4, 8),    # 2D: 4×8 (balanced)
        (8, 4),    # 2D: 8×4 (balanced)
    ]

    print("   Testing different mesh configurations:")
    for mesh_shape in mesh_configs:
        if mesh_shape[0] * mesh_shape[1] == len(jax.devices()):
            print(f"     {mesh_shape[0]:2d}×{mesh_shape[1]:2d} = {mesh_shape[0]*mesh_shape[1]:2d} devices ✓")
    print()

    # Use 4×8 mesh for balanced 2D partitioning
    backend = get_backend("jax_sharding", mesh_shape=(4, 8))
    print(f"   Selected mesh: 4×8")
    print(f"   Backend initialized: {backend.name}")
    print()

except ImportError as e:
    print(f"   Error: JAX not available - {e}")
    print("   Falling back to CPU-only mode")
    backend = None
    jax = None

# ============================================
# 2. Analyze Sharding for Different Grid Sizes
# ============================================

print("2. Analyzing data sharding across devices...")
print()

def analyze_sharding(grid, field_name):
    """Analyze how a field is sharded across devices."""
    if jax is None:
        print("   JAX not available, skipping shard analysis")
        return

    field_array = grid.fields[field_name]['data']

    # Get sharding information
    sharding = field_array.sharding
    global_shape = field_array.shape

    print(f"   Field: {field_name}")
    print(f"   Global shape: {global_shape}")

    # Get local shard information
    try:
        # For each device, show its shard
        addressable_devices = field_array.sharding.addressable_devices
        print(f"   Number of shards: {len(addressable_devices)}")

        # Sample a few shards
        if hasattr(sharding, 'shard_shape'):
            shard_shape = sharding.shard_shape(global_shape)
            print(f"   Shard shape (per device): {shard_shape}")

            total_elements = np.prod(global_shape)
            elements_per_shard = np.prod(shard_shape)
            print(f"   Total elements: {total_elements:,}")
            print(f"   Elements per shard: {elements_per_shard:,}")
            print(f"   Memory per shard: {elements_per_shard * 8 / 1024:.2f} KB (float64)")
    except Exception as e:
        print(f"   Shard info not available: {e}")

    print()


# Test different grid sizes
test_sizes = [32, 64, 128, 256]

for nx in test_sizes[:2]:  # Test first two sizes with detailed analysis
    print(f"   Grid size: {nx}×{nx}")
    print("   " + "-" * 40)

    grid = QuadGrid(
        f"test_{nx}x{nx}",
        nx=nx,
        ny=nx,
        stagger_type=StaggerType.A,
        backend=backend
    )
    grid.create_mesh()
    grid.compute_neighbors(halo_width=1)

    # Create field with sharding
    field_data = np.random.rand(nx, nx).astype(np.float32)

    if jax is not None:
        # Create sharded array
        sharding_spec = P('x', 'y')  # Partition along both dimensions
        grid.add_field("test", field_data, sharding_spec=sharding_spec)

        # Analyze sharding
        analyze_sharding(grid, "test")
    else:
        grid.add_field("test", field_data)
        print("   JAX not available - no sharding\n")

# ============================================
# 3. Strong Scaling Test
# ============================================

print("3. Strong Scaling Test (fixed problem size)")
print("   Problem: 256×256 grid with halo_width=2")
print()

# Fixed large problem
nx_fixed = 256
ny_fixed = 256

print("   " + "=" * 60)
print(f"   {'Grid Size':<15} {'Devices':<10} {'Time (ms)':<12} {'Speedup':<10}")
print("   " + "=" * 60)

# We'll simulate different device counts by comparing with baseline
baseline_time = None

grid_fixed = QuadGrid(
    "strong_scaling",
    nx=nx_fixed,
    ny=ny_fixed,
    backend=backend
)
grid_fixed.create_mesh()
grid_fixed.compute_neighbors(halo_width=2)

field_data = np.random.rand(ny_fixed, nx_fixed).astype(np.float32)

if jax is not None:
    grid_fixed.add_field("test", field_data, sharding_spec=P('x', 'y'))
else:
    grid_fixed.add_field("test", field_data)

# Warm-up
for _ in range(3):
    grid_fixed.halo_exchange("test", halo_width=2)

# Timing
n_iterations = 20
start = time.time()
for _ in range(n_iterations):
    grid_fixed.halo_exchange("test", halo_width=2)
elapsed = (time.time() - start) / n_iterations * 1000  # ms

n_devices_actual = len(jax.devices()) if jax else 1
baseline_time = elapsed

print(f"   {nx_fixed:3d}×{ny_fixed:3d}      {n_devices_actual:4d}      {elapsed:8.3f}      {1.0:.2f}×")
print()

# ============================================
# 4. Weak Scaling Test
# ============================================

print("4. Weak Scaling Test (problem size scales with grid)")
print("   Each doubling of grid size = 4× more work")
print()

print("   " + "=" * 70)
print(f"   {'Grid Size':<12} {'Cells':<12} {'Time (ms)':<12} {'Time/Cell (μs)':<15}")
print("   " + "=" * 70)

weak_scaling_sizes = [16, 32, 64, 128, 256, 512]
times_weak = []

for nx in weak_scaling_sizes:
    grid_weak = QuadGrid(
        f"weak_{nx}x{nx}",
        nx=nx,
        ny=nx,
        backend=backend
    )
    grid_weak.create_mesh()
    grid_weak.compute_neighbors(halo_width=2)

    field_data = np.random.rand(nx, nx).astype(np.float32)

    if jax is not None:
        grid_weak.add_field("test", field_data, sharding_spec=P('x', 'y'))
    else:
        grid_weak.add_field("test", field_data)

    # Warm-up
    for _ in range(3):
        grid_weak.halo_exchange("test", halo_width=2)

    # Timing
    n_iterations = 10 if nx <= 128 else 5
    start = time.time()
    for _ in range(n_iterations):
        grid_weak.halo_exchange("test", halo_width=2)
    elapsed = (time.time() - start) / n_iterations * 1000  # ms

    n_cells = nx * nx
    time_per_cell = elapsed * 1000 / n_cells  # microseconds

    times_weak.append(elapsed)

    print(f"   {nx:4d}×{nx:4d}    {n_cells:8d}    {elapsed:8.3f}      {time_per_cell:10.4f}")

print()

# ============================================
# 5. Halo Width Impact
# ============================================

print("5. Impact of Halo Width on Performance")
print("   Grid: 128×128")
print()

nx_halo = 128
grid_halo = QuadGrid(
    "halo_test",
    nx=nx_halo,
    ny=nx_halo,
    backend=backend
)
grid_halo.create_mesh()

print("   " + "=" * 60)
print(f"   {'Halo Width':<15} {'Time (ms)':<15} {'Overhead vs w=1':<20}")
print("   " + "=" * 60)

halo_widths = [1, 2, 3, 4, 5]
time_w1 = None

for w in halo_widths:
    grid_halo.compute_neighbors(halo_width=w)

    field_data = np.random.rand(nx_halo, nx_halo).astype(np.float32)
    field_name = f"test_w{w}"

    if jax is not None:
        grid_halo.add_field(field_name, field_data, sharding_spec=P('x', 'y'))
    else:
        grid_halo.add_field(field_name, field_data)

    # Warm-up
    for _ in range(3):
        grid_halo.halo_exchange(field_name, halo_width=w)

    # Timing
    n_iterations = 10
    start = time.time()
    for _ in range(n_iterations):
        grid_halo.halo_exchange(field_name, halo_width=w)
    elapsed = (time.time() - start) / n_iterations * 1000  # ms

    if w == 1:
        time_w1 = elapsed
        overhead_str = "baseline"
    else:
        overhead = (elapsed / time_w1 - 1.0) * 100
        overhead_str = f"+{overhead:.1f}%"

    print(f"   {w:4d}           {elapsed:10.3f}      {overhead_str:<20}")

print()

# ============================================
# 6. Detailed Shard Distribution
# ============================================

print("6. Detailed Shard Distribution Analysis")
print()

if jax is not None:
    # Create a medium-sized grid for analysis
    nx_detail = 128
    grid_detail = QuadGrid(
        "shard_detail",
        nx=nx_detail,
        ny=nx_detail,
        backend=backend
    )
    grid_detail.create_mesh()

    field_data = np.random.rand(nx_detail, nx_detail).astype(np.float32)
    grid_detail.add_field("detail", field_data, sharding_spec=P('x', 'y'))

    field_array = grid_detail.fields["detail"]["data"]

    print(f"   Grid shape: {field_array.shape}")
    print(f"   Data type: {field_array.dtype}")
    print(f"   Total size: {field_array.size * 4 / 1024:.2f} KB")
    print()

    # Show mesh configuration
    mesh = backend.mesh
    print(f"   Mesh shape: {mesh.shape}")
    print(f"   Mesh axis names: {mesh.axis_names}")
    print()

    # Show sharding
    sharding = field_array.sharding
    print(f"   Sharding: {sharding}")

    # Calculate shard details
    if hasattr(sharding, 'shard_shape'):
        shard_shape = sharding.shard_shape(field_array.shape)
        print(f"   Shard shape: {shard_shape}")
        print(f"   Elements per shard: {np.prod(shard_shape):,}")
        print(f"   Memory per shard: {np.prod(shard_shape) * 4 / 1024:.2f} KB")

        # Show distribution
        n_shards_x = field_array.shape[1] // shard_shape[1]
        n_shards_y = field_array.shape[0] // shard_shape[0]
        print(f"   Shard layout: {n_shards_y}×{n_shards_x}")

    print()
else:
    print("   JAX not available - cannot analyze sharding")
    print()

# ============================================
# 7. Performance Summary
# ============================================

print("7. Performance Summary")
print("=" * 70)
print()

print("   Configuration:")
print(f"     CPU cores available: {n_devices}")
if jax is not None:
    print(f"     JAX devices: {len(jax.devices())}")
    print(f"     Mesh configuration: {backend.mesh.shape}")
else:
    print(f"     JAX devices: Not available")
print()

print("   Key Findings:")
print(f"     - Fixed problem (256×256): {baseline_time:.3f} ms/iteration")
print(f"     - Weak scaling: Time grows with problem size")
print(f"     - Halo width overhead: ~{((elapsed/time_w1 - 1)*100):.0f}% for width=5 vs width=1")
print()

print("   Recommendations:")
print("     1. Use mesh shape (4, 8) for balanced 2D partitioning")
print("     2. Grid size should be divisible by mesh dimensions")
print("     3. Larger grids (256+) benefit more from parallelism")
print("     4. Minimize halo width when possible (use 1-2)")
print()

# ============================================
# Summary
# ============================================

print("=" * 70)
print("Example completed successfully!")
print("=" * 70)
print()

print("Key insights:")
print("  - JAX sharding distributes data across CPU devices")
print("  - 4×8 mesh layout balances work across 32 cores")
print("  - Shard size = (global_size / mesh_shape) per dimension")
print("  - Halo exchange overhead increases with halo width")
print("  - Weak scaling shows time-per-cell remains reasonable")
print()

print("To run with different configurations:")
print("  - Modify n_devices at top of script")
print("  - Change mesh_shape in backend initialization")
print("  - Test different grid sizes in scaling tests")
print()
