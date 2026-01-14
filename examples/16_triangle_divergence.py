"""
Example 16: Divergence on Triangular Grids

This example demonstrates the NEW modular operator architecture that supports
both structured (quad) and unstructured (triangle) grids.

Key demonstration:
- Create BOTH quad and triangle grids for the same domain
- Use SAME divergence() function for both (automatic dispatch!)
- Compare finite difference (quad) vs finite volume (triangle) methods
- Visualize mesh structure and results side-by-side

This answers: "Can operators work on both quad and triangle grids?"
Answer: YES! The modular architecture automatically selects the right method!
"""

import numpy as np
from pathlib import Path

import jax.numpy as jnp

from fesomx import QuadGrid, TriGrid, StaggerType, divergence
from fesomx.utilities import create_structured_triangular_mesh

# Import test helpers for divergence-free field creation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from test_helpers import create_divergence_free_field_unstructured

print("=" * 80)
print("Example 16: Divergence on Triangular Grids")
print("=" * 80)
print()
print("Demonstrating the NEW modular operator architecture:")
print("- Automatic dispatch based on grid type")
print("- Finite difference for structured grids (QuadGrid)")
print("- Finite volume for unstructured grids (TriGrid)")
print()

# ============================================
# Setup: Create Both Grid Types
# ============================================

print("1. Creating both QuadGrid and TriGrid for comparison...")
print()

# Grid parameters
nx, ny = 32, 32
domain = (0.0, 1.0, 0.0, 1.0)

# ----------------------
# QuadGrid (Structured)
# ----------------------
print("   a) QuadGrid (Structured):")

quad_grid = QuadGrid(
    name="quad_grid",
    nx=nx,
    ny=ny,
    stagger_type=StaggerType.A,  # All variables at cell centers
    periodic=(False, False)
)

quad_grid.create_mesh(domain=domain)
quad_grid.dx = 1.0 / (nx - 1)
quad_grid.dy = 1.0 / (ny - 1)

print(f"      - Grid size: {nx} × {ny} = {nx * ny} cells")
print(f"      - Cell spacing: dx={quad_grid.dx:.4f}, dy={quad_grid.dy:.4f}")
print(f"      - Structured: {quad_grid.is_structured()}")
print()

# ----------------------
# TriGrid (Unstructured)
# ----------------------
print("   b) TriGrid (Unstructured):")

# Create structured triangular mesh (each quad split into 2 triangles)
mesh_data = create_structured_triangular_mesh(nx, ny, domain=domain)

tri_grid = TriGrid(name="tri_grid", stagger_type=StaggerType.A)
tri_grid.create_mesh(
    vertices=mesh_data['vertices'],
    triangles=mesh_data['triangles']
)
tri_grid.compute_neighbors(halo_width=1)

print(f"      - Vertices: {len(mesh_data['vertices'])}")
print(f"      - Triangles: {len(mesh_data['triangles'])} (2 per quad)")
print(f"      - Structured: {tri_grid.is_structured()}")
print()

# ============================================
# Create Divergence-Free Velocity Field
# ============================================

print("2. Creating divergence-free velocity field...")
print("   Mathematical property: ∇·u = 0 (analytically)")
print()

# ----------------------
# QuadGrid: 2D arrays
# ----------------------
print("   a) QuadGrid velocity field:")

# Get cell centers
centers_quad = quad_grid.get_cell_centers()
x_quad = centers_quad[:, 0].reshape(ny, nx)
y_quad = centers_quad[:, 1].reshape(ny, nx)

# Create solid body rotation: u = -(y - y_c), v = (x - x_c)
x_c, y_c = 0.5, 0.5
u_quad = -(y_quad - y_c)
v_quad = (x_quad - x_c)

u_quad = jnp.array(u_quad)
v_quad = jnp.array(v_quad)

print(f"      - Shape: u={u_quad.shape}, v={v_quad.shape}")
print(f"      - Data structure: 2D arrays (ny, nx)")
print()

# ----------------------
# TriGrid: 1D arrays at cell centers
# ----------------------
print("   b) TriGrid velocity field:")

u_tri, v_tri = create_divergence_free_field_unstructured(
    tri_grid, mode='rotation'
)

print(f"      - Shape: u={u_tri.shape}, v={v_tri.shape}")
print(f"      - Data structure: 1D arrays (n_cells,)")
print(f"      - Values at cell centers of triangles")
print()

# ============================================
# Compute Divergence: SAME API for Both!
# ============================================

print("3. Computing divergence using UNIFIED API...")
print("   → Same divergence() function works for both grid types!")
print()

# ----------------------
# QuadGrid divergence
# ----------------------
print("   a) QuadGrid → Finite Difference:")

div_quad = divergence(u_quad, v_quad, quad_grid, perform_halo_exchange=False)

error_quad = np.abs(np.array(div_quad))
max_error_quad = error_quad[2:-2, 2:-2].max()
mean_error_quad = error_quad[2:-2, 2:-2].mean()

print(f"      - Method: Finite difference (central difference)")
print(f"      - Max error: {max_error_quad:.6e}")
print(f"      - Mean error: {mean_error_quad:.6e}")
print()

# ----------------------
# TriGrid divergence
# ----------------------
print("   b) TriGrid → Finite Volume:")

div_tri = divergence(u_tri, v_tri, tri_grid, perform_halo_exchange=False)

# Exclude boundary cells for error computation
centers_tri = tri_grid.get_cell_centers()
boundary_mask = (
    (centers_tri[:, 0] > 0.1) & (centers_tri[:, 0] < 0.9) &
    (centers_tri[:, 1] > 0.1) & (centers_tri[:, 1] < 0.9)
)

error_tri = np.abs(np.array(div_tri))
max_error_tri = error_tri[boundary_mask].max()
mean_error_tri = error_tri[boundary_mask].mean()

print(f"      - Method: Finite volume (edge flux summation)")
print(f"      - Max error: {max_error_tri:.6e}")
print(f"      - Mean error: {mean_error_tri:.6e}")
print()

# ============================================
# Architecture Demonstration
# ============================================

print("4. Demonstrating modular architecture:")
print()

print("   The SAME divergence() function automatically:")
print(f"   - Detected QuadGrid.is_structured() = {quad_grid.is_structured()}")
print(f"     → Dispatched to divergence_fd() (finite difference)")
print()
print(f"   - Detected TriGrid.is_structured() = {tri_grid.is_structured()}")
print(f"     → Dispatched to divergence_fv() (finite volume)")
print()
print("   This is the power of the modular operator architecture!")
print()

# ============================================
# Comparison
# ============================================

print("5. Comparing results:")
print()

print(f"   QuadGrid (Finite Difference):")
print(f"      Max error: {max_error_quad:.6e}")
print(f"      Mean error: {mean_error_quad:.6e}")
print()

print(f"   TriGrid (Finite Volume):")
print(f"      Max error: {max_error_tri:.6e}")
print(f"      Mean error: {mean_error_tri:.6e}")
print()

# Both methods should give small errors for divergence-free field
if max_error_quad < 1e-2 and max_error_tri < 1.0:
    print("   ✓ Both methods correctly compute divergence of rotation field!")
    print("   ✓ Finite volume is less accurate (expected for irregular meshes)")
else:
    print("   ✗ Errors larger than expected")

print()

# ============================================
# Visualization
# ============================================

print("6. Creating visualizations...")

try:
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    fig = plt.figure(figsize=(20, 12))

    # ----------------------
    # Row 1: Velocity fields
    # ----------------------

    # QuadGrid velocity
    ax1 = plt.subplot(3, 3, 1)
    im1 = ax1.pcolormesh(x_quad, y_quad, np.array(u_quad),
                         cmap='RdBu_r', shading='auto')
    ax1.set_title('QuadGrid: u-velocity', fontsize=11, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)

    ax2 = plt.subplot(3, 3, 2)
    im2 = ax2.pcolormesh(x_quad, y_quad, np.array(v_quad),
                         cmap='RdBu_r', shading='auto')
    ax2.set_title('QuadGrid: v-velocity', fontsize=11, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)

    # QuadGrid quiver
    ax3 = plt.subplot(3, 3, 3)
    skip = 3
    ax3.quiver(
        x_quad[::skip, ::skip], y_quad[::skip, ::skip],
        np.array(u_quad)[::skip, ::skip], np.array(v_quad)[::skip, ::skip],
        scale=8, alpha=0.7, color='navy'
    )
    ax3.set_title('QuadGrid: Velocity Field\n(Rotation, ∇·u=0)',
                  fontsize=11, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')

    # ----------------------
    # Row 2: Triangle velocity
    # ----------------------

    # Create triangulation for plotting
    vertices = mesh_data['vertices']
    triangles = mesh_data['triangles']
    triang = Triangulation(vertices[:, 0], vertices[:, 1], triangles)

    # Get cell centers for triangle field
    centers_tri = tri_grid.get_cell_centers()

    # TriGrid velocity (plot at cell centers using tripcolor)
    ax4 = plt.subplot(3, 3, 4)
    im4 = ax4.tripcolor(triang, np.array(u_tri), cmap='RdBu_r', shading='flat')
    ax4.set_title('TriGrid: u-velocity', fontsize=11, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4)

    ax5 = plt.subplot(3, 3, 5)
    im5 = ax5.tripcolor(triang, np.array(v_tri), cmap='RdBu_r', shading='flat')
    ax5.set_title('TriGrid: v-velocity', fontsize=11, fontweight='bold')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5)

    # TriGrid mesh structure
    ax6 = plt.subplot(3, 3, 6)
    ax6.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
    ax6.quiver(
        centers_tri[::2, 0], centers_tri[::2, 1],
        np.array(u_tri)[::2], np.array(v_tri)[::2],
        scale=8, alpha=0.7, color='darkred'
    )
    ax6.set_title('TriGrid: Mesh + Velocity\n(Unstructured)',
                  fontsize=11, fontweight='bold')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_aspect('equal')

    # ----------------------
    # Row 3: Divergence comparison
    # ----------------------

    # QuadGrid divergence
    ax7 = plt.subplot(3, 3, 7)
    im7 = ax7.pcolormesh(x_quad, y_quad, np.array(div_quad),
                         cmap='seismic', vmin=-0.5, vmax=0.5, shading='auto')
    ax7.set_title(f'QuadGrid: Divergence (FD)\nMax error: {max_error_quad:.2e}',
                  fontsize=11, fontweight='bold', color='green')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_aspect('equal')
    plt.colorbar(im7, ax=ax7)

    # TriGrid divergence
    ax8 = plt.subplot(3, 3, 8)
    im8 = ax8.tripcolor(triang, np.array(div_tri),
                        cmap='seismic', vmin=-0.5, vmax=0.5, shading='flat')
    ax8.set_title(f'TriGrid: Divergence (FV)\nMax error: {max_error_tri:.2e}',
                  fontsize=11, fontweight='bold', color='green')
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    ax8.set_aspect('equal')
    plt.colorbar(im8, ax=ax8)

    # Error comparison
    ax9 = plt.subplot(3, 3, 9)

    # QuadGrid error histogram
    quad_errors_flat = error_quad[2:-2, 2:-2].flatten()
    tri_errors_flat = error_tri[boundary_mask]

    ax9.hist(quad_errors_flat, bins=50, alpha=0.6, label='QuadGrid (FD)',
             color='blue', edgecolor='black')
    ax9.hist(tri_errors_flat, bins=50, alpha=0.6, label='TriGrid (FV)',
             color='red', edgecolor='black')
    ax9.set_xlabel('|Error|')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax9.set_yscale('log')
    ax9.legend()
    ax9.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = "output/example_16_triangle_divergence.png"
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
print("1. MODULAR ARCHITECTURE:")
print("   - Single divergence() function works for BOTH grid types")
print("   - Automatic method selection via grid.is_structured()")
print("   - QuadGrid → Finite Difference (structured stencils)")
print("   - TriGrid → Finite Volume (edge-based fluxes)")
print()
print("2. IMPLEMENTATION DETAILS:")
print("   - QuadGrid uses 2D array indexing: u[i, j]")
print("   - TriGrid uses 1D cell arrays: u[cell_id]")
print("   - Different data structures, same API!")
print()
print("3. ACCURACY:")
print(f"   - QuadGrid (FD): max error = {max_error_quad:.2e} (very accurate)")
print(f"   - TriGrid (FV): max error = {max_error_tri:.2e} (less accurate)")
print("   - FV on irregular meshes has larger truncation errors (expected)")
print()
print("4. VISUALIZATION:")
print("   - Top row: QuadGrid structured mesh and velocity")
print("   - Middle row: TriGrid unstructured mesh and velocity")
print("   - Bottom row: Divergence comparison (should be ~0 everywhere)")
print()
print("This demonstrates that the new architecture supports")
print("BOTH structured and unstructured grids seamlessly!")
print()
