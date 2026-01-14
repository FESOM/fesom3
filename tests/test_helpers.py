"""
Helper functions for testing operators on different grid types.
"""

import numpy as np
import jax.numpy as jnp

from fesomx import QuadGrid, TriGrid, StaggerType


def create_structured_triangular_mesh(nx, ny, domain=(0.0, 1.0, 0.0, 1.0)):
    """
    Create structured triangular mesh by splitting quads.

    Args:
        nx: Number of cells in x (before splitting)
        ny: Number of cells in y (before splitting)
        domain: (x_min, x_max, y_min, y_max)

    Returns:
        dict with 'vertices' and 'triangles'
    """
    x_min, x_max, y_min, y_max = domain

    # Create vertices
    x = np.linspace(x_min, x_max, nx + 1)
    y = np.linspace(y_min, y_max, ny + 1)
    X, Y = np.meshgrid(x, y)
    vertices = np.column_stack([X.ravel(), Y.ravel()])

    # Create triangles by splitting each quad
    triangles = []
    for j in range(ny):
        for i in range(nx):
            # Bottom-left vertex index
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)

            # Split quad into two triangles
            triangles.append([v0, v1, v2])  # Lower triangle
            triangles.append([v0, v2, v3])  # Upper triangle

    return {
        'vertices': vertices,
        'triangles': np.array(triangles)
    }


def create_divergence_free_field_unstructured(grid, mode='rotation'):
    """
    Create divergence-free velocity field on unstructured grid.

    Args:
        grid: TriGrid or MixedGrid
        mode: 'rotation' or 'sine'

    Returns:
        u, v: Velocity components at cell centers (n_cells,)
    """
    centers = grid.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]

    if mode == 'rotation':
        # Solid body rotation around center
        x_c, y_c = 0.5, 0.5
        u = -(y - y_c)
        v = (x - x_c)

    elif mode == 'sine':
        # Sine-based divergence-free field
        u = np.sin(np.pi * x) * np.cos(np.pi * y)
        v = -np.cos(np.pi * x) * np.sin(np.pi * y)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return jnp.array(u), jnp.array(v)


def create_gaussian_field_unstructured(grid, x0=0.5, y0=0.5, sigma=0.15):
    """
    Create Gaussian scalar field on unstructured grid.

    Args:
        grid: TriGrid or MixedGrid
        x0, y0: Center coordinates
        sigma: Width parameter

    Returns:
        phi: Scalar field at cell centers (n_cells,)
    """
    centers = grid.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]

    r2 = (x - x0)**2 + (y - y0)**2
    phi = np.exp(-r2 / (2 * sigma**2))

    return jnp.array(phi)


def analytical_gradient_gaussian_unstructured(grid, x0=0.5, y0=0.5, sigma=0.15):
    """
    Analytical gradient of Gaussian field on unstructured grid.

    Returns:
        dphidx, dphidy: Gradients at cell centers (n_cells,)
    """
    centers = grid.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]

    phi = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    dphidx = -phi * (x - x0) / sigma**2
    dphidy = -phi * (y - y0) / sigma**2

    return jnp.array(dphidx), jnp.array(dphidy)


def analytical_laplacian_gaussian_unstructured(grid, x0=0.5, y0=0.5, sigma=0.15):
    """
    Analytical Laplacian of Gaussian field on unstructured grid.

    Returns:
        lapl: Laplacian at cell centers (n_cells,)
    """
    centers = grid.get_cell_centers()
    x = centers[:, 0]
    y = centers[:, 1]

    r2 = (x - x0)**2 + (y - y0)**2
    phi = np.exp(-r2 / (2 * sigma**2))
    lapl = phi * (r2 / sigma**4 - 2 / sigma**2)

    return jnp.array(lapl)


def get_interior_mask(grid):
    """
    Get mask for interior cells (exclude boundary).

    Args:
        grid: Grid object

    Returns:
        mask: Boolean array (n_cells,) or slice for structured grids
    """
    if grid.is_structured():
        # For structured grids, return slice
        return (slice(2, -2), slice(2, -2))
    else:
        # For unstructured grids, exclude boundary cells
        boundary = grid.get_boundary_cells()
        mask = np.ones(grid.n_cells, dtype=bool)
        mask[boundary] = False
        return mask


def compute_error(numerical, analytical, grid, margin=2):
    """
    Compute error between numerical and analytical solutions.

    Args:
        numerical: Numerical result
        analytical: Analytical solution
        grid: Grid object
        margin: Number of cells to exclude at boundaries

    Returns:
        max_error: Maximum error
    """
    if grid.is_structured():
        # Structured grid: use array slicing
        error = np.abs(np.array(numerical[margin:-margin, margin:-margin]) -
                      np.array(analytical[margin:-margin, margin:-margin]))
    else:
        # Unstructured grid: exclude boundary cells
        mask = get_interior_mask(grid)
        error = np.abs(np.array(numerical)[mask] - np.array(analytical)[mask])

    return error.max()
