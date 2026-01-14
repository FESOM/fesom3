"""Mesh generation utilities using xarray and uxarray."""

from typing import Tuple, Optional
import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False
    xr = None

try:
    import uxarray as ux
    UXARRAY_AVAILABLE = True
except ImportError:
    UXARRAY_AVAILABLE = False
    ux = None


def create_rectangular_mesh(
    nx: int,
    ny: int,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    use_xarray: bool = True,
) -> dict:
    """Create a rectangular/quadrilateral mesh.

    Args:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        domain: (x_min, x_max, y_min, y_max)
        use_xarray: Whether to use xarray for mesh structure

    Returns:
        Dictionary with mesh data:
            - 'vertices': vertex coordinates (n_vertices, 2)
            - 'cells': cell connectivity (n_cells, 4)
            - 'x_coords': x-coordinates of vertices
            - 'y_coords': y-coordinates of vertices
            - 'xarray_grid': xarray Dataset (if use_xarray=True)
    """
    x_min, x_max, y_min, y_max = domain

    # Create coordinate arrays
    x_coords = np.linspace(x_min, x_max, nx + 1)
    y_coords = np.linspace(y_min, y_max, ny + 1)

    # Create vertex mesh
    X, Y = np.meshgrid(x_coords, y_coords)
    vertices = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Create cell connectivity
    cells = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)
            cells.append([v0, v1, v2, v3])
    cells = np.array(cells)

    mesh_data = {
        'vertices': vertices,
        'cells': cells,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }

    # Create xarray representation if requested
    if use_xarray and XARRAY_AVAILABLE:
        # Create cell-centered coordinates
        x_center = 0.5 * (x_coords[:-1] + x_coords[1:])
        y_center = 0.5 * (y_coords[:-1] + y_coords[1:])

        # Create xarray Dataset
        ds = xr.Dataset(
            coords={
                'x': (['x'], x_center),
                'y': (['y'], y_center),
                'x_vert': (['x_vert'], x_coords),
                'y_vert': (['y_vert'], y_coords),
            },
            attrs={
                'nx': nx,
                'ny': ny,
                'domain': domain,
                'grid_type': 'rectangular',
            }
        )
        mesh_data['xarray_grid'] = ds

    return mesh_data


def create_triangular_mesh(
    vertices: np.ndarray,
    method: str = 'delaunay',
    **kwargs
) -> dict:
    """Create a triangular mesh from vertices.

    Args:
        vertices: Vertex coordinates (n_vertices, 2)
        method: Triangulation method ('delaunay', 'structured', etc.)
        **kwargs: Additional arguments for triangulation

    Returns:
        Dictionary with mesh data:
            - 'vertices': vertex coordinates
            - 'triangles': triangle connectivity (n_triangles, 3)
            - 'uxarray_grid': UxDataset (if uxarray available)
    """
    if method == 'delaunay':
        from scipy.spatial import Delaunay
        tri = Delaunay(vertices)
        triangles = tri.simplices
    else:
        raise ValueError(f"Unknown triangulation method: {method}")

    mesh_data = {
        'vertices': vertices,
        'triangles': triangles,
    }

    # Create uxarray representation if available
    if UXARRAY_AVAILABLE:
        # UXarray uses specific format for unstructured grids
        # This is a placeholder - full implementation would properly format for uxarray
        mesh_data['method'] = method

    return mesh_data


def create_structured_triangular_mesh(
    nx: int,
    ny: int,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
) -> dict:
    """Create a structured triangular mesh by splitting quads.

    Each rectangular cell is split into two triangles.

    Args:
        nx: Number of quads in x-direction
        ny: Number of quads in y-direction
        domain: (x_min, x_max, y_min, y_max)

    Returns:
        Dictionary with mesh data
    """
    x_min, x_max, y_min, y_max = domain

    # Create vertices
    x_coords = np.linspace(x_min, x_max, nx + 1)
    y_coords = np.linspace(y_min, y_max, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords)
    vertices = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Create triangles by splitting each quad
    triangles = []
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)

            # Split into two triangles
            triangles.append([v0, v1, v2])
            triangles.append([v0, v2, v3])

    triangles = np.array(triangles)

    return {
        'vertices': vertices,
        'triangles': triangles,
        'x_coords': x_coords,
        'y_coords': y_coords,
    }


def create_mixed_mesh(
    nx: int,
    ny: int,
    domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    tri_fraction: float = 0.3,
    seed: Optional[int] = None,
) -> dict:
    """Create a mixed quad/triangle mesh.

    Args:
        nx: Number of cells in x-direction
        ny: Number of cells in y-direction
        domain: (x_min, x_max, y_min, y_max)
        tri_fraction: Fraction of quads to split into triangles
        seed: Random seed for reproducibility

    Returns:
        Dictionary with mesh data:
            - 'vertices': vertex coordinates
            - 'cells': list of cells (variable connectivity)
            - 'cell_types': array of cell types (3=tri, 4=quad)
    """
    if seed is not None:
        np.random.seed(seed)

    x_min, x_max, y_min, y_max = domain

    # Create vertices
    x_coords = np.linspace(x_min, x_max, nx + 1)
    y_coords = np.linspace(y_min, y_max, ny + 1)
    X, Y = np.meshgrid(x_coords, y_coords)
    vertices = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Determine which quads to split
    n_quads = nx * ny
    n_to_split = int(n_quads * tri_fraction)
    split_indices = set(np.random.choice(n_quads, n_to_split, replace=False))

    # Create cells
    cells = []
    cell_types = []

    quad_idx = 0
    for j in range(ny):
        for i in range(nx):
            v0 = j * (nx + 1) + i
            v1 = v0 + 1
            v2 = v0 + (nx + 1) + 1
            v3 = v0 + (nx + 1)

            if quad_idx in split_indices:
                # Split into triangles
                cells.append(np.array([v0, v1, v2]))
                cells.append(np.array([v0, v2, v3]))
                cell_types.extend([3, 3])
            else:
                # Keep as quad
                cells.append(np.array([v0, v1, v2, v3]))
                cell_types.append(4)

            quad_idx += 1

    return {
        'vertices': vertices,
        'cells': cells,
        'cell_types': np.array(cell_types),
        'x_coords': x_coords,
        'y_coords': y_coords,
    }
