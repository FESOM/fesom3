"""Visualization utilities for grids and fields."""

from typing import Optional, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None


def plot_grid(
    vertices: np.ndarray,
    cells: np.ndarray | list,
    ax: Optional[Any] = None,
    show_vertices: bool = True,
    show_cells: bool = True,
    title: str = "Grid",
    **kwargs
) -> Any:
    """Plot grid structure.

    Args:
        vertices: Vertex coordinates (n_vertices, 2)
        cells: Cell connectivity (n_cells, n_verts_per_cell)
        ax: Matplotlib axis (creates new if None)
        show_vertices: Whether to plot vertex points
        show_cells: Whether to plot cell edges
        title: Plot title
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Plot cells
    if show_cells:
        if isinstance(cells, list):
            # Mixed connectivity - plot each cell
            for cell in cells:
                cell_verts = vertices[cell]
                # Close the polygon
                cell_verts = np.vstack([cell_verts, cell_verts[0]])
                ax.plot(cell_verts[:, 0], cell_verts[:, 1],
                       'k-', linewidth=0.5, alpha=0.7)
        else:
            # Uniform connectivity
            for cell in cells:
                cell_verts = vertices[cell]
                cell_verts = np.vstack([cell_verts, cell_verts[0]])
                ax.plot(cell_verts[:, 0], cell_verts[:, 1],
                       'k-', linewidth=0.5, alpha=0.7)

    # Plot vertices
    if show_vertices:
        ax.plot(vertices[:, 0], vertices[:, 1], 'ro', markersize=2, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    return ax


def plot_field(
    vertices: np.ndarray,
    cells: np.ndarray | list,
    field_data: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Field",
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    **kwargs
) -> Any:
    """Plot field on grid.

    Args:
        vertices: Vertex coordinates
        cells: Cell connectivity
        field_data: Field values (per cell or per vertex)
        ax: Matplotlib axis
        title: Plot title
        cmap: Colormap name
        vmin: Minimum value for colorbar
        vmax: Maximum value for colorbar
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    # Determine if field is cell-centered or vertex-centered
    if isinstance(cells, list):
        n_cells = len(cells)
    else:
        n_cells = len(cells)

    n_vertices = len(vertices)

    if len(field_data) == n_cells:
        # Cell-centered data
        _plot_cell_centered_field(vertices, cells, field_data, ax, cmap, vmin, vmax)
    elif len(field_data) == n_vertices:
        # Vertex-centered data
        _plot_vertex_centered_field(vertices, cells, field_data, ax, cmap, vmin, vmax)
    else:
        raise ValueError(
            f"Field data size {len(field_data)} doesn't match "
            f"n_cells={n_cells} or n_vertices={n_vertices}"
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_aspect('equal')

    return ax


def _plot_cell_centered_field(vertices, cells, field_data, ax, cmap, vmin, vmax):
    """Plot cell-centered field data."""
    if vmin is None:
        vmin = field_data.min()
    if vmax is None:
        vmax = field_data.max()

    patches = []

    if isinstance(cells, list):
        # Mixed cells
        for cell in cells:
            polygon = Polygon(vertices[cell], closed=True)
            patches.append(polygon)
    else:
        # Uniform cells
        for cell in cells:
            polygon = Polygon(vertices[cell], closed=True)
            patches.append(polygon)

    collection = PatchCollection(patches, cmap=cmap, alpha=0.8)
    collection.set_array(field_data)
    collection.set_clim(vmin, vmax)

    ax.add_collection(collection)
    plt.colorbar(collection, ax=ax, label='Field Value')

    # Set plot limits
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())


def _plot_vertex_centered_field(vertices, cells, field_data, ax, cmap, vmin, vmax):
    """Plot vertex-centered field data using triangulation."""
    if vmin is None:
        vmin = field_data.min()
    if vmax is None:
        vmax = field_data.max()

    # Create triangulation (works for triangular grids)
    # For quad grids, this is approximate
    triang = mtri.Triangulation(vertices[:, 0], vertices[:, 1])

    # Plot using tripcolor
    tcf = ax.tripcolor(triang, field_data, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(tcf, ax=ax, label='Field Value')


def plot_halo_regions(
    vertices: np.ndarray,
    cells: np.ndarray,
    halo_width: int,
    shape: tuple,
    ax: Optional[Any] = None,
    title: str = "Halo Regions",
) -> Any:
    """Visualize halo regions on a structured grid.

    Args:
        vertices: Vertex coordinates
        cells: Cell connectivity
        halo_width: Width of halo region
        shape: Grid shape (ny, nx)
        ax: Matplotlib axis
        title: Plot title

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ny, nx = shape

    # Create field indicating halo regions
    # 0 = interior, 1 = halo
    field = np.zeros(ny * nx)

    # Mark halo cells
    for j in range(ny):
        for i in range(nx):
            cell_id = j * nx + i
            if i < halo_width or i >= nx - halo_width:
                field[cell_id] = 1
            if j < halo_width or j >= ny - halo_width:
                field[cell_id] = 1

    # Plot
    plot_field(vertices, cells, field, ax=ax, title=title, cmap='RdYlGn_r')

    return ax


def plot_partition_boundaries(
    vertices: np.ndarray,
    cells: np.ndarray | list,
    partition: np.ndarray,
    ax: Optional[Any] = None,
    title: str = "Mesh Partitioning",
    show_boundaries: bool = True,
    **kwargs
) -> Any:
    """Plot mesh partitioning with colored regions.

    Args:
        vertices: Vertex coordinates
        cells: Cell connectivity
        partition: Partition ID for each cell
        ax: Matplotlib axis
        title: Plot title
        show_boundaries: Whether to highlight partition boundaries
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    n_partitions = int(partition.max()) + 1

    # Plot each partition with different color
    import matplotlib.cm as cm
    colors = cm.get_cmap('tab10', n_partitions)

    # Plot cells colored by partition
    plot_field(
        vertices, cells, partition.astype(float),
        ax=ax, title=title, cmap='tab10',
        vmin=0, vmax=n_partitions-1
    )

    # Highlight partition boundaries
    if show_boundaries:
        # For each cell, check if neighbors are in different partitions
        if isinstance(cells, list):
            # Mixed grid - handle variable connectivity
            pass  # TODO: implement for mixed grids
        else:
            # Uniform connectivity - find boundary edges
            n_cells = len(cells)

            # Build cell-to-cell adjacency
            edges_to_cells = {}
            for cell_id, cell in enumerate(cells):
                n_verts = len(cell)
                for i in range(n_verts):
                    edge = tuple(sorted([cell[i], cell[(i+1) % n_verts]]))
                    if edge not in edges_to_cells:
                        edges_to_cells[edge] = []
                    edges_to_cells[edge].append(cell_id)

            # Find edges on partition boundaries
            boundary_edges = []
            for edge, cell_ids in edges_to_cells.items():
                if len(cell_ids) == 2:
                    c0, c1 = cell_ids
                    if partition[c0] != partition[c1]:
                        # This edge is on a partition boundary
                        boundary_edges.append(edge)

            # Draw boundary edges
            for edge in boundary_edges:
                v0, v1 = edge
                ax.plot(
                    [vertices[v0, 0], vertices[v1, 0]],
                    [vertices[v0, 1], vertices[v1, 1]],
                    'k-', linewidth=2, alpha=0.8
                )

    return ax


def plot_ghost_cells(
    vertices: np.ndarray,
    cells: np.ndarray | list,
    partition: np.ndarray,
    device_id: int,
    ghost_cell_map: dict,
    ax: Optional[Any] = None,
    title: str = None,
    **kwargs
) -> Any:
    """Plot ghost cells for a specific device/partition.

    Args:
        vertices: Vertex coordinates
        cells: Cell connectivity
        partition: Partition ID for each cell
        device_id: Device to show ghost cells for
        ghost_cell_map: Ghost cell mapping from create_partition_info
        ax: Matplotlib axis
        title: Plot title
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))

    if title is None:
        title = f"Ghost Cells for Device {device_id}"

    n_cells = len(partition)

    # Create field showing owned vs ghost cells
    # 0 = not this device, 1 = owned by this device, 2 = ghost for this device
    cell_type = np.zeros(n_cells)

    # Mark owned cells
    owned_cells = np.where(partition == device_id)[0]
    cell_type[owned_cells] = 1

    # Mark ghost cells
    if device_id in ghost_cell_map:
        for neighbor_id, ghost_cells in ghost_cell_map[device_id].items():
            for cell_id in ghost_cells:
                cell_type[cell_id] = 2 + (neighbor_id % 8) * 0.1  # Slightly different for each neighbor

    # Plot
    plot_field(
        vertices, cells, cell_type,
        ax=ax, title=title, cmap='RdYlGn',
        vmin=0, vmax=3
    )

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Other devices'),
        Patch(facecolor='yellow', label=f'Owned by Device {device_id}'),
        Patch(facecolor='green', label='Ghost cells needed'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    return ax


def plot_communication_graph(
    partition_info: dict,
    ax: Optional[Any] = None,
    title: str = "Device Communication Graph",
    layout: str = 'spring',
    **kwargs
) -> Any:
    """Plot graph showing which devices communicate.

    Args:
        partition_info: Partition info dict from create_partition_info
        ax: Matplotlib axis
        title: Plot title
        layout: Graph layout ('spring', 'circular', 'grid')
        **kwargs: Additional plotting arguments

    Returns:
        Matplotlib axis
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    try:
        import networkx as nx
    except ImportError:
        print("NetworkX required for communication graph. Install with: pip install networkx")
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Build communication graph
    G = nx.Graph()

    # Determine number of devices
    if 'mesh_shape' in partition_info:
        # Structured
        n_devices = np.prod(partition_info['mesh_shape'])
        neighbor_map = partition_info['neighbor_map']

        # Add edges from neighbor map
        for device_id, neighbors in neighbor_map.items():
            for direction, neighbor_id in neighbors.items():
                G.add_edge(device_id, neighbor_id)

    elif 'ghost_cell_map' in partition_info:
        # Unstructured
        ghost_map = partition_info['ghost_cell_map']
        n_devices = partition_info['n_devices']

        # Add edges from ghost cell map
        for device_id, neighbors in ghost_map.items():
            for neighbor_id in neighbors.keys():
                G.add_edge(device_id, neighbor_id)

    # Choose layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'grid' and 'mesh_shape' in partition_info:
        # Grid layout for structured meshes
        ny, nx = partition_info['mesh_shape']
        pos = {}
        for device_id in range(n_devices):
            j = device_id // nx
            i = device_id % nx
            pos[device_id] = (i, ny - 1 - j)  # Flip y for display
    else:
        pos = nx.spring_layout(G, seed=42)

    # Draw graph
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, ax=ax)

    # Add edge labels showing communication volume (if available)
    if 'ghost_cell_map' in partition_info:
        ghost_map = partition_info['ghost_cell_map']
        edge_labels = {}
        for device_id, neighbors in ghost_map.items():
            for neighbor_id, ghost_cells in neighbors.items():
                edge = (device_id, neighbor_id) if device_id < neighbor_id else (neighbor_id, device_id)
                edge_labels[edge] = len(ghost_cells)

        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')

    return ax


def save_plot(filename: str, dpi: int = 300, **kwargs):
    """Save current matplotlib figure.

    Args:
        filename: Output filename
        dpi: Resolution
        **kwargs: Additional savefig arguments
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib required for plotting")

    plt.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)
    print(f"Plot saved to {filename}")
