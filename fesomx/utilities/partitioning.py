"""Mesh partitioning utilities for unstructured grids.

Supports:
- Graph-based partitioning using METIS
- Simple geometric partitioning (fallback)
- Neighbor/ghost cell identification
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

# Try to import METIS
try:
    import metis
    METIS_AVAILABLE = True
except ImportError:
    METIS_AVAILABLE = False
    metis = None


def partition_graph_metis(
    adjacency: Dict[int, List[int]],
    n_partitions: int,
    **kwargs
) -> np.ndarray:
    """Partition a graph using METIS.

    Args:
        adjacency: Dict mapping cell_id -> [neighbor_cell_ids]
        n_partitions: Number of partitions to create
        **kwargs: Additional METIS options

    Returns:
        Array of partition IDs for each cell

    Raises:
        ImportError: If METIS is not available
    """
    if not METIS_AVAILABLE:
        raise ImportError(
            "METIS not available. Install with: pip install metis\n"
            "Or use geometric partitioning as fallback."
        )

    n_cells = len(adjacency)

    # Convert adjacency dict to METIS format
    # METIS expects: xadj and adjncy arrays
    xadj = [0]
    adjncy = []

    for cell_id in range(n_cells):
        neighbors = adjacency.get(cell_id, [])
        adjncy.extend(neighbors)
        xadj.append(len(adjncy))

    # Call METIS partitioner
    try:
        _, parts = metis.part_graph(
            xadj, adjncy, nparts=n_partitions, **kwargs
        )
        return np.array(parts)
    except Exception as e:
        raise RuntimeError(f"METIS partitioning failed: {e}")


def partition_geometric(
    cell_centers: np.ndarray,
    n_partitions: int,
    method: str = 'rcb'
) -> np.ndarray:
    """Partition mesh geometrically (fallback when METIS unavailable).

    Args:
        cell_centers: Array of shape (n_cells, 2) with (x, y) coordinates
        n_partitions: Number of partitions
        method: 'rcb' (recursive coordinate bisection) or 'simple' (grid)

    Returns:
        Array of partition IDs for each cell
    """
    n_cells = len(cell_centers)

    if method == 'simple':
        # Simple grid-based partitioning
        # Find bounding box
        x_min, y_min = cell_centers.min(axis=0)
        x_max, y_max = cell_centers.max(axis=0)

        # Determine grid layout
        n_x = int(np.sqrt(n_partitions))
        n_y = n_partitions // n_x

        # Assign cells to partitions based on position
        x_normalized = (cell_centers[:, 0] - x_min) / (x_max - x_min)
        y_normalized = (cell_centers[:, 1] - y_min) / (y_max - y_min)

        x_part = np.floor(x_normalized * n_x).astype(int)
        y_part = np.floor(y_normalized * n_y).astype(int)

        # Clamp to valid range
        x_part = np.clip(x_part, 0, n_x - 1)
        y_part = np.clip(y_part, 0, n_y - 1)

        parts = y_part * n_x + x_part

    elif method == 'rcb':
        # Recursive Coordinate Bisection
        parts = np.zeros(n_cells, dtype=int)
        _rcb_partition(cell_centers, np.arange(n_cells), parts, 0, n_partitions, 0)

    else:
        raise ValueError(f"Unknown geometric partitioning method: {method}")

    return parts


def _rcb_partition(
    centers: np.ndarray,
    indices: np.ndarray,
    parts: np.ndarray,
    part_offset: int,
    n_parts: int,
    depth: int
):
    """Recursive coordinate bisection helper."""
    if n_parts == 1:
        parts[indices] = part_offset
        return

    if len(indices) == 0:
        return

    # Choose coordinate to split (alternate x, y)
    coord = depth % 2

    # Find median
    coords_to_split = centers[indices, coord]
    median = np.median(coords_to_split)

    # Split into two groups
    left_mask = coords_to_split <= median
    left_indices = indices[left_mask]
    right_indices = indices[~left_mask]

    # Recurse
    n_left = n_parts // 2
    n_right = n_parts - n_left

    _rcb_partition(centers, left_indices, parts, part_offset, n_left, depth + 1)
    _rcb_partition(centers, right_indices, parts, part_offset + n_left, n_right, depth + 1)


def build_ghost_cell_map(
    cell_neighbors: List[List[int]],
    partition: np.ndarray,
    halo_layers: int = 1
) -> Dict[int, Dict[int, List[int]]]:
    """Build ghost cell mapping for unstructured halo exchange.

    Args:
        cell_neighbors: List where cell_neighbors[i] = [neighbor indices]
        partition: Array of partition IDs for each cell
        halo_layers: Number of neighbor layers for ghost cells

    Returns:
        Dict: {partition_id: {neighbor_partition_id: [cell_indices_needed]}}

    Example:
        Partition 0 needs cells [12, 15, 20] from Partition 1
        → ghost_map[0][1] = [12, 15, 20]
    """
    n_cells = len(cell_neighbors)
    n_partitions = int(partition.max()) + 1

    # Initialize ghost map
    ghost_map = {p: {} for p in range(n_partitions)}

    # For each partition, find boundary cells and their neighbors
    for part_id in range(n_partitions):
        # Cells owned by this partition
        owned_cells = np.where(partition == part_id)[0]

        # Track ghost cells needed from each neighbor partition
        ghost_cells_needed = {}

        for cell_id in owned_cells:
            neighbors = cell_neighbors[cell_id]

            for neighbor_id in neighbors:
                neighbor_part = partition[neighbor_id]

                if neighbor_part != part_id:
                    # This neighbor is in a different partition - need as ghost
                    if neighbor_part not in ghost_cells_needed:
                        ghost_cells_needed[neighbor_part] = set()
                    ghost_cells_needed[neighbor_part].add(neighbor_id)

        # Convert sets to lists
        for neighbor_part, cells in ghost_cells_needed.items():
            ghost_map[part_id][neighbor_part] = sorted(list(cells))

    return ghost_map


def partition_triangular_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    n_partitions: int,
    method: str = 'metis'
) -> Tuple[np.ndarray, Dict]:
    """Partition a triangular mesh.

    Args:
        vertices: Array of shape (n_vertices, 2)
        triangles: Array of shape (n_triangles, 3) with vertex indices
        n_partitions: Number of partitions
        method: 'metis' or 'geometric'

    Returns:
        Tuple of:
        - partition: Array of partition IDs for each triangle
        - info: Dict with additional partitioning information
    """
    n_triangles = len(triangles)

    # Build cell adjacency graph
    adjacency = build_triangle_adjacency(triangles)

    # Partition the graph
    if method == 'metis' and METIS_AVAILABLE:
        try:
            partition = partition_graph_metis(adjacency, n_partitions)
        except Exception as e:
            print(f"Warning: METIS failed ({e}), falling back to geometric")
            method = 'geometric'

    if method == 'geometric' or not METIS_AVAILABLE:
        # Calculate cell centers
        cell_centers = vertices[triangles].mean(axis=1)
        partition = partition_geometric(cell_centers, n_partitions, method='rcb')

    # Build neighbor list (needed for ghost cells)
    cell_neighbors = adjacency_to_neighbor_list(adjacency, n_triangles)

    # Build ghost cell map
    ghost_map = build_ghost_cell_map(cell_neighbors, partition, halo_layers=1)

    # Compute partition statistics
    info = {
        'n_partitions': n_partitions,
        'method': method,
        'cells_per_partition': {},
        'boundary_cells': {},
        'ghost_cells_needed': {},
    }

    for part_id in range(n_partitions):
        owned = np.sum(partition == part_id)
        info['cells_per_partition'][part_id] = owned

        # Count boundary cells (cells with neighbors in other partitions)
        boundary_count = 0
        for cell_id in np.where(partition == part_id)[0]:
            for neighbor_id in cell_neighbors[cell_id]:
                if partition[neighbor_id] != part_id:
                    boundary_count += 1
                    break
        info['boundary_cells'][part_id] = boundary_count

        # Count ghost cells needed
        total_ghosts = sum(len(cells) for cells in ghost_map[part_id].values())
        info['ghost_cells_needed'][part_id] = total_ghosts

    return partition, info


def build_triangle_adjacency(triangles: np.ndarray) -> Dict[int, List[int]]:
    """Build adjacency graph for triangular mesh.

    Args:
        triangles: Array of shape (n_triangles, 3)

    Returns:
        Dict mapping triangle_id -> [neighbor_triangle_ids]
    """
    n_triangles = len(triangles)

    # Build edge-to-triangles mapping
    edge_to_triangles = {}

    for tri_id, tri in enumerate(triangles):
        edges = [
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ]

        for edge in edges:
            if edge not in edge_to_triangles:
                edge_to_triangles[edge] = []
            edge_to_triangles[edge].append(tri_id)

    # Build adjacency from edge sharing
    adjacency = {i: [] for i in range(n_triangles)}

    for edge, tris in edge_to_triangles.items():
        if len(tris) == 2:
            # Interior edge: two triangles share it
            tri0, tri1 = tris
            adjacency[tri0].append(tri1)
            adjacency[tri1].append(tri0)

    return adjacency


def adjacency_to_neighbor_list(
    adjacency: Dict[int, List[int]],
    n_cells: int
) -> List[List[int]]:
    """Convert adjacency dict to neighbor list.

    Args:
        adjacency: Dict mapping cell_id -> [neighbor_ids]
        n_cells: Total number of cells

    Returns:
        List where neighbors[i] = [neighbor indices of cell i]
    """
    neighbors = [[] for _ in range(n_cells)]

    for cell_id, neighbor_ids in adjacency.items():
        neighbors[cell_id] = neighbor_ids

    return neighbors


def print_partition_summary(partition: np.ndarray, info: Dict):
    """Print summary of partitioning.

    Args:
        partition: Partition array
        info: Info dict from partition_triangular_mesh
    """
    print("Partitioning Summary:")
    print("=" * 60)
    print(f"Method: {info['method']}")
    print(f"Number of partitions: {info['n_partitions']}")
    print()

    print(f"{'Part':<6} {'Cells':<10} {'Boundary':<12} {'Ghosts':<10}")
    print("-" * 60)

    for part_id in range(info['n_partitions']):
        n_cells = info['cells_per_partition'][part_id]
        n_boundary = info['boundary_cells'][part_id]
        n_ghosts = info['ghost_cells_needed'][part_id]

        print(f"{part_id:<6} {n_cells:<10} {n_boundary:<12} {n_ghosts:<10}")

    print()

    # Load balance
    cells_per_part = list(info['cells_per_partition'].values())
    if cells_per_part:
        min_cells = min(cells_per_part)
        max_cells = max(cells_per_part)
        avg_cells = np.mean(cells_per_part)
        imbalance = (max_cells - min_cells) / avg_cells * 100

        print(f"Load Balance:")
        print(f"  Min cells: {min_cells}")
        print(f"  Max cells: {max_cells}")
        print(f"  Avg cells: {avg_cells:.1f}")
        print(f"  Imbalance: {imbalance:.1f}%")
