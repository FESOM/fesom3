"""Triangular (unstructured) grid implementation."""

from typing import Optional, Tuple
import numpy as np

from .base_grid import BaseGrid
from .backend import Backend
from .staggering import StaggerType


class TriGrid(BaseGrid):
    """Unstructured triangular grid.

    Represents a mesh composed entirely of triangular cells.
    """

    def __init__(
        self,
        name: str,
        stagger_type: StaggerType = StaggerType.A,
        backend: Optional[Backend] = None,
        periodic: Tuple[bool, bool] = (False, False),
    ):
        """Initialize triangular grid.

        Args:
            name: Grid identifier
            stagger_type: Type of grid staggering
            backend: Parallel backend
            periodic: Periodic boundaries (note: limited support for unstructured)
        """
        super().__init__(name, stagger_type, backend, periodic)

        # Triangle-specific attributes
        self.cell_neighbors = None  # Neighbor cells for each cell

    def create_mesh(
        self,
        vertices: np.ndarray,
        triangles: np.ndarray,
    ) -> None:
        """Create triangular mesh from vertices and connectivity.

        Args:
            vertices: Array of shape (n_vertices, 2) with (x, y) coordinates
            triangles: Array of shape (n_triangles, 3) with vertex indices
        """
        self.vertices = vertices
        self.cells = triangles

        # Build edge connectivity
        self._build_edges()

        # Compute neighbor connectivity
        self._build_cell_neighbors()

    def _build_edges(self) -> None:
        """Build edge connectivity from triangle definitions."""
        edges_set = set()

        for tri in self.cells:
            # Each triangle has 3 edges
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]])),
            ]
            for edge in edges:
                edges_set.add(edge)

        self.edges = np.array(list(edges_set))

    def _build_cell_neighbors(self) -> None:
        """Build cell neighbor connectivity.

        Two cells are neighbors if they share an edge.
        For periodic boundaries, cells on opposite boundary edges are connected.
        """
        # Create edge-to-cell mapping
        edge_to_cells = {}

        for cell_id, tri in enumerate(self.cells):
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]])),
            ]
            for edge in edges:
                if edge not in edge_to_cells:
                    edge_to_cells[edge] = []
                edge_to_cells[edge].append(cell_id)

        # Build neighbor list for each cell
        self.cell_neighbors = [[] for _ in range(self.n_cells)]

        # Track boundary edges for periodic matching
        boundary_edges = {}  # edge -> cell_id

        for edge, cells in edge_to_cells.items():
            if len(cells) == 2:
                # Interior edge: two cells share it
                c0, c1 = cells
                self.cell_neighbors[c0].append(c1)
                self.cell_neighbors[c1].append(c0)
            elif len(cells) == 1:
                # Boundary edge
                boundary_edges[edge] = cells[0]

        # Handle periodic boundaries if enabled
        if any(self.periodic) and len(boundary_edges) > 0:
            self._connect_periodic_boundaries(boundary_edges, edge_to_cells)

    def _connect_periodic_boundaries(
        self,
        boundary_edges: dict,
        edge_to_cells: dict
    ) -> None:
        """Connect cells across periodic boundaries.

        For structured triangular grids on rectangular domains:
        - Match edges on left/right (periodic_x)
        - Match edges on bottom/top (periodic_y)

        Args:
            boundary_edges: {edge: cell_id} for boundary edges
            edge_to_cells: Full edge-to-cell mapping
        """
        if self.vertices is None:
            return

        # Get domain bounds
        x_min, x_max = self.vertices[:, 0].min(), self.vertices[:, 0].max()
        y_min, y_max = self.vertices[:, 1].min(), self.vertices[:, 1].max()
        tol = 1e-10 * max(x_max - x_min, y_max - y_min)

        # Classify boundary edges by location
        left_edges = []    # x ≈ x_min
        right_edges = []   # x ≈ x_max
        bottom_edges = []  # y ≈ y_min
        top_edges = []     # y ≈ y_max

        for edge, cell_id in boundary_edges.items():
            v0, v1 = edge
            x0, y0 = self.vertices[v0]
            x1, y1 = self.vertices[v1]

            edge_x_min = min(x0, x1)
            edge_x_max = max(x0, x1)
            edge_y_min = min(y0, y1)
            edge_y_max = max(y0, y1)

            # Classify by position (use midpoint for robustness)
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            if abs(mid_x - x_min) < tol:
                left_edges.append((edge, cell_id, mid_y))
            elif abs(mid_x - x_max) < tol:
                right_edges.append((edge, cell_id, mid_y))
            elif abs(mid_y - y_min) < tol:
                bottom_edges.append((edge, cell_id, mid_x))
            elif abs(mid_y - y_max) < tol:
                top_edges.append((edge, cell_id, mid_x))

        # Connect left-right (periodic_x)
        if self.periodic[0]:
            self._match_periodic_edges(left_edges, right_edges, coord_idx=2)

        # Connect bottom-top (periodic_y)
        if self.periodic[1]:
            self._match_periodic_edges(bottom_edges, top_edges, coord_idx=2)

    def _match_periodic_edges(
        self,
        edges_a: list,
        edges_b: list,
        coord_idx: int
    ) -> None:
        """Match edges on opposite boundaries and connect their cells.

        Args:
            edges_a: List of (edge, cell_id, coord) on one boundary
            edges_b: List of (edge, cell_id, coord) on opposite boundary
            coord_idx: Index of matching coordinate in tuples (2 for mid_y or mid_x)
        """
        if len(edges_a) == 0 or len(edges_b) == 0:
            return

        # Sort by coordinate
        edges_a_sorted = sorted(edges_a, key=lambda x: x[coord_idx])
        edges_b_sorted = sorted(edges_b, key=lambda x: x[coord_idx])

        # Match edges with closest coordinates
        tol = 1e-10 * max(
            edges_a_sorted[-1][coord_idx] - edges_a_sorted[0][coord_idx],
            1e-6
        )

        for edge_a, cell_a, coord_a in edges_a_sorted:
            # Find matching edge in edges_b
            for edge_b, cell_b, coord_b in edges_b_sorted:
                if abs(coord_a - coord_b) < tol:
                    # Connect cells
                    if cell_b not in self.cell_neighbors[cell_a]:
                        self.cell_neighbors[cell_a].append(cell_b)
                    if cell_a not in self.cell_neighbors[cell_b]:
                        self.cell_neighbors[cell_b].append(cell_a)
                    break

    def get_cell_centers(self) -> np.ndarray:
        """Get coordinates of cell centers (centroids).

        Returns:
            Array of shape (n_cells, 2) with (x, y) coordinates
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        # Centroid is average of three vertices
        cell_vertices = self.vertices[self.cells]
        centers = cell_vertices.mean(axis=1)
        return centers

    def get_cell_volumes(self) -> np.ndarray:
        """Get cell areas.

        Returns:
            Array of shape (n_cells,) with triangle areas
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        # Area of triangle using cross product
        areas = np.zeros(self.n_cells)

        for i, tri in enumerate(self.cells):
            v0, v1, v2 = self.vertices[tri]
            # Area = 0.5 * |cross product of two edge vectors|
            edge1 = v1 - v0
            edge2 = v2 - v0
            area = 0.5 * np.abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])
            areas[i] = area

        return areas

    def compute_neighbors(self, halo_width: int = 1) -> None:
        """Compute neighbor connectivity for halo exchange.

        For unstructured grids, this involves finding cells within
        halo_width distance (in terms of cell layers).

        Args:
            halo_width: Width of halo region in cell layers
        """
        if self.cell_neighbors is None:
            self._build_cell_neighbors()

        # For now, store direct neighbors
        # Full halo implementation will need multi-layer neighbor finding
        self.neighbors = {
            'direct': self.cell_neighbors,
            'halo_width': halo_width,
        }

    def is_structured(self) -> bool:
        """Check if grid has regular structured topology.

        Returns:
            False (triangular grids are unstructured)
        """
        return False

    def get_boundary_cells(self) -> np.ndarray:
        """Get indices of cells on the boundary.

        Returns:
            Array of boundary cell indices
        """
        if self.cell_neighbors is None:
            self._build_cell_neighbors()

        boundary = []
        for cell_id, neighbors in enumerate(self.cell_neighbors):
            # Boundary cells have fewer than 3 neighbors
            if len(neighbors) < 3:
                boundary.append(cell_id)

        return np.array(boundary)

    def create_partition_info(
        self,
        n_partitions: int = None,
        method: str = 'metis'
    ) -> dict:
        """Create graph-based partition metadata for halo exchange.

        Args:
            n_partitions: Number of partitions. If None, uses backend device count.
            method: Partitioning method ('metis' or 'geometric')

        Returns:
            Dict with partition information:
                - 'n_devices': number of partitions
                - 'device_to_cells': {device_id: [cell_indices]}
                - 'ghost_cell_map': {device_id: {neighbor_device: [cell_indices]}}
                - 'partition': array of partition IDs for each cell
                - 'strategy': 'unstructured'
                - 'info': additional partitioning statistics
        """
        # Import partitioning utilities
        from fesomx.utilities.partitioning import partition_triangular_mesh, build_ghost_cell_map

        # Determine number of partitions
        if n_partitions is None:
            if hasattr(self.backend, 'size'):
                n_partitions = self.backend.size
            else:
                n_partitions = 1

        # Partition the mesh
        partition, info = partition_triangular_mesh(
            self.vertices,
            self.cells,
            n_partitions,
            method=method
        )

        # Build device-to-cells mapping
        device_to_cells = {}
        for part_id in range(n_partitions):
            device_to_cells[part_id] = list(np.where(partition == part_id)[0])

        # Ghost cell map is already in info from partitioning
        # We need to rebuild it with the cell neighbors
        if self.cell_neighbors is None:
            self._build_cell_neighbors()

        from fesomx.utilities.partitioning import build_ghost_cell_map
        ghost_cell_map = build_ghost_cell_map(
            self.cell_neighbors,
            partition,
            halo_layers=1
        )

        # Build global-to-local index maps for explicit ghost cell storage
        # Each partition has: [owned_cells | ghost_cells_from_neighbor_0 | ...]
        global_to_local = {}
        local_array_sizes = {}

        for device_id in range(n_partitions):
            owned = device_to_cells[device_id]
            local_map = {}

            # Owned cells map to local indices 0...n_owned-1
            for local_idx, global_idx in enumerate(owned):
                local_map[global_idx] = local_idx

            # Ghost cells map to indices starting at n_owned
            next_idx = len(owned)
            if device_id in ghost_cell_map:
                for neighbor_id, ghost_indices in ghost_cell_map[device_id].items():
                    for global_idx in ghost_indices:
                        local_map[global_idx] = next_idx
                        next_idx += 1

            global_to_local[device_id] = local_map
            local_array_sizes[device_id] = next_idx

        return {
            'n_devices': n_partitions,
            'device_to_cells': device_to_cells,
            'ghost_cell_map': ghost_cell_map,
            'partition': partition,
            'strategy': 'unstructured',
            'info': info,
            'global_to_local': global_to_local,  # NEW: Index mapping
            'local_array_sizes': local_array_sizes,  # NEW: Size of local+ghost arrays
        }

    def __repr__(self) -> str:
        return (
            f"TriGrid(name='{self.name}', "
            f"n_cells={self.n_cells}, "
            f"n_vertices={self.n_vertices}, "
            f"stagger={self.stagger_type.value}, "
            f"backend={self.backend.name})"
        )
