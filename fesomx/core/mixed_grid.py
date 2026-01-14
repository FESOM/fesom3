"""Mixed quadrilateral and triangular grid implementation."""

from typing import Optional, Tuple, List
import numpy as np

from .base_grid import BaseGrid
from .backend import Backend
from .staggering import StaggerType


class MixedGrid(BaseGrid):
    """Mixed unstructured grid with quads and triangles.

    Represents a mesh that can contain both triangular and quadrilateral cells.
    """

    def __init__(
        self,
        name: str,
        stagger_type: StaggerType = StaggerType.A,
        backend: Optional[Backend] = None,
        periodic: Tuple[bool, bool] = (False, False),
    ):
        """Initialize mixed grid.

        Args:
            name: Grid identifier
            stagger_type: Type of grid staggering
            backend: Parallel backend
            periodic: Periodic boundaries (limited support for unstructured)
        """
        super().__init__(name, stagger_type, backend, periodic)

        # Mixed grid specific attributes
        self.cell_types = None  # Array indicating cell type (3=tri, 4=quad)
        self.cell_neighbors = None

    def create_mesh(
        self,
        vertices: np.ndarray,
        cells: List[np.ndarray],
    ) -> None:
        """Create mixed mesh from vertices and variable-connectivity cells.

        Args:
            vertices: Array of shape (n_vertices, 2) with (x, y) coordinates
            cells: List of arrays, each containing vertex indices for a cell.
                   Length 3 = triangle, length 4 = quadrilateral
        """
        self.vertices = vertices

        # Store cells and their types
        self.cells = cells
        self.cell_types = np.array([len(cell) for cell in cells])

        # Validate cell types
        if not all(ct in [3, 4] for ct in self.cell_types):
            raise ValueError("Only triangular (3) and quadrilateral (4) cells supported")

        # Build connectivity
        self._build_edges()
        self._build_cell_neighbors()

    def _build_edges(self) -> None:
        """Build edge connectivity from cell definitions."""
        edges_set = set()

        for cell in self.cells:
            n_verts = len(cell)
            for i in range(n_verts):
                v0 = cell[i]
                v1 = cell[(i + 1) % n_verts]
                edge = tuple(sorted([v0, v1]))
                edges_set.add(edge)

        self.edges = np.array(list(edges_set))

    def _build_cell_neighbors(self) -> None:
        """Build cell neighbor connectivity.

        Two cells are neighbors if they share an edge.
        For periodic boundaries, cells on opposite boundary edges are connected.
        """
        # Create edge-to-cell mapping
        edge_to_cells = {}

        for cell_id, cell in enumerate(self.cells):
            n_verts = len(cell)
            for i in range(n_verts):
                v0 = cell[i]
                v1 = cell[(i + 1) % n_verts]
                edge = tuple(sorted([v0, v1]))

                if edge not in edge_to_cells:
                    edge_to_cells[edge] = []
                edge_to_cells[edge].append(cell_id)

        # Build neighbor list for each cell
        self.cell_neighbors = [[] for _ in range(self.n_cells)]

        # Track boundary edges for periodic matching
        boundary_edges = {}  # edge -> cell_id

        for edge, cells_sharing in edge_to_cells.items():
            if len(cells_sharing) == 2:
                c0, c1 = cells_sharing
                self.cell_neighbors[c0].append(c1)
                self.cell_neighbors[c1].append(c0)
            elif len(cells_sharing) == 1:
                # Boundary edge
                boundary_edges[edge] = cells_sharing[0]

        # Handle periodic boundaries if enabled
        if any(self.periodic) and len(boundary_edges) > 0:
            self._connect_periodic_boundaries(boundary_edges, edge_to_cells)

    def _connect_periodic_boundaries(
        self,
        boundary_edges: dict,
        edge_to_cells: dict
    ) -> None:
        """Connect cells across periodic boundaries.

        For structured mixed grids on rectangular domains:
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
        """Get coordinates of cell centers.

        Returns:
            Array of shape (n_cells, 2) with (x, y) coordinates
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        centers = np.zeros((self.n_cells, 2))

        for i, cell in enumerate(self.cells):
            cell_verts = self.vertices[cell]
            centers[i] = cell_verts.mean(axis=0)

        return centers

    def get_cell_volumes(self) -> np.ndarray:
        """Get cell areas.

        Returns:
            Array of shape (n_cells,) with cell areas
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        areas = np.zeros(self.n_cells)

        for i, cell in enumerate(self.cells):
            verts = self.vertices[cell]

            if self.cell_types[i] == 3:
                # Triangle area using cross product
                v0, v1, v2 = verts
                edge1 = v1 - v0
                edge2 = v2 - v0
                area = 0.5 * np.abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])
            else:
                # Quadrilateral area using shoelace formula
                x = verts[:, 0]
                y = verts[:, 1]
                area = 0.5 * np.abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )

            areas[i] = area

        return areas

    def compute_neighbors(self, halo_width: int = 1) -> None:
        """Compute neighbor connectivity for halo exchange.

        Args:
            halo_width: Width of halo region in cell layers
        """
        if self.cell_neighbors is None:
            self._build_cell_neighbors()

        self.neighbors = {
            'direct': self.cell_neighbors,
            'halo_width': halo_width,
        }

    def is_structured(self) -> bool:
        """Check if grid has regular structured topology.

        Returns:
            False (mixed grids are unstructured)
        """
        return False

    def get_triangular_cells(self) -> np.ndarray:
        """Get indices of triangular cells.

        Returns:
            Array of cell indices that are triangles
        """
        return np.where(self.cell_types == 3)[0]

    def get_quadrilateral_cells(self) -> np.ndarray:
        """Get indices of quadrilateral cells.

        Returns:
            Array of cell indices that are quadrilaterals
        """
        return np.where(self.cell_types == 4)[0]

    def get_boundary_cells(self) -> np.ndarray:
        """Get indices of cells on the boundary.

        Returns:
            Array of boundary cell indices
        """
        if self.cell_neighbors is None:
            self._build_cell_neighbors()

        boundary = []
        for cell_id, neighbors in enumerate(self.cell_neighbors):
            expected_neighbors = self.cell_types[cell_id]
            if len(neighbors) < expected_neighbors:
                boundary.append(cell_id)

        return np.array(boundary)

    def __repr__(self) -> str:
        n_tri = np.sum(self.cell_types == 3) if self.cell_types is not None else 0
        n_quad = np.sum(self.cell_types == 4) if self.cell_types is not None else 0

        return (
            f"MixedGrid(name='{self.name}', "
            f"n_cells={self.n_cells} ({n_tri} tri, {n_quad} quad), "
            f"n_vertices={self.n_vertices}, "
            f"stagger={self.stagger_type.value}, "
            f"backend={self.backend.name})"
        )
