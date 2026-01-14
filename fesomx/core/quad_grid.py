"""Quadrilateral (structured) grid implementation."""

from typing import Tuple, Optional
import numpy as np

from .base_grid import BaseGrid
from .backend import Backend
from .staggering import StaggerType


class QuadGrid(BaseGrid):
    """Structured quadrilateral grid.

    Represents a logically rectangular grid (may be curvilinear in physical space).
    """

    def __init__(
        self,
        name: str,
        nx: int,
        ny: int,
        stagger_type: StaggerType = StaggerType.A,
        backend: Optional[Backend] = None,
        periodic: Tuple[bool, bool] = (False, False),
    ):
        """Initialize quadrilateral grid.

        Args:
            name: Grid identifier
            nx: Number of cells in x-direction
            ny: Number of cells in y-direction
            stagger_type: Type of grid staggering
            backend: Parallel backend
            periodic: Periodic boundaries in (x, y)
        """
        super().__init__(name, stagger_type, backend, periodic)
        self.nx = nx
        self.ny = ny

        # Structured grid dimensions
        self.shape = (ny, nx)  # Following numpy convention (rows, cols)

    def create_mesh(
        self,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
        domain: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
    ) -> None:
        """Create structured quadrilateral mesh.

        Args:
            x_coords: Custom x-coordinates (nx+1,) for vertices
            y_coords: Custom y-coordinates (ny+1,) for vertices
            domain: (x_min, x_max, y_min, y_max) if coords not provided
        """
        if x_coords is None:
            x_min, x_max = domain[0], domain[1]
            x_coords = np.linspace(x_min, x_max, self.nx + 1)

        if y_coords is None:
            y_min, y_max = domain[2], domain[3]
            y_coords = np.linspace(y_min, y_max, self.ny + 1)

        # Create vertex coordinates (meshgrid)
        X, Y = np.meshgrid(x_coords, y_coords)
        self.vertices = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Cell connectivity (each quad has 4 vertices)
        # Vertices numbered in row-major order
        self.cells = []
        for j in range(self.ny):
            for i in range(self.nx):
                # Bottom-left, bottom-right, top-right, top-left
                v0 = j * (self.nx + 1) + i
                v1 = v0 + 1
                v2 = v0 + (self.nx + 1) + 1
                v3 = v0 + (self.nx + 1)
                self.cells.append([v0, v1, v2, v3])
        self.cells = np.array(self.cells)

        # Store coordinate arrays for structured operations
        self.x_coords = x_coords
        self.y_coords = y_coords

    def get_cell_centers(self) -> np.ndarray:
        """Get coordinates of cell centers.

        Returns:
            Array of shape (n_cells, 2) with (x, y) coordinates
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        # For structured grid, centers are midpoints
        cell_vertices = self.vertices[self.cells]
        centers = cell_vertices.mean(axis=1)
        return centers

    def get_cell_volumes(self) -> np.ndarray:
        """Get cell areas.

        Returns:
            Array of shape (n_cells,) with cell areas
        """
        if self.cells is None:
            raise ValueError("Mesh not created. Call create_mesh() first.")

        # For quads, compute area using shoelace formula
        areas = np.zeros(self.n_cells)
        for i, cell in enumerate(self.cells):
            verts = self.vertices[cell]
            # Shoelace formula for polygon area
            x = verts[:, 0]
            y = verts[:, 1]
            area = 0.5 * np.abs(
                np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
            )
            areas[i] = area

        return areas

    def compute_neighbors(self, halo_width: int = 1) -> None:
        """Compute neighbor connectivity for halo exchange.

        For structured grids, neighbors are adjacent cells in i, j directions.

        Args:
            halo_width: Width of halo region in cells
        """
        # For structured grids, neighbors are straightforward
        # Store as directional neighbors: 'left', 'right', 'top', 'bottom'
        self.neighbors = {
            'left': -1 if not self.periodic[0] else self.nx - 1,
            'right': self.nx if not self.periodic[0] else 0,
            'bottom': -1 if not self.periodic[1] else self.ny - 1,
            'top': self.ny if not self.periodic[1] else 0,
        }
        self.halo_width = halo_width

    def is_structured(self) -> bool:
        """Check if grid has regular structured topology.

        Returns:
            True (quadrilateral grids are always structured)
        """
        return True

    def get_logical_indices(self, cell_id: int) -> Tuple[int, int]:
        """Convert linear cell index to (i, j) logical indices.

        Args:
            cell_id: Linear cell index

        Returns:
            (i, j) tuple where i is x-index, j is y-index
        """
        j = cell_id // self.nx
        i = cell_id % self.nx
        return (i, j)

    def get_linear_index(self, i: int, j: int) -> int:
        """Convert logical (i, j) indices to linear cell index.

        Args:
            i: x-index
            j: y-index

        Returns:
            Linear cell index
        """
        return j * self.nx + i

    def create_partition_info(self, mesh_shape: Tuple[int, int] = None) -> dict:
        """Create structured partition metadata for halo exchange.

        Args:
            mesh_shape: (n_rows, n_cols) device mesh shape.
                       If None, uses backend's mesh if available.

        Returns:
            Dict with partition information:
                - 'mesh_shape': (n_devices_y, n_devices_x)
                - 'periodic': (periodic_x, periodic_y)
                - 'grid_shape': (ny, nx) global grid size
                - 'device_to_cells': {device_id: (j_start, j_end, i_start, i_end)}
                - 'edge_indices': edge cell indices for each direction
                - 'neighbor_map': {device_id: {direction: neighbor_device_id}}
        """
        # Get mesh shape from backend if not provided
        if mesh_shape is None:
            if hasattr(self.backend, 'mesh') and self.backend.mesh is not None:
                mesh_shape = self.backend.mesh.shape
            else:
                mesh_shape = (1, 1)

        n_devices_y, n_devices_x = mesh_shape
        n_devices = n_devices_y * n_devices_x

        # Calculate cells per device
        cells_per_device_y = self.ny // n_devices_y
        cells_per_device_x = self.nx // n_devices_x

        # Build device-to-cells mapping
        device_to_cells = {}
        for dev_j in range(n_devices_y):
            for dev_i in range(n_devices_x):
                device_id = dev_j * n_devices_x + dev_i

                j_start = dev_j * cells_per_device_y
                j_end = (dev_j + 1) * cells_per_device_y if dev_j < n_devices_y - 1 else self.ny

                i_start = dev_i * cells_per_device_x
                i_end = (dev_i + 1) * cells_per_device_x if dev_i < n_devices_x - 1 else self.nx

                device_to_cells[device_id] = (j_start, j_end, i_start, i_end)

        # Build neighbor map
        neighbor_map = {}
        for dev_j in range(n_devices_y):
            for dev_i in range(n_devices_x):
                device_id = dev_j * n_devices_x + dev_i
                neighbors = {}

                # Left neighbor
                if dev_i > 0:
                    neighbors['left'] = dev_j * n_devices_x + (dev_i - 1)
                elif self.periodic[0]:
                    neighbors['left'] = dev_j * n_devices_x + (n_devices_x - 1)

                # Right neighbor
                if dev_i < n_devices_x - 1:
                    neighbors['right'] = dev_j * n_devices_x + (dev_i + 1)
                elif self.periodic[0]:
                    neighbors['right'] = dev_j * n_devices_x + 0

                # Top neighbor
                if dev_j > 0:
                    neighbors['top'] = (dev_j - 1) * n_devices_x + dev_i
                elif self.periodic[1]:
                    neighbors['top'] = (n_devices_y - 1) * n_devices_x + dev_i

                # Bottom neighbor
                if dev_j < n_devices_y - 1:
                    neighbors['bottom'] = (dev_j + 1) * n_devices_x + dev_i
                elif self.periodic[1]:
                    neighbors['bottom'] = 0 * n_devices_x + dev_i

                neighbor_map[device_id] = neighbors

        return {
            'mesh_shape': mesh_shape,
            'periodic': self.periodic,
            'grid_shape': (self.ny, self.nx),
            'device_to_cells': device_to_cells,
            'neighbor_map': neighbor_map,
            'strategy': 'structured',
        }

    def __repr__(self) -> str:
        return (
            f"QuadGrid(name='{self.name}', "
            f"shape=({self.ny}, {self.nx}), "
            f"stagger={self.stagger_type.value}, "
            f"backend={self.backend.name})"
        )
