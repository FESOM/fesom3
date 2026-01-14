"""Abstract base class for grid structures."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .backend import Backend, get_backend
from .staggering import StaggerType, StaggeredField, StaggerLocation


class BaseGrid(ABC):
    """Abstract base class for all grid types.

    Provides common interface for quadrilateral, triangular, and mixed grids
    with support for different staggering schemes.
    """

    def __init__(
        self,
        name: str,
        stagger_type: StaggerType = StaggerType.A,
        backend: Optional[Backend] = None,
        periodic: Tuple[bool, bool] = (False, False),
    ):
        """Initialize base grid.

        Args:
            name: Grid identifier
            stagger_type: Type of grid staggering (A, B, or C)
            backend: Parallel backend (defaults to JAX sharding)
            periodic: Periodic boundary conditions in (x, y)
        """
        self.name = name
        self.stagger_type = stagger_type
        self.backend = backend if backend is not None else get_backend()
        self.periodic = periodic

        # Grid geometry (to be set by subclasses)
        self.vertices = None  # Vertex coordinates
        self.cells = None     # Cell connectivity
        self.edges = None     # Edge connectivity

        # Fields stored on the grid
        self.fields: Dict[str, Any] = {}

        # Neighbor information for halo exchange
        self.neighbors: Dict[str, int] = {}

    @abstractmethod
    def create_mesh(self, *args, **kwargs) -> None:
        """Create the grid mesh structure.

        To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def get_cell_centers(self) -> np.ndarray:
        """Get coordinates of cell centers.

        Returns:
            Array of shape (n_cells, 2) with (x, y) coordinates
        """
        pass

    @abstractmethod
    def get_cell_volumes(self) -> np.ndarray:
        """Get volumes/areas of cells.

        Returns:
            Array of shape (n_cells,) with cell areas
        """
        pass

    @abstractmethod
    def compute_neighbors(self, halo_width: int = 1) -> None:
        """Compute neighbor connectivity for halo exchange.

        Args:
            halo_width: Width of halo region in cells
        """
        pass

    @abstractmethod
    def is_structured(self) -> bool:
        """Check if grid has regular structured topology.

        Returns:
            True if grid is structured (regular i,j indexing),
            False for unstructured grids (irregular connectivity)
        """
        pass

    def add_field(
        self,
        field_name: str,
        data: np.ndarray,
        location: Optional[StaggerLocation] = None,
        sharding_spec: Optional[Any] = None,
    ) -> None:
        """Add a field to the grid.

        Args:
            field_name: Name of the field (e.g., 'u', 'v', 'p', 'T')
            data: Field data as numpy array
            location: Where field is located (None uses stagger config)
            sharding_spec: Optional sharding specification for backend
        """
        staggered_field = StaggeredField(
            field_name,
            location if location is not None else StaggerLocation.CENTER,
            self.stagger_type
        )

        # Convert to backend array
        backend_array = self.backend.array(data, sharding_spec)

        self.fields[field_name] = {
            'data': backend_array,
            'stagger': staggered_field,
            'sharding': sharding_spec,
        }

    def get_field(self, field_name: str) -> Any:
        """Get field data.

        Args:
            field_name: Name of the field

        Returns:
            Backend-specific array
        """
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not found on grid '{self.name}'")
        return self.fields[field_name]['data']

    def get_field_numpy(self, field_name: str) -> np.ndarray:
        """Get field data as numpy array.

        Args:
            field_name: Name of the field

        Returns:
            Numpy array
        """
        field_data = self.get_field(field_name)
        return self.backend.to_numpy(field_data)

    def halo_exchange(self, field_name: str, halo_width: int = 1) -> None:
        """Perform halo exchange for a field.

        Args:
            field_name: Name of the field to exchange
            halo_width: Width of halo region
        """
        if field_name not in self.fields:
            raise KeyError(f"Field '{field_name}' not found")

        field_data = self.fields[field_name]['data']

        # Perform halo exchange using backend
        updated_data = self.backend.halo_exchange(
            field_data,
            halo_width,
            self.neighbors,
            self.periodic
        )

        self.fields[field_name]['data'] = updated_data

    @property
    def n_cells(self) -> int:
        """Number of cells in the grid."""
        if self.cells is not None:
            return len(self.cells)
        return 0

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the grid."""
        if self.vertices is not None:
            return len(self.vertices)
        return 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"stagger={self.stagger_type.value}, "
            f"n_cells={self.n_cells}, "
            f"n_vertices={self.n_vertices}, "
            f"backend={self.backend.name})"
        )
