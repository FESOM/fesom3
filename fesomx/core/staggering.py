"""Grid staggering definitions (Arakawa A, B, C grids)."""

from enum import Enum
from typing import NamedTuple, Tuple


class StaggerType(Enum):
    """Grid staggering types.

    A-grid: All variables co-located at cell centers
    B-grid: Velocity components at cell corners, scalars at centers
    C-grid: Velocity components staggered on cell faces, scalars at centers
    """
    A = "A"  # All variables at cell centers
    B = "B"  # Velocities at corners, scalars at centers
    C = "C"  # Velocities at faces, scalars at centers


class StaggerLocation(Enum):
    """Location of variables on the grid."""
    CENTER = "center"      # Cell center
    CORNER = "corner"      # Cell corner/vertex
    EDGE_X = "edge_x"      # Edge in x-direction
    EDGE_Y = "edge_y"      # Edge in y-direction
    FACE_X = "face_x"      # Face normal to x-axis
    FACE_Y = "face_y"      # Face normal to y-axis


class StaggeredField(NamedTuple):
    """Represents a field with its stagger location.

    Attributes:
        name: Field name (e.g., 'u', 'v', 'p', 'T')
        location: Where the field is located on the grid
        stagger_type: Grid staggering type being used
    """
    name: str
    location: StaggerLocation
    stagger_type: StaggerType

    def get_offset(self) -> Tuple[float, float]:
        """Get the offset from cell center in grid units.

        Returns:
            (offset_x, offset_y) in units of cell spacing
        """
        if self.location == StaggerLocation.CENTER:
            return (0.0, 0.0)
        elif self.location == StaggerLocation.CORNER:
            return (0.5, 0.5)
        elif self.location == StaggerLocation.EDGE_X:
            return (0.5, 0.0)
        elif self.location == StaggerLocation.EDGE_Y:
            return (0.0, 0.5)
        elif self.location == StaggerLocation.FACE_X:
            return (0.5, 0.0)
        elif self.location == StaggerLocation.FACE_Y:
            return (0.0, 0.5)
        else:
            raise ValueError(f"Unknown stagger location: {self.location}")


def get_stagger_config(stagger_type: StaggerType) -> dict:
    """Get the standard variable locations for a stagger type.

    Args:
        stagger_type: Type of staggering (A, B, or C)

    Returns:
        Dictionary mapping variable names to their locations
    """
    if stagger_type == StaggerType.A:
        # A-grid: all variables at centers
        return {
            'u': StaggerLocation.CENTER,
            'v': StaggerLocation.CENTER,
            'p': StaggerLocation.CENTER,
            'T': StaggerLocation.CENTER,
        }
    elif stagger_type == StaggerType.B:
        # B-grid: velocities at corners, scalars at centers
        return {
            'u': StaggerLocation.CORNER,
            'v': StaggerLocation.CORNER,
            'p': StaggerLocation.CENTER,
            'T': StaggerLocation.CENTER,
        }
    elif stagger_type == StaggerType.C:
        # C-grid: velocities on faces, scalars at centers
        return {
            'u': StaggerLocation.FACE_X,
            'v': StaggerLocation.FACE_Y,
            'p': StaggerLocation.CENTER,
            'T': StaggerLocation.CENTER,
        }
    else:
        raise ValueError(f"Unknown stagger type: {stagger_type}")


def create_staggered_field(
    name: str,
    stagger_type: StaggerType,
    custom_location: StaggerLocation = None
) -> StaggeredField:
    """Create a staggered field with automatic location assignment.

    Args:
        name: Field name
        stagger_type: Grid staggering type
        custom_location: Override default location for this field

    Returns:
        StaggeredField instance
    """
    if custom_location is not None:
        location = custom_location
    else:
        config = get_stagger_config(stagger_type)
        location = config.get(name, StaggerLocation.CENTER)

    return StaggeredField(name, location, stagger_type)
