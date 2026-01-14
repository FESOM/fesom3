"""Halo pattern definitions for different exchange scenarios."""

from enum import Enum
from typing import List, Tuple
import numpy as np


class HaloPattern(Enum):
    """Common halo exchange patterns."""
    STAR = "star"          # 4-point stencil (N, S, E, W)
    BOX = "box"            # 8-point stencil (includes diagonals)
    EXTENDED = "extended"  # Multi-layer halo


class HaloWidth:
    """Configuration for halo region width.

    Can specify different widths for different directions or uniform width.
    """

    def __init__(self, width: int | Tuple[int, int] | dict):
        """Initialize halo width configuration.

        Args:
            width: Either:
                - int: uniform width in all directions
                - Tuple[int, int]: (width_x, width_y)
                - dict: {'left': w, 'right': w, 'top': w, 'bottom': w}
        """
        if isinstance(width, int):
            self.uniform = True
            self.width = width
            self.width_x = width
            self.width_y = width
            self.widths = {
                'left': width,
                'right': width,
                'top': width,
                'bottom': width,
            }
        elif isinstance(width, tuple):
            self.uniform = False
            self.width_x, self.width_y = width
            self.width = max(width)
            self.widths = {
                'left': self.width_x,
                'right': self.width_x,
                'top': self.width_y,
                'bottom': self.width_y,
            }
        elif isinstance(width, dict):
            self.uniform = False
            self.widths = width
            self.width_x = max(width.get('left', 1), width.get('right', 1))
            self.width_y = max(width.get('top', 1), width.get('bottom', 1))
            self.width = max(self.width_x, self.width_y)
        else:
            raise ValueError(f"Invalid width specification: {width}")

    def get_direction_width(self, direction: str) -> int:
        """Get halo width for a specific direction.

        Args:
            direction: 'left', 'right', 'top', or 'bottom'

        Returns:
            Halo width for that direction
        """
        return self.widths.get(direction, self.width)

    def __repr__(self) -> str:
        if self.uniform:
            return f"HaloWidth(uniform={self.width})"
        return f"HaloWidth({self.widths})"


def get_halo_indices_2d(
    shape: Tuple[int, int],
    halo_width: HaloWidth,
    direction: str,
) -> Tuple[slice, slice]:
    """Get slice indices for halo region in a 2D array.

    Args:
        shape: Shape of the array (ny, nx)
        halo_width: Halo width configuration
        direction: Direction ('left', 'right', 'top', 'bottom')

    Returns:
        (slice_y, slice_x) for accessing the halo region
    """
    ny, nx = shape
    w = halo_width.get_direction_width(direction)

    if direction == 'left':
        return (slice(None), slice(0, w))
    elif direction == 'right':
        return (slice(None), slice(nx - w, nx))
    elif direction == 'bottom':
        return (slice(0, w), slice(None))
    elif direction == 'top':
        return (slice(ny - w, ny), slice(None))
    else:
        raise ValueError(f"Unknown direction: {direction}")


def get_interior_indices_2d(
    shape: Tuple[int, int],
    halo_width: HaloWidth,
) -> Tuple[slice, slice]:
    """Get slice indices for interior (non-halo) region.

    Args:
        shape: Shape of the array (ny, nx)
        halo_width: Halo width configuration

    Returns:
        (slice_y, slice_x) for accessing interior
    """
    ny, nx = shape
    wx = halo_width.width_x
    wy = halo_width.width_y

    return (slice(wy, ny - wy), slice(wx, nx - wx))


def get_neighbor_directions(pattern: HaloPattern) -> List[str]:
    """Get list of neighbor directions for a halo pattern.

    Args:
        pattern: Halo exchange pattern

    Returns:
        List of direction strings
    """
    if pattern == HaloPattern.STAR:
        return ['left', 'right', 'top', 'bottom']
    elif pattern == HaloPattern.BOX:
        return [
            'left', 'right', 'top', 'bottom',
            'top-left', 'top-right', 'bottom-left', 'bottom-right'
        ]
    elif pattern == HaloPattern.EXTENDED:
        return ['left', 'right', 'top', 'bottom']  # Can be extended
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def compute_halo_buffer_size(
    local_shape: Tuple[int, int],
    halo_width: HaloWidth,
    direction: str,
) -> int:
    """Compute size of halo buffer for a direction.

    Args:
        local_shape: Shape of local data (ny, nx)
        halo_width: Halo width configuration
        direction: Direction of exchange

    Returns:
        Number of elements in the halo buffer
    """
    ny, nx = local_shape
    w = halo_width.get_direction_width(direction)

    if direction in ['left', 'right']:
        return ny * w
    elif direction in ['top', 'bottom']:
        return nx * w
    elif direction in ['top-left', 'top-right', 'bottom-left', 'bottom-right']:
        return w * w
    else:
        raise ValueError(f"Unknown direction: {direction}")
