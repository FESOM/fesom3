"""
Boundary Treatment for Unstructured Grid Operators

This module provides ghost cell extrapolation for improving gradient accuracy
at non-periodic boundaries. Essential for ocean grids with coastlines and islands.

Methods:
    - Linear extrapolation: φ_ghost = 2*φ_0 - φ_1 (O(h²))
    - Quadratic extrapolation: φ_ghost = 3*φ_0 - 3*φ_1 + φ_2 (O(h³))
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from fesomx.core import BaseGrid


def identify_boundary_cells(
    grid: BaseGrid,
    land_mask: Optional[np.ndarray] = None
) -> Dict:
    """
    Identify boundary cells and compute extrapolation information.

    For each boundary cell, finds:
    - The boundary edge(s)
    - The outward normal direction
    - Donor cells for extrapolation (interior neighbors along the normal)
    - Ghost cell position (reflected across boundary)

    Args:
        grid: Unstructured grid (TriGrid or MixedGrid)
        land_mask: Optional boolean array where True = land/masked cells.
                   If provided, cells adjacent to land are treated as boundaries.

    Returns:
        boundary_info: Dict containing:
            - 'boundary_cells': List of boundary cell indices
            - 'ghost_info': Dict mapping boundary cell -> ghost cell data
            - 'n_ghosts': Number of ghost cells created
    """
    n_cells = grid.n_cells
    cell_centers = grid.get_cell_centers()
    vertices = grid.vertices
    cells = grid.cells

    # Build cell neighbors if not available
    if grid.cell_neighbors is None:
        grid.compute_neighbors(halo_width=1)

    cell_neighbors = grid.cell_neighbors

    # Identify boundary cells
    boundary_cells = []
    ghost_info = {}

    for cell_id in range(n_cells):
        # Skip masked (land) cells
        if land_mask is not None and land_mask[cell_id]:
            continue

        neighbors = cell_neighbors[cell_id]
        cell_verts = cells[cell_id]
        n_verts = len(cell_verts)

        # Check each edge for boundary condition
        for edge_idx in range(n_verts):
            v0 = cell_verts[edge_idx]
            v1 = cell_verts[(edge_idx + 1) % n_verts]

            # Find neighbor across this edge
            edge_neighbor = _find_edge_neighbor(grid, cell_id, v0, v1)

            is_boundary_edge = False

            if edge_neighbor < 0:
                # Domain boundary (no neighbor)
                is_boundary_edge = True
            elif land_mask is not None and land_mask[edge_neighbor]:
                # Internal boundary (neighbor is land)
                is_boundary_edge = True

            if is_boundary_edge:
                if cell_id not in boundary_cells:
                    boundary_cells.append(cell_id)

                # Compute ghost cell info for this edge
                ghost_data = _compute_ghost_for_edge(
                    grid, cell_id, v0, v1, cell_centers, vertices,
                    cell_neighbors, land_mask
                )

                if ghost_data is not None:
                    edge_key = (cell_id, tuple(sorted([v0, v1])))
                    ghost_info[edge_key] = ghost_data

    return {
        'boundary_cells': boundary_cells,
        'ghost_info': ghost_info,
        'n_ghosts': len(ghost_info),
        'land_mask': land_mask,
    }


def _find_edge_neighbor(grid: BaseGrid, cell_id: int, v0: int, v1: int) -> int:
    """Find the neighbor cell across the edge (v0, v1)."""
    cells = grid.cells
    n_cells = grid.n_cells

    edge = tuple(sorted([v0, v1]))

    for other_id in range(n_cells):
        if other_id == cell_id:
            continue
        other_verts = cells[other_id]
        n_other = len(other_verts)
        for i in range(n_other):
            other_edge = tuple(sorted([other_verts[i], other_verts[(i + 1) % n_other]]))
            if other_edge == edge:
                return other_id

    return -1  # No neighbor (boundary edge)


def _compute_ghost_for_edge(
    grid: BaseGrid,
    cell_id: int,
    v0: int, v1: int,
    cell_centers: np.ndarray,
    vertices: np.ndarray,
    cell_neighbors: List,
    land_mask: Optional[np.ndarray]
) -> Optional[Dict]:
    """
    Compute ghost cell information for a boundary edge.

    Returns ghost cell position and donor cell indices for extrapolation.
    """
    # Edge midpoint and normal
    p0 = vertices[v0]
    p1 = vertices[v1]
    edge_mid = 0.5 * (p0 + p1)

    # Edge tangent and outward normal
    edge_vec = p1 - p0
    edge_length = np.linalg.norm(edge_vec)
    if edge_length < 1e-14:
        return None

    # Outward normal (perpendicular to edge, pointing out of cell)
    normal = np.array([edge_vec[1], -edge_vec[0]]) / edge_length

    # Check normal direction (should point away from cell center)
    cell_center = cell_centers[cell_id]
    to_edge = edge_mid - cell_center
    if np.dot(normal, to_edge) < 0:
        normal = -normal

    # Distance from cell center to edge
    dist_to_edge = np.abs(np.dot(to_edge, normal))

    # Ghost cell position: reflect cell center across edge
    ghost_position = cell_center + 2 * dist_to_edge * normal

    # Find donor cells for extrapolation (cells along inward normal)
    donors = _find_donor_cells(
        grid, cell_id, -normal, cell_centers, cell_neighbors, land_mask, max_donors=2
    )

    if len(donors) < 1:
        # Can't extrapolate without donors, use cell value
        donors = [cell_id]

    return {
        'ghost_position': ghost_position,
        'boundary_cell': cell_id,
        'edge_vertices': (v0, v1),
        'edge_midpoint': edge_mid,
        'outward_normal': normal,
        'donor_cells': donors,
        'dist_to_edge': dist_to_edge,
    }


def _find_donor_cells(
    grid: BaseGrid,
    boundary_cell: int,
    inward_normal: np.ndarray,
    cell_centers: np.ndarray,
    cell_neighbors: List,
    land_mask: Optional[np.ndarray],
    max_donors: int = 2
) -> List[int]:
    """
    Find donor cells for extrapolation along the inward normal direction.

    Starts from boundary cell and walks inward, selecting cells that are
    approximately along the normal direction.
    """
    donors = [boundary_cell]
    visited = {boundary_cell}
    current = boundary_cell

    for _ in range(max_donors - 1):
        neighbors = cell_neighbors[current]
        best_neighbor = -1
        best_alignment = -2.0  # cos(angle), -1 to 1

        for neighbor in neighbors:
            if neighbor in visited:
                continue
            if land_mask is not None and land_mask[neighbor]:
                continue

            # Check alignment with inward normal
            direction = cell_centers[neighbor] - cell_centers[current]
            dist = np.linalg.norm(direction)
            if dist < 1e-14:
                continue

            alignment = np.dot(direction / dist, inward_normal)

            if alignment > best_alignment:
                best_alignment = alignment
                best_neighbor = neighbor

        if best_neighbor >= 0 and best_alignment > 0.3:  # At least ~70° alignment
            donors.append(best_neighbor)
            visited.add(best_neighbor)
            current = best_neighbor
        else:
            break

    return donors


def compute_ghost_values(
    phi: np.ndarray,
    boundary_info: Dict,
    extrapolation_order: int = 1,
    cell_centers: Optional[np.ndarray] = None,
    use_gradient_extrapolation: bool = False,
) -> Dict:
    """
    Compute ghost cell values using polynomial extrapolation.

    Args:
        phi: Scalar field at cell centers (n_cells,)
        boundary_info: From identify_boundary_cells()
        extrapolation_order: 1 for linear, 2 for quadratic
        cell_centers: Cell center coordinates (required if use_gradient_extrapolation=True)
        use_gradient_extrapolation: If True, use local gradient to extrapolate
                                    (better for LS methods)

    Returns:
        ghost_values: Dict mapping (cell_id, edge) -> ghost value
    """
    ghost_info = boundary_info['ghost_info']
    ghost_values = {}

    for edge_key, info in ghost_info.items():
        boundary_cell = info['boundary_cell']
        ghost_position = info['ghost_position']
        donors = info['donor_cells']
        n_donors = len(donors)

        if use_gradient_extrapolation and cell_centers is not None and n_donors >= 2:
            # Gradient-based extrapolation: φ_ghost = φ_0 + ∇φ · (r_ghost - r_0)
            # Compute local gradient from neighbors
            phi_0 = float(phi[boundary_cell])
            x_0, y_0 = cell_centers[boundary_cell]
            x_g, y_g = ghost_position

            # Simple least-squares gradient from donors
            a11, a12, a22, b1, b2 = 0.0, 0.0, 0.0, 0.0, 0.0
            for donor_id in donors:
                if donor_id != boundary_cell:
                    x_d, y_d = cell_centers[donor_id]
                    phi_d = float(phi[donor_id])
                    dx = x_d - x_0
                    dy = y_d - y_0
                    dphi = phi_d - phi_0
                    dist_sq = dx**2 + dy**2
                    if dist_sq > 1e-14:
                        w = 1.0 / dist_sq
                        a11 += w * dx * dx
                        a12 += w * dx * dy
                        a22 += w * dy * dy
                        b1 += w * dx * dphi
                        b2 += w * dy * dphi

            det = a11 * a22 - a12 * a12
            if abs(det) > 1e-14:
                grad_x = (a22 * b1 - a12 * b2) / det
                grad_y = (a11 * b2 - a12 * b1) / det
                # Extrapolate to ghost position
                dx_ghost = x_g - x_0
                dy_ghost = y_g - y_0
                ghost_values[edge_key] = phi_0 + grad_x * dx_ghost + grad_y * dy_ghost
            else:
                # Fall back to zero-order
                ghost_values[edge_key] = phi_0

        elif n_donors == 0:
            # No donors, use boundary cell value
            ghost_values[edge_key] = float(phi[info['boundary_cell']])
        elif n_donors == 1 or extrapolation_order == 0:
            # Zero-order: just copy boundary value
            ghost_values[edge_key] = float(phi[donors[0]])
        elif n_donors == 2 and extrapolation_order >= 1:
            # Linear extrapolation: φ_ghost = 2*φ_0 - φ_1
            phi_0 = float(phi[donors[0]])
            phi_1 = float(phi[donors[1]])
            ghost_values[edge_key] = 2 * phi_0 - phi_1
        elif n_donors >= 3 and extrapolation_order >= 2:
            # Quadratic extrapolation: φ_ghost = 3*φ_0 - 3*φ_1 + φ_2
            phi_0 = float(phi[donors[0]])
            phi_1 = float(phi[donors[1]])
            phi_2 = float(phi[donors[2]])
            ghost_values[edge_key] = 3 * phi_0 - 3 * phi_1 + phi_2
        else:
            # Fall back to linear
            phi_0 = float(phi[donors[0]])
            phi_1 = float(phi[donors[1]]) if n_donors > 1 else phi_0
            ghost_values[edge_key] = 2 * phi_0 - phi_1

    return ghost_values


def get_ghost_for_cell_edge(
    cell_id: int,
    v0: int, v1: int,
    ghost_values: Dict,
    phi: np.ndarray
) -> float:
    """
    Get ghost value for a specific cell edge.

    Args:
        cell_id: Cell index
        v0, v1: Edge vertex indices
        ghost_values: From compute_ghost_values()
        phi: Scalar field

    Returns:
        Ghost cell value (or cell value if no ghost exists)
    """
    edge_key = (cell_id, tuple(sorted([v0, v1])))

    if edge_key in ghost_values:
        return ghost_values[edge_key]
    else:
        # No ghost for this edge, return cell value
        return float(phi[cell_id])


def create_extended_neighbors(
    grid: BaseGrid,
    boundary_info: Dict
) -> Dict[int, List]:
    """
    Create extended neighbor connectivity including ghost cells.

    For LS/ELS gradient methods, boundary cells get virtual ghost neighbors
    added to their stencil for symmetric reconstruction.

    Args:
        grid: Unstructured grid
        boundary_info: From identify_boundary_cells()

    Returns:
        extended_neighbors: Dict mapping cell_id -> [neighbor_ids + ghost_ids]
                           Ghost IDs are negative: -1, -2, ...
    """
    if grid.cell_neighbors is None:
        grid.compute_neighbors(halo_width=1)

    n_cells = grid.n_cells
    extended_neighbors = {i: list(grid.cell_neighbors[i]) for i in range(n_cells)}

    # Add ghost neighbors to boundary cells
    ghost_id = -1
    ghost_id_to_info = {}

    for edge_key, info in boundary_info['ghost_info'].items():
        cell_id = info['boundary_cell']

        # Add ghost as a virtual neighbor
        extended_neighbors[cell_id].append(ghost_id)
        ghost_id_to_info[ghost_id] = {
            'edge_key': edge_key,
            'position': info['ghost_position'],
            'boundary_cell': cell_id,
        }

        ghost_id -= 1

    return extended_neighbors, ghost_id_to_info
