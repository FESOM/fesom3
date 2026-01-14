"""
Divergence Operator (Finite Volume): ∇·u

Computes the divergence of a 2D vector field using finite volume method
on unstructured grids (triangles, mixed meshes).

Mathematical definition (integral form):
    ∇·u = (1/A) ∮_∂V u·n̂ dS

where A is cell area, ∂V is cell boundary, n̂ is outward normal.

Implementation:
    For each cell, sum fluxes through all edges and divide by cell area.
"""

from typing import Optional, Dict
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jnp = np

from fesomx.core import BaseGrid


def divergence_fv(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Optional[Dict] = None,
    perform_halo_exchange: bool = True,
) -> jnp.ndarray:
    """
    Compute divergence using finite volume method.

    Args:
        u: X-component of velocity at cell centers (n_cells,)
        v: Y-component of velocity at cell centers (n_cells,)
        grid: Unstructured grid object (TriGrid or MixedGrid)
        partition_info: Partitioning metadata (for halo exchange)
        perform_halo_exchange: Whether to exchange halos

    Returns:
        div: Divergence at cell centers (n_cells,)

    Method:
        For each cell:
        1. Loop over edges
        2. Compute flux through each edge: F = u·n̂ * edge_length
        3. Sum fluxes: Σ F
        4. Divide by cell area: div = (1/A) Σ F

    Edge values are interpolated from cell centers.
    """
    # Get cell areas
    areas = grid.get_cell_volumes()
    n_cells = len(areas)

    # Initialize divergence
    div = jnp.zeros(n_cells)

    # Get geometric information
    vertices = grid.vertices
    cells = grid.cells

    # Perform halo exchange if needed
    if perform_halo_exchange and partition_info is not None:
        from fesomx.core.halo_exchange import UnstructuredHaloExchanger
        from fesomx.core.backend import get_backend

        # Get or create backend
        if hasattr(grid, 'backend') and grid.backend is not None:
            backend = grid.backend
        else:
            backend = get_backend('jax_sharding')

        # Create exchanger
        exchanger = UnstructuredHaloExchanger(backend)

        # Exchange u and v components
        # This now creates extended arrays with explicit ghost storage
        u_with_halos, _ = exchanger.exchange(u, partition_info, halo_width=1)
        v_with_halos, _ = exchanger.exchange(v, partition_info, halo_width=1)

        is_extended = partition_info.get('_is_extended', False)
    else:
        u_with_halos = u
        v_with_halos = v
        is_extended = False

    # Dispatch to appropriate computation path
    if is_extended and partition_info is not None:
        # Use partition-local indexing with explicit ghost cells
        div = _divergence_extended(u_with_halos, v_with_halos, grid, partition_info)
    else:
        # Use global indexing (current path for backward compatibility)
        div = _divergence_global(u_with_halos, v_with_halos, grid)

    return div


def _divergence_global(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: BaseGrid,
) -> jnp.ndarray:
    """
    Compute divergence using global indexing.

    This is the original implementation that assumes all cells are accessible
    via their global indices. Works with shared-memory JAX sharding.

    Args:
        u: X-component of velocity at cell centers (n_cells,)
        v: Y-component of velocity at cell centers (n_cells,)
        grid: Unstructured grid object

    Returns:
        div: Divergence at cell centers (n_cells,)
    """
    areas = grid.get_cell_volumes()
    n_cells = len(areas)
    div = jnp.zeros(n_cells)

    vertices = grid.vertices
    cells = grid.cells

    # Compute divergence for each cell
    for cell_id in range(n_cells):
        cell_vertices = cells[cell_id]
        area = areas[cell_id]

        # Get velocity at this cell
        u_cell = float(u[cell_id])
        v_cell = float(v[cell_id])

        # Sum fluxes through edges
        flux_sum = 0.0
        n_vertices = len(cell_vertices)

        for i in range(n_vertices):
            # Edge from vertex i to vertex i+1
            v0 = cell_vertices[i]
            v1 = cell_vertices[(i + 1) % n_vertices]

            # Get vertex coordinates
            x0, y0 = vertices[v0]
            x1, y1 = vertices[v1]

            # Edge vector
            dx_edge = x1 - x0
            dy_edge = y1 - y0
            edge_length = np.sqrt(dx_edge**2 + dy_edge**2)

            # Outward normal (rotate edge vector 90° clockwise for outward)
            nx = dy_edge / edge_length
            ny = -dx_edge / edge_length

            # Find neighbor cell across this edge
            neighbor_id = _find_neighbor_across_edge(grid, cell_id, v0, v1)

            # Interpolate velocity to edge
            if neighbor_id >= 0:  # Interior edge
                u_neighbor = float(u[neighbor_id])
                v_neighbor = float(v[neighbor_id])
                u_edge = 0.5 * (u_cell + u_neighbor)
                v_edge = 0.5 * (v_cell + v_neighbor)
            else:  # Boundary edge
                u_edge = u_cell
                v_edge = v_cell

            # Flux through edge: F = u·n̂ * length
            flux = (u_edge * nx + v_edge * ny) * edge_length
            flux_sum += flux

        # Divergence = flux_sum / area
        div = div.at[cell_id].set(flux_sum / area)

    return div


def _divergence_extended(
    u_extended: jnp.ndarray,
    v_extended: jnp.ndarray,
    grid: BaseGrid,
    partition_info: Dict,
) -> jnp.ndarray:
    """
    Compute divergence using extended arrays with explicit ghost cells.

    Extended arrays have structure: [P0: owned|ghosts | P1: owned|ghosts | ...]
    This function uses partition-local indexing to access owned and ghost data.

    Args:
        u_extended: X-velocity in extended format (sum of all local_array_sizes,)
        v_extended: Y-velocity in extended format (sum of all local_array_sizes,)
        grid: Unstructured grid object
        partition_info: Must contain:
            - 'device_to_cells': {device_id: [owned_cell_ids]}
            - 'global_to_local': {device_id: {global_id: local_idx}}
            - '_extended_offsets': {device_id: offset_in_extended_array}

    Returns:
        div: Divergence at cell centers in global indexing (n_cells,)
    """
    areas = grid.get_cell_volumes()
    n_cells = len(areas)
    div = jnp.zeros(n_cells)

    vertices = grid.vertices
    cells = grid.cells

    device_to_cells = partition_info['device_to_cells']
    global_to_local = partition_info['global_to_local']
    partition_offsets = partition_info['_extended_offsets']

    # Compute divergence for each partition's owned cells
    for device_id, owned_cells in device_to_cells.items():
        offset = partition_offsets[device_id]
        g2l = global_to_local[device_id]

        for global_cell_id in owned_cells:
            # Get local index for this cell
            local_idx = g2l[global_cell_id]

            cell_vertices = cells[global_cell_id]
            area = areas[global_cell_id]

            # Access data using partition offset + local index
            u_cell = float(u_extended[offset + local_idx])
            v_cell = float(v_extended[offset + local_idx])

            # Sum fluxes through edges
            flux_sum = 0.0
            n_vertices = len(cell_vertices)

            for i in range(n_vertices):
                # Edge from vertex i to vertex i+1
                v0 = cell_vertices[i]
                v1 = cell_vertices[(i + 1) % n_vertices]

                # Get vertex coordinates
                x0, y0 = vertices[v0]
                x1, y1 = vertices[v1]

                # Edge vector
                dx_edge = x1 - x0
                dy_edge = y1 - y0
                edge_length = np.sqrt(dx_edge**2 + dy_edge**2)

                # Outward normal
                nx = dy_edge / edge_length
                ny = -dx_edge / edge_length

                # Find neighbor cell across this edge
                neighbor_global_id = _find_neighbor_across_edge(
                    grid, global_cell_id, v0, v1
                )

                # Interpolate velocity to edge
                if neighbor_global_id >= 0:  # Interior edge
                    # Check if neighbor is in this partition's local map (owned or ghost)
                    if neighbor_global_id in g2l:
                        neighbor_local_idx = g2l[neighbor_global_id]
                        u_neighbor = float(u_extended[offset + neighbor_local_idx])
                        v_neighbor = float(v_extended[offset + neighbor_local_idx])
                        u_edge = 0.5 * (u_cell + u_neighbor)
                        v_edge = 0.5 * (v_cell + v_neighbor)
                    else:
                        # Neighbor not in local map: treat as boundary (should not happen with correct halos)
                        u_edge = u_cell
                        v_edge = v_cell
                else:  # Boundary edge
                    u_edge = u_cell
                    v_edge = v_cell

                # Flux through edge: F = u·n̂ * length
                flux = (u_edge * nx + v_edge * ny) * edge_length
                flux_sum += flux

            # Store result in global indexing
            div = div.at[global_cell_id].set(flux_sum / area)

    return div


def _find_neighbor_across_edge(
    grid: BaseGrid,
    cell_id: int,
    v0: int,
    v1: int
) -> int:
    """
    Find neighbor cell that shares edge (v0, v1).

    Args:
        grid: Grid object
        cell_id: Current cell ID
        v0: First vertex of edge
        v1: Second vertex of edge

    Returns:
        neighbor_id: ID of neighbor cell, or -1 if boundary edge
    """
    # Check if grid has precomputed cell neighbors
    if hasattr(grid, 'cell_neighbors') and grid.cell_neighbors is not None:
        # Use precomputed neighbors
        neighbors = grid.cell_neighbors[cell_id]
        if neighbors is None:
            return -1

        # Check each neighbor to see if it shares the edge (v0, v1)
        for neighbor_id in neighbors:
            if neighbor_id < 0:
                continue

            neighbor_vertices = set(grid.cells[neighbor_id])

            # If neighbor contains both v0 and v1, it shares the edge
            if v0 in neighbor_vertices and v1 in neighbor_vertices:
                return neighbor_id

    return -1  # Boundary edge or neighbor not found


# ============================================================================
# Utility Functions
# ============================================================================

def compute_cell_gradients_lsq(
    phi: jnp.ndarray,
    grid: BaseGrid
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute cell-centered gradients using least-squares reconstruction.

    Useful for Green-Gauss gradient operator and higher-order schemes.

    Args:
        phi: Scalar field at cell centers (n_cells,)
        grid: Unstructured grid

    Returns:
        dphidx: X-gradient at cell centers (n_cells,)
        dphidy: Y-gradient at cell centers (n_cells,)

    Method:
        For each cell, solve least-squares problem:
        min Σ (∇φ·Δr - Δφ)²
        where Δr is vector to neighbor, Δφ is value difference.
    """
    n_cells = len(phi)
    dphidx = jnp.zeros(n_cells)
    dphidy = jnp.zeros(n_cells)

    centers = grid.get_cell_centers()

    for cell_id in range(n_cells):
        if not hasattr(grid, 'cell_neighbors') or grid.cell_neighbors is None:
            continue

        neighbors = grid.cell_neighbors[cell_id]
        if neighbors is None or len(neighbors) == 0:
            continue

        # Build least-squares system: A @ grad = b
        # where A[i] = [Δx_i, Δy_i], b[i] = Δφ_i
        A_list = []
        b_list = []

        x_cell, y_cell = centers[cell_id]
        phi_cell = float(phi[cell_id])

        for neighbor_id in neighbors:
            if neighbor_id < 0:
                continue

            x_neighbor, y_neighbor = centers[neighbor_id]
            phi_neighbor = float(phi[neighbor_id])

            dx = x_neighbor - x_cell
            dy = y_neighbor - y_cell
            dphi = phi_neighbor - phi_cell

            A_list.append([dx, dy])
            b_list.append(dphi)

        if len(A_list) < 2:
            continue  # Need at least 2 neighbors

        A = np.array(A_list)
        b = np.array(b_list)

        # Solve least-squares: grad = (A^T A)^{-1} A^T b
        try:
            ATA = A.T @ A
            ATb = A.T @ b
            grad = np.linalg.solve(ATA, ATb)

            dphidx = dphidx.at[cell_id].set(grad[0])
            dphidy = dphidy.at[cell_id].set(grad[1])
        except np.linalg.LinAlgError:
            # Singular matrix, keep zero gradient
            pass

    return dphidx, dphidy
