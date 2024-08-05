import jax
import jax.numpy as jnp
from jax import jit

# Function to calculate divergence
def calculate_divergence(u, v, areas, normals, edge_lengths):
    velocities = jnp.stack([u, v], axis=-1)[:, jnp.newaxis, :]  # Shape: (n_cells, 1, 2)
    fluxes = jnp.sum(normals * velocities * edge_lengths[:, :, jnp.newaxis], axis=(1, 2))
    divergence = fluxes / areas
    return divergence

# JIT compile the divergence calculation
calculate_divergence_jit = jit(calculate_divergence)