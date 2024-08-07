import pytest
import jax
import jax.numpy as jnp
from jax import jit
# Create a mesh with random cell centers
n_cells = 100

@pytest.fixture(scope='module')
def cell_centers():
    n_cells = 100
    return jax.random.uniform(jax.random.PRNGKey(0), (n_cells, 2))

# Assume properties of the mesh are given
@pytest.fixture(scope='module')
def mesh():
    n_cells = 100
    areas = jax.random.uniform(jax.random.PRNGKey(1), (n_cells,))  # Random areas
    normals = jax.random.normal(jax.random.PRNGKey(2), (n_cells, 3, 2))  # Random normals per edge
    edge_lengths = jax.random.uniform(jax.random.PRNGKey(3), (n_cells, 3))  # Random edge lengths
    return (areas, normals, edge_lengths)

# Define the vector field
def vector_field(positions):
    u = positions[:, 0]
    v = positions[:, 1]
    return u, v

# Just-in-time compilation of vector field function
vector_field_jit = jit(vector_field)
@pytest.fixture(scope='module')
def uv(cell_centers):
    # Evaluate the vector field at cell centers
    u, v = vector_field_jit(cell_centers)
    return u, v
