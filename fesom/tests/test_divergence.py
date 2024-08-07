from fesom.operators import divergence
import jax.numpy as jnp


def test_divergence(cell_centers):
    assert 1


def test_hello():
    assert divergence.hello() == 'hello'


def test_calculate_divergence(uv, mesh):
    u, v = uv
    areas, normals, edge_lengths = mesh
    div = divergence.calculate_divergence_jit(u, v, areas, normals, edge_lengths)
    assert div.shape == (100,)
    assert div.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(div))
    #assert jnp.all(div >= 0)
    #assert jnp.all(jnp.abs(div) <= 1e-6)
