import pytest
import numpy as np
import jax.numpy as jnp
from fesom.backends.array_interfaces import array_factory, JaxStyleNumpyArray

@pytest.mark.parametrize("backend", ["numpy", "numpy-immutable", "jax"])
@pytest.mark.parametrize("data, slice_idx, set_value", [
    ([1, 2, 3, 4, 5], slice(1, 4), 10),
    ([1.0, 2.0, 3.0, 4.0, 5.0], slice(0, 3), 20.0),
    ([1, 2, 3, 4, 5], slice(2, 5), 30),
    ([1.0, 2.0, 3.0, 4.0, 5.0], slice(1, 4), 40.0)
])

def test_array_slicing_and_setting(backend, data, slice_idx, set_value):
    arr = array_factory(data, backend=backend)
    if backend == "jax":
        arr = arr.at[slice_idx].set(set_value)
        assert jnp.all(arr[slice_idx] == set_value)
    elif backend == "numpy-immutable":
        arr = arr.at[slice_idx].set(set_value)
        assert np.all(arr[slice_idx] == set_value)
    else:
        arr.at[slice_idx].set(set_value)
        assert np.all(arr[slice_idx] == set_value)