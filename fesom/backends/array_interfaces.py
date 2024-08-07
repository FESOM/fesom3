import numpy as np
import jax.numpy as jnp
def array_factory(data, backend="numpy"):
    if backend == "numpy":
        return np.array(data).view(JaxStyleNumpyArray)
    if backend == "numpy-immutable":
        return np.array(data).view(ImmutableJaxStyleNumpyArray)
    elif backend == "jax":
        return jnp.array(data)
    else:
        raise ValueError(f"Unsupported backend: {backend}")


class NumpyJaxIndexer:
    __slots__ = ("array",)

    def __init__(self, nparray):
        self.array = nparray

    def __getitem__(self, index):
        return NumpyJaxIndexerSet(self.array, index)

    def __repr__(self):
        return f"NumpyJaxIndexer({self.array!r})"


class NumpyJaxIndexerSet:
    __slots__ = ("array", "index")

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def add(self, value):
        self.array[self.index] += value

    def multiply(self, value):
        self.array[self.index] *= value

    def divide(self, value):
        self.array[self.index] /= value

    def power(self, value):
        self.array[self.index] **= value

    def min(self, value):
        self.array[self.index] = np.minimum(self.array[self.index], value)

    def max(self, value):
        self.array[self.index] = np.maximum(self.array[self.index], value)

    def apply(self, ufunc):
        self.array[self.index] = ufunc(self.array[self.index])

    def get(self):
        return self.array[self.index]

    def __repr__(self):
        return f"NumpyJaxIndexerSet({self.array!r}, {self.index!r})"

    def set(self, value):
        self.array[self.index] = value


class JaxStyleNumpyArray(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    @property
    def at(self):
        return NumpyJaxIndexer(self)


class ImmutableNumpyJaxIndexer:
    __slots__ = ("array",)

    def __init__(self, array):
        self.array = array

    def __getitem__(self, index):
        return ImmutableNumpyJaxIndexerSet(self.array, index)

    def __repr__(self):
        return f"ImmutableNumpyJaxIndexer({self.array!r})"

class ImmutableNumpyJaxIndexerSet:
    __slots__ = ("array", "index")

    def __init__(self, array, index):
        self.array = array
        self.index = index

    def add(self, value):
        return np.add(self.array, value, where=self._index_mask())

    def multiply(self, value):
        return np.multiply(self.array, value, where=self._index_mask())

    def divide(self, value):
        return np.divide(self.array, value, where=self._index_mask())

    def power(self, value):
        return np.power(self.array, value, where=self._index_mask())

    def min(self, value):
        return np.minimum(self.array, np.full_like(self.array, value, dtype=self.array.dtype), where=self._index_mask())

    def max(self, value):
        return np.maximum(self.array, np.full_like(self.array, value, dtype=self.array.dtype), where=self._index_mask())

    def apply(self, ufunc):
        return ufunc(self.array, where=self._index_mask())

    def set(self, value):
        return np.where(self._index_mask(), np.full_like(self.array, value, dtype=self.array.dtype), self.array)

    def get(self):
        return self.array[self.index]

    def _index_mask(self):
        # Create a mask that is True at self.index and False everywhere else
        mask = np.zeros_like(self.array, dtype=bool)
        mask[self.index] = True
        return mask

    def __repr__(self):
        return f"ImmutableNumpyJaxIndexerSet({self.array!r}, {self.index!r})"

class ImmutableJaxStyleNumpyArray(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.info = info
        return obj

    @property
    def at(self):
        return ImmutableNumpyJaxIndexer(self)
