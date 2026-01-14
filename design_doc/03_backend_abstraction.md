# Backend Abstraction

## Rationale

The backend abstraction layer allows FESOMx to support multiple parallel computing frameworks without changing grid or operator code:

- **Current**: JAX sharding (GPU/TPU native)
- **Future**: JAX MPI (multi-node), NumPy+MPI, PyTorch, Numba

This design future-proofs the codebase as parallel computing frameworks evolve.

## Design Pattern

FESOMx uses an **Abstract Factory + Registry** pattern:

```
┌─────────────────────────────────────────────────────┐
│              Backend Factory & Registry              │
├─────────────────────────────────────────────────────┤
│                                                      │
│  get_backend(name) ───→ Lookup in _BACKENDS         │
│                                ↓                     │
│                    Check _CURRENT_BACKEND            │
│                    (cached singleton)                │
│                                ↓                     │
│              Initialize & cache new backend          │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Abstract Backend Interface

**File**: `fesomx/core/backend.py`

```python
class Backend(ABC):
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
```

### Abstract Methods (Must Implement)

| Method | Purpose | Returns |
|--------|---------|---------|
| `initialize(**kwargs)` | Setup parallel environment | None |
| `array(data, sharding_spec)` | Create backend array from NumPy | Backend array |
| `to_numpy(array)` | Convert backend array to NumPy | np.ndarray |
| `halo_exchange(array, halo_width, neighbors, periodic)` | Ghost cell synchronization | Updated array |
| `get_local_shape(array)` | Local shard shape | Tuple[int, ...] |
| `get_global_shape(array)` | Global array shape | Tuple[int, ...] |
| `barrier()` | Synchronize all devices/processes | None |
| `rank` (property) | Current device/process ID | int |
| `size` (property) | Total devices/processes | int |

### Precision Control

Backends manage floating-point precision (single vs double):

| Precision | Type | Memory | Use Case |
|-----------|------|--------|----------|
| Single | float32 | 4 bytes | GPU-optimized, ML applications |
| Double | float64 | 8 bytes | High-accuracy scientific computing |

```python
# Set precision at backend initialization
backend = get_backend("jax_sharding", precision="float64")

# Or configure JAX globally
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64
```

**Considerations**:
- GPUs are significantly faster with float32
- Ocean models often require float64 for numerical stability
- Backend should allow runtime precision selection

### Optional Methods

| Method | Purpose | Use Case |
|--------|---------|----------|
| `finalize()` | Clean up resources | MPI backends |

## Registry Pattern

### Global State

```python
_BACKENDS = {}           # name → backend class
_CURRENT_BACKEND = None  # Cached instance (singleton)
```

### Registration

```python
def register_backend(name: str, backend_class: type) -> None:
    """Register a backend implementation."""
    _BACKENDS[name] = backend_class
```

### Factory Function

```python
def get_backend(name: str = "jax_sharding", **kwargs) -> Backend:
    """Get or create a backend instance."""
    global _CURRENT_BACKEND

    # Return cached if same name
    if _CURRENT_BACKEND is not None and _CURRENT_BACKEND.name == name:
        return _CURRENT_BACKEND

    # Lookup and create
    if name not in _BACKENDS:
        raise ValueError(f"Unknown backend: {name}")

    backend = _BACKENDS[name]()
    backend.initialize(**kwargs)
    _CURRENT_BACKEND = backend
    return backend
```

### Benefits

- **Singleton**: One instance per backend type
- **Lazy init**: Created only when needed
- **Caching**: Avoids recreation overhead
- **Extensible**: Add backends without modifying core

## JAX Sharding Backend

**File**: `fesomx/core/jax_sharding_backend.py`

### Initialization

```python
class JAXShardingBackend(Backend):
    def __init__(self):
        super().__init__("jax_sharding")
        self.devices = None
        self.mesh = None

    def initialize(self, mesh_shape=None, **kwargs):
        self.devices = jax.devices()

        if mesh_shape is None:
            mesh_shape = (len(self.devices),)

        device_array = np.array(self.devices).reshape(mesh_shape)
        axis_names = ("x",) if len(mesh_shape) == 1 else ("x", "y")
        self.mesh = Mesh(device_array, axis_names)
```

### Array Operations

**Creating Arrays**:
```python
def array(self, data, sharding_spec=None):
    if sharding_spec is None:
        return jnp.array(data)  # Replicated

    sharding = NamedSharding(self.mesh, sharding_spec)
    return jax.device_put(data, sharding)
```

**Usage Examples**:
```python
from jax.sharding import PartitionSpec as P

# Replicated (all devices)
arr = backend.array(data)

# Sharded along x
arr = backend.array(data, P('x', None))

# Sharded in 2D
arr = backend.array(data, P('x', 'y'))
```

**Conversion**:
```python
def to_numpy(self, array):
    return np.array(array)
```

### Device Properties

```python
@property
def rank(self):
    if jax.process_count() > 1:
        return jax.process_index()
    return 0

@property
def size(self):
    return len(self.devices) if self.devices else len(jax.devices())

def barrier(self):
    dummy = jnp.ones(1)
    for device in self.devices:
        jax.device_put(dummy, device).block_until_ready()
```

### Auto-Registration

```python
# At module import
if JAX_AVAILABLE:
    register_backend("jax_sharding", JAXShardingBackend)
```

## Grid Integration

Grids use backends transparently:

```python
class BaseGrid(ABC):
    def __init__(self, name, backend=None, ...):
        self.backend = backend if backend else get_backend()
```

### Field Management

```python
def add_field(self, name, data, sharding_spec=None):
    backend_array = self.backend.array(data, sharding_spec)
    self.fields[name] = {'data': backend_array, ...}

def get_field_numpy(self, name):
    return self.backend.to_numpy(self.fields[name]['data'])
```

### Halo Exchange

```python
def halo_exchange(self, field_name, halo_width=1):
    data = self.fields[field_name]['data']
    updated = self.backend.halo_exchange(data, halo_width,
                                          self.neighbors, self.periodic)
    self.fields[field_name]['data'] = updated
```

## Adding New Backends

To add a new backend (e.g., JAX MPI):

```python
# fesomx/core/jax_mpi_backend.py

class JAXMPIBackend(Backend):
    def __init__(self):
        super().__init__("jax_mpi")

    def initialize(self, **kwargs):
        # Initialize MPI
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        # ...

    def array(self, data, sharding_spec=None):
        # MPI-aware array creation
        pass

    def halo_exchange(self, array, halo_width, neighbors, periodic):
        # Use MPI send/recv
        pass

    # ... implement remaining methods

# Register
register_backend("jax_mpi", JAXMPIBackend)
```

Usage:
```python
backend = get_backend("jax_mpi", comm=MPI.COMM_WORLD)
grid = QuadGrid("test", backend=backend)
```

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of Concerns** | Backend logic decoupled from grids |
| **Lazy Initialization** | Created only when needed |
| **Singleton Caching** | One instance per backend type |
| **Type Safety** | Clear type hints on all methods |
| **Extensibility** | Registry allows adding backends |
| **Backward Compatibility** | Default backend if none specified |

## Planned Backends

| Backend | Status | Use Case |
|---------|--------|----------|
| `jax_sharding` | Implemented | Single-node multi-GPU |
| `jax_mpi` | Planned | Multi-node clusters |
| `numpy_mpi` | Planned | CPU-only MPI |
| `pytorch` | Future | PyTorch ecosystem |
| `numba` | Future | CPU optimization |

## Related Documents

- [00_goals_and_architecture_overview.md](00_goals_and_architecture_overview.md) - Technology stack
- [04_halo_exchange.md](04_halo_exchange.md) - Backend's halo_exchange implementation
