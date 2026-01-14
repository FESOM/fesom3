# Operators and Their Dispatch

## Overview

FESOMx provides mathematical operators (divergence, gradient, laplacian) with a unified API that automatically dispatches to the appropriate implementation based on grid type:

- **Structured grids** (QuadGrid) → Finite Difference (FD) methods
- **Unstructured grids** (TriGrid, MixedGrid) → Finite Volume (FV) methods

## Dispatch Mechanism

**File**: `fesomx/operators/__init__.py`

The dispatch is simple and based on `grid.is_structured()`:

```python
def divergence(u, v, grid, partition_info=None, **kwargs):
    if grid.is_structured():
        return divergence_fd(u, v, grid, partition_info, **kwargs)
    else:
        return divergence_fv(u, v, grid, partition_info, **kwargs)

def gradient(phi, grid, partition_info=None, **kwargs):
    if grid.is_structured():
        return gradient_fd(phi, grid, partition_info, **kwargs)
    else:
        return gradient_fv(phi, grid, partition_info, **kwargs)

def laplacian(phi, grid, partition_info=None, **kwargs):
    if grid.is_structured():
        return laplacian_fd(phi, grid, partition_info, **kwargs)
    else:
        return laplacian_fv(phi, grid, partition_info, **kwargs)
```

### Design Benefits

- **Unified API**: Same function call for all grid types
- **Automatic selection**: User doesn't need to know implementation details
- **Flexible kwargs**: Implementation-specific options pass through
- **Optional partition_info**: Enables halo exchange for distributed grids

### Halo Width Considerations

Operators may require different halo widths depending on the order of accuracy:

| Order | Required Halo Width | Stencil Reach |
|-------|---------------------|---------------|
| 2nd order | 1 | ±1 cell |
| 4th order | 2 | ±2 cells |
| 6th order | 3 | ±3 cells |

When using higher-order operators on distributed grids, ensure sufficient halo width is configured before calling the operator. The halo exchange must be performed with a width that matches or exceeds the operator's stencil requirements.

```python
# Higher-order operator requires wider halo
grid.halo_exchange("field", halo_width=2)  # For 4th order
result = laplacian(field, grid, order=4)
```

## Available Operators

| Operator | Mathematical Form | Description |
|----------|-------------------|-------------|
| `divergence(u, v, grid)` | ∇·**u** = ∂u/∂x + ∂v/∂y | Vector field divergence |
| `gradient(φ, grid)` | ∇φ = (∂φ/∂x, ∂φ/∂y) | Scalar field gradient |
| `laplacian(φ, grid)` | ∇²φ = ∂²φ/∂x² + ∂²φ/∂y² | Scalar Laplacian |

## Structured Implementations (Finite Difference)

**Location**: `fesomx/operators/structured/`

### Stencil Orders

| Order | Stencil Size | Halo Width | Truncation Error |
|-------|--------------|------------|------------------|
| 2nd order | 5-point | 1 | O(Δx², Δy²) |
| 4th order | 9-point | 2 | O(Δx⁴, Δy⁴) |

### 2nd Order Stencils (Default)

**Gradient** (centered difference):
```
∂φ/∂x ≈ (φ[i+1,j] - φ[i-1,j]) / (2Δx)
∂φ/∂y ≈ (φ[i,j+1] - φ[i,j-1]) / (2Δy)
```

**Laplacian** (5-point stencil):
```
∇²φ ≈ (φ[i+1,j] - 2φ[i,j] + φ[i-1,j])/Δx²
     + (φ[i,j+1] - 2φ[i,j] + φ[i,j-1])/Δy²
```

### 4th Order Stencils

**Gradient**:
```
∂φ/∂x ≈ (-φ[i+2] + 8φ[i+1] - 8φ[i-1] + φ[i-2]) / (12Δx)
```

**Laplacian**:
```
∂²φ/∂x² ≈ (-φ[i+2] + 16φ[i+1] - 30φ[i] + 16φ[i-1] - φ[i-2]) / (12Δx²)
```

### Arakawa Staggering Support

| Stagger | Divergence | Gradient | Notes |
|---------|------------|----------|-------|
| A-grid | Centered at centers | At centers | Simple but checkerboard issues |
| B-grid | Average corners to center | At corners | FESOM2-like |
| C-grid | Natural form | Natural form | Optimal for ocean models |

**C-grid Divergence** (natural form - no interpolation needed):
```python
div[i,j] = (u[i+1,j] - u[i,j])/dx + (v[i,j+1] - v[i,j])/dy
```

**C-grid Gradient** (natural form):
```python
∂φ/∂x at x-faces: (φ[i+1,j] - φ[i,j]) / dx
∂φ/∂y at y-faces: (φ[i,j+1] - φ[i,j]) / dy
```

### Periodic Boundary Conditions

Implemented via efficient `jnp.roll()`:

```python
if periodic_x:
    dphidx = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2*dx)
```

## Unstructured Implementations (Finite Volume)

**Location**: `fesomx/operators/unstructured/`

### Gradient Methods

| Order | Method | Stencil | Accuracy |
|-------|--------|---------|----------|
| 1 | Green-Gauss (GG) | Direct neighbors | ~1st order on irregular grids |
| 2 | Least-Squares (LS) | Direct neighbors | 2nd order |
| 4 | Extended LS (ELS) | Neighbors-of-neighbors | Higher order |

**Green-Gauss (order=1)**:
```
∇φ = (1/A) Σ_edges φ_edge · n̂ · edge_length

where φ_edge = (φ_cell + φ_neighbor) / 2
```

**Least-Squares (order=2)**:

Minimizes: Σⱼ wⱼ |φⱼ - φᵢ - ∇φ·(rⱼ - rᵢ)|²

Leads to 2×2 normal equations with inverse-distance weighting.

**Extended Least-Squares (order=4)**:

Same formulation but with extended stencil including neighbors-of-neighbors.

### Divergence (Edge-Based Flux)

For each cell:
1. Loop over edges
2. Compute outward normal: n̂ = (dy, -dx) / edge_length
3. Interpolate velocity to edge: **u**_edge = 0.5 × (**u**_cell + **u**_neighbor)
4. Compute flux: F = **u**_edge · n̂ × edge_length
5. Sum fluxes: div = Σ F / cell_area

### Laplacian

Implemented as composition: ∇²φ = ∇·(∇φ)

1. Compute gradient using selected method
2. Apply divergence to gradient components

### Boundary Treatment

**File**: `fesomx/operators/unstructured/boundary_treatment.py`

| Option | Method | Use Case |
|--------|--------|----------|
| `'none'` | Asymmetric stencils | Default, faster |
| `'extrapolate'` | Ghost cell extrapolation | Better accuracy at boundaries |

**Linear Extrapolation** (for order ≤ 2):
```
φ_ghost = 2·φ_boundary - φ_interior
```

**Quadratic Extrapolation** (for order = 4):
```
φ_ghost = 3·φ₀ - 3·φ₁ + φ₂
```

## Directory Structure

```
fesomx/operators/
├── __init__.py              # Unified API with dispatch
├── base.py                  # Abstract operator interface
├── divergence.py            # Divergence helpers and utilities
├── gradient.py              # Gradient helpers and utilities
├── laplacian.py             # Laplacian helpers and utilities
├── structured/              # Finite Difference implementations
│   ├── __init__.py
│   ├── divergence_fd.py     # FD divergence (A/B/C grids)
│   ├── gradient_fd.py       # FD gradient (2nd/4th order)
│   └── laplacian_fd.py      # FD Laplacian (2nd/4th order)
└── unstructured/            # Finite Volume implementations
    ├── __init__.py
    ├── divergence_fv.py     # Edge-based flux divergence
    ├── gradient_fv.py       # GG, LS, ELS methods
    ├── laplacian_fv.py      # ∇·(∇φ) composition
    └── boundary_treatment.py # Ghost cell extrapolation
```

## Usage Examples

### Basic Usage

```python
from fesomx import divergence, gradient, laplacian

# Automatic dispatch based on grid type
div = divergence(u, v, grid)
grad_x, grad_y = gradient(phi, grid)
lapl = laplacian(phi, grid)
```

### Higher-Order Methods

```python
# 4th order on structured grids
lapl = laplacian(phi, quad_grid, order=4)

# Least-squares on unstructured grids
grad_x, grad_y = gradient(phi, tri_grid, order=2)
```

### Distributed Computation

```python
# With partition info for halo exchange
div = divergence(u, v, grid, partition_info=partition_metadata)
```

### Boundary Treatment

```python
# Better accuracy at domain boundaries
grad_x, grad_y = gradient(
    phi, tri_grid,
    order=2,
    boundary_treatment='extrapolate'
)
```

### Periodic Boundaries

```python
# Periodic in both directions
grad_x, grad_y = gradient(phi, grid, periodic=(True, True))
```

## JAX Integration

- All implementations use `jax.numpy` arrays
- JIT-compiled versions for performance
- Static arguments marked for jitting
- Graceful fallback to NumPy if JAX unavailable

## Composing Complex Operators

The base operators (divergence, gradient, laplacian) are designed as building blocks for constructing more complex operators needed in ocean/climate models.

### Example: Advection Operator

Advection of a scalar φ by velocity field **u**:

```python
def advection(phi, u, v, grid, **kwargs):
    """Compute advection: u·∇φ"""
    grad_x, grad_y = gradient(phi, grid, **kwargs)
    return u * grad_x + v * grad_y
```

### Example: Diffusion with Variable Coefficient

```python
def variable_diffusion(phi, kappa, grid, **kwargs):
    """Compute ∇·(κ∇φ) for variable diffusivity κ"""
    grad_x, grad_y = gradient(phi, grid, **kwargs)
    flux_x = kappa * grad_x
    flux_y = kappa * grad_y
    return divergence(flux_x, flux_y, grid, **kwargs)
```

### Example: Momentum Equations

```python
def pressure_gradient_force(pressure, grid, rho=1.0, **kwargs):
    """Compute -∇p/ρ"""
    grad_x, grad_y = gradient(pressure, grid, **kwargs)
    return -grad_x / rho, -grad_y / rho
```

### Composition Benefits

- **Reusability**: Base operators tested once, reused everywhere
- **Consistency**: Same numerical scheme across all composed operators
- **Flexibility**: Easy to swap implementations (FD ↔ FV) via grid type
- **Maintainability**: Bug fixes in base operators propagate to all compositions

## Testing Requirements

Beyond simple correctness tests, operators should be validated for physical properties:

### Conservation Testing

Operators must be tested for **conservation on dual cells**:
- Divergence theorem: ∫∫ ∇·**u** dA = ∮ **u**·**n** dl
- Integral of divergence over domain should equal net flux through boundaries
- For periodic domains with divergence-free fields, total divergence should be zero

### Consistency Testing

- Gradient and divergence should be **adjoint operators** (up to sign and boundary terms)
- Laplacian = divergence of gradient should hold numerically
- Operators should commute appropriately with grid transformations

### Example Conservation Test

```python
def test_divergence_conservation(u, v, grid):
    """Verify divergence theorem on closed domain."""
    div = divergence(u, v, grid)

    # Interior integral
    interior_sum = (div * grid.cell_areas).sum()

    # Boundary flux (should match for conservation)
    boundary_flux = compute_boundary_flux(u, v, grid)

    assert jnp.allclose(interior_sum, boundary_flux, rtol=1e-10)
```

## Summary Table

| Aspect | Structured (FD) | Unstructured (FV) |
|--------|-----------------|-------------------|
| Grid Types | QuadGrid | TriGrid, MixedGrid |
| Divergence | Centered/one-sided FD | Edge-based flux sum |
| Gradient | 2nd/4th order FD | GG (1st), LS (2nd), ELS (4th) |
| Laplacian | 5-point/9-point stencil | ∇·(∇φ) composition |
| Staggering | A, B, C Arakawa | N/A (cell-centered) |
| Boundary | Halos (implicit) | Ghost cells (explicit) |
| Periodic | `jnp.roll()` | Precomputed neighbors |

## Related Documents

- [01_grids_and_staggering.md](01_grids_and_staggering.md) - Grid types that determine dispatch
- [04_halo_exchange.md](04_halo_exchange.md) - How halo exchange supports operators
