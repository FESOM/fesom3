"""
Base Classes for Operators

Defines abstract interfaces for numerical operators on structured
and unstructured grids.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import jax.numpy as jnp

from fesomx.core import BaseGrid


class OperatorInterface(ABC):
    """
    Abstract interface for numerical differential operators.

    Implementations provide grid-specific methods (finite difference,
    finite volume, finite element, etc.)
    """

    @abstractmethod
    def compute_divergence(
        self,
        u: jnp.ndarray,
        v: jnp.ndarray,
        grid: BaseGrid,
        partition_info: Optional[Dict] = None,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute divergence: ∇·(u,v) = ∂u/∂x + ∂v/∂y

        Args:
            u: X-component of vector field
            v: Y-component of vector field
            grid: Grid object
            partition_info: Partitioning metadata (for halo exchange)
            **kwargs: Implementation-specific options

        Returns:
            div: Divergence field
        """
        pass

    @abstractmethod
    def compute_gradient(
        self,
        phi: jnp.ndarray,
        grid: BaseGrid,
        partition_info: Optional[Dict] = None,
        **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Compute gradient: ∇φ = (∂φ/∂x, ∂φ/∂y)

        Args:
            phi: Scalar field
            grid: Grid object
            partition_info: Partitioning metadata
            **kwargs: Implementation-specific options

        Returns:
            dphidx: X-component of gradient
            dphidy: Y-component of gradient
        """
        pass

    @abstractmethod
    def compute_laplacian(
        self,
        phi: jnp.ndarray,
        grid: BaseGrid,
        partition_info: Optional[Dict] = None,
        **kwargs
    ) -> jnp.ndarray:
        """
        Compute Laplacian: ∇²φ = ∂²φ/∂x² + ∂²φ/∂y²

        Args:
            phi: Scalar field
            grid: Grid object
            partition_info: Partitioning metadata
            **kwargs: Implementation-specific options

        Returns:
            lapl: Laplacian field
        """
        pass


class StructuredOperator(OperatorInterface):
    """
    Finite difference operators for structured grids.

    Uses regular stencils on 2D arrays with uniform or non-uniform spacing.
    Supports Arakawa A/B/C grid staggering.
    """

    def __init__(self):
        self.name = "Finite Difference"

    def compute_divergence(self, u, v, grid, partition_info=None, **kwargs):
        """Compute divergence using finite differences."""
        from operators.structured.divergence_fd import divergence_fd
        return divergence_fd(u, v, grid, partition_info, **kwargs)

    def compute_gradient(self, phi, grid, partition_info=None, **kwargs):
        """Compute gradient using finite differences."""
        from operators.structured.gradient_fd import gradient_fd
        return gradient_fd(phi, grid, partition_info, **kwargs)

    def compute_laplacian(self, phi, grid, partition_info=None, **kwargs):
        """Compute Laplacian using finite differences."""
        from operators.structured.laplacian_fd import laplacian_fd
        return laplacian_fd(phi, grid, partition_info, **kwargs)


class UnstructuredOperator(OperatorInterface):
    """
    Finite volume operators for unstructured grids.

    Uses cell-centered storage with face-based fluxes.
    Works on arbitrary polygonal/polyhedral cells.
    """

    def __init__(self):
        self.name = "Finite Volume"

    def compute_divergence(self, u, v, grid, partition_info=None, **kwargs):
        """Compute divergence using finite volume method."""
        from operators.unstructured.divergence_fv import divergence_fv
        return divergence_fv(u, v, grid, partition_info, **kwargs)

    def compute_gradient(self, phi, grid, partition_info=None, **kwargs):
        """Compute gradient using Green-Gauss method."""
        from operators.unstructured.gradient_fv import gradient_fv
        return gradient_fv(phi, grid, partition_info, **kwargs)

    def compute_laplacian(self, phi, grid, partition_info=None, **kwargs):
        """Compute Laplacian using finite volume method."""
        from operators.unstructured.laplacian_fv import laplacian_fv
        return laplacian_fv(phi, grid, partition_info, **kwargs)


def get_operator(grid: BaseGrid) -> OperatorInterface:
    """
    Factory function to get appropriate operator for grid type.

    Args:
        grid: Grid object

    Returns:
        operator: StructuredOperator or UnstructuredOperator
    """
    if grid.is_structured():
        return StructuredOperator()
    else:
        return UnstructuredOperator()
