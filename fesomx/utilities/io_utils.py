"""I/O utilities for saving and loading grids and fields."""

import pickle
from pathlib import Path
from typing import Any, Optional, Dict
import numpy as np

try:
    import xarray as xr
    XARRAY_AVAILABLE = True
except ImportError:
    XARRAY_AVAILABLE = False


def save_grid(grid: Any, filename: str, format: str = 'pickle') -> None:
    """Save grid to file.

    Args:
        grid: Grid object to save
        filename: Output filename
        format: File format ('pickle', 'npz')
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(grid, f)
        print(f"Grid saved to {filepath}")

    elif format == 'npz':
        # Save grid data as numpy arrays
        data = {
            'vertices': grid.vertices,
            'name': grid.name,
            'stagger_type': grid.stagger_type.value,
        }

        if hasattr(grid, 'cells'):
            if isinstance(grid.cells, list):
                # Mixed grid - save with pickle
                print("Warning: Mixed grids should use pickle format")
                data['cells'] = np.array(grid.cells, dtype=object)
            else:
                data['cells'] = grid.cells

        if hasattr(grid, 'shape'):
            data['shape'] = grid.shape

        np.savez(filepath, **data)
        print(f"Grid saved to {filepath}")

    else:
        raise ValueError(f"Unknown format: {format}")


def load_grid(filename: str, format: str = 'pickle') -> Any:
    """Load grid from file.

    Args:
        filename: Input filename
        format: File format ('pickle', 'npz')

    Returns:
        Grid object
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Grid file not found: {filepath}")

    if format == 'pickle':
        with open(filepath, 'rb') as f:
            grid = pickle.load(f)
        print(f"Grid loaded from {filepath}")
        return grid

    elif format == 'npz':
        data = np.load(filepath, allow_pickle=True)
        print(f"Grid data loaded from {filepath}")
        return dict(data)

    else:
        raise ValueError(f"Unknown format: {format}")


def save_field(
    field_data: np.ndarray,
    filename: str,
    metadata: Optional[Dict] = None,
    format: str = 'npy'
) -> None:
    """Save field data to file.

    Args:
        field_data: Field array
        filename: Output filename
        metadata: Optional metadata dictionary
        format: File format ('npy', 'npz', 'nc')
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == 'npy':
        np.save(filepath, field_data)
        print(f"Field saved to {filepath}")

    elif format == 'npz':
        if metadata:
            np.savez(filepath, data=field_data, **metadata)
        else:
            np.savez(filepath, data=field_data)
        print(f"Field saved to {filepath}")

    elif format == 'nc':
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for NetCDF format")

        # Create xarray DataArray
        da = xr.DataArray(
            field_data,
            attrs=metadata if metadata else {}
        )
        da.to_netcdf(filepath)
        print(f"Field saved to {filepath}")

    else:
        raise ValueError(f"Unknown format: {format}")


def load_field(filename: str, format: str = 'npy') -> np.ndarray | Dict:
    """Load field data from file.

    Args:
        filename: Input filename
        format: File format ('npy', 'npz', 'nc')

    Returns:
        Field array (or dict with data and metadata for npz)
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Field file not found: {filepath}")

    if format == 'npy':
        data = np.load(filepath)
        print(f"Field loaded from {filepath}")
        return data

    elif format == 'npz':
        data = np.load(filepath)
        print(f"Field loaded from {filepath}")
        return dict(data)

    elif format == 'nc':
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for NetCDF format")

        da = xr.open_dataarray(filepath)
        print(f"Field loaded from {filepath}")
        return da.values

    else:
        raise ValueError(f"Unknown format: {format}")


def save_results(
    filename: str,
    fields: Dict[str, np.ndarray],
    grid_info: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """Save multiple fields and grid information to a single file.

    Args:
        filename: Output filename (should end in .npz or .nc)
        fields: Dictionary of field_name -> field_data
        grid_info: Optional grid information
        metadata: Optional metadata
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.suffix == '.npz':
        save_dict = {**fields}
        if grid_info:
            save_dict['grid_info'] = grid_info
        if metadata:
            save_dict['metadata'] = metadata

        np.savez(filepath, **save_dict)
        print(f"Results saved to {filepath}")

    elif filepath.suffix == '.nc':
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for NetCDF format")

        # Create xarray Dataset
        data_vars = {}
        for name, data in fields.items():
            data_vars[name] = xr.DataArray(data)

        ds = xr.Dataset(data_vars, attrs=metadata if metadata else {})
        ds.to_netcdf(filepath)
        print(f"Results saved to {filepath}")

    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")


def load_results(filename: str) -> Dict:
    """Load results from file.

    Args:
        filename: Input filename

    Returns:
        Dictionary with loaded data
    """
    filepath = Path(filename)

    if not filepath.exists():
        raise FileNotFoundError(f"Results file not found: {filepath}")

    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        print(f"Results loaded from {filepath}")
        return dict(data)

    elif filepath.suffix == '.nc':
        if not XARRAY_AVAILABLE:
            raise ImportError("xarray required for NetCDF format")

        ds = xr.open_dataset(filepath)
        print(f"Results loaded from {filepath}")
        return ds

    else:
        raise ValueError(f"Unsupported file extension: {filepath.suffix}")
