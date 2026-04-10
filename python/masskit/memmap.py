"""
Memory-mapped array support for large LC-MS datasets.

Provides numpy memmap-backed storage for large feature matrices,
intensity data, and spectral libraries that exceed available RAM.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np


@dataclass
class MemmapConfig:
    """Configuration for memory-mapped storage."""
    base_dir: str = ".masskit_cache"
    dtype: str = "float64"
    mode: str = "r+"  # 'r+', 'w+', 'c' (copy-on-write)
    chunk_size: int = 10000  # rows per chunk for streaming writes


class MemmapMatrix:
    """
    Memory-mapped feature intensity matrix.

    Wraps numpy memmap for transparent disk-backed array operations.
    Useful for consensus maps with thousands of features x hundreds of samples.

    Example:
        >>> mm = MemmapMatrix.create("features.dat", n_features=50000, n_samples=100)
        >>> mm[0:100, :] = intensity_block  # Write a block
        >>> subset = mm[500:600, :]  # Read a block (lazy)
        >>> mm.close()
    """

    def __init__(self, filepath: str, shape: Tuple[int, ...],
                 dtype: str = "float64", mode: str = "r+"):
        self._filepath = filepath
        self._shape = shape
        self._dtype = dtype
        self._mode = mode
        self._data: Optional[np.memmap] = None
        self._metadata_path = filepath + ".meta.json"
        self._open()

    def _open(self):
        self._data = np.memmap(
            self._filepath, dtype=self._dtype,
            mode=self._mode, shape=self._shape,
        )

    @classmethod
    def create(
        cls,
        filepath: str,
        n_rows: int,
        n_cols: int,
        dtype: str = "float64",
        metadata: Optional[Dict] = None,
    ) -> "MemmapMatrix":
        """Create a new memory-mapped matrix file."""
        shape = (n_rows, n_cols)
        # Create file
        data = np.memmap(filepath, dtype=dtype, mode="w+", shape=shape)
        data.flush()
        del data

        # Save metadata
        meta = {
            "shape": list(shape),
            "dtype": dtype,
            "n_rows": n_rows,
            "n_cols": n_cols,
        }
        if metadata:
            meta.update(metadata)
        meta_path = filepath + ".meta.json"
        Path(meta_path).write_text(json.dumps(meta, indent=2))

        return cls(filepath, shape, dtype, mode="r+")

    @classmethod
    def open(cls, filepath: str, mode: str = "r+") -> "MemmapMatrix":
        """Open an existing memory-mapped matrix."""
        meta_path = filepath + ".meta.json"
        meta = json.loads(Path(meta_path).read_text())
        shape = tuple(meta["shape"])
        dtype = meta.get("dtype", "float64")
        return cls(filepath, shape, dtype, mode)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def n_rows(self) -> int:
        return self._shape[0]

    @property
    def n_cols(self) -> int:
        return self._shape[1]

    @property
    def data(self) -> np.memmap:
        if self._data is None:
            self._open()
        return self._data

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def flush(self):
        """Flush changes to disk."""
        if self._data is not None:
            self._data.flush()

    def close(self):
        """Close the memory-mapped file."""
        if self._data is not None:
            self._data.flush()
            del self._data
            self._data = None

    def to_array(self) -> np.ndarray:
        """Load entire matrix into memory. Use with caution for large matrices."""
        return np.array(self.data)

    def column_means(self) -> np.ndarray:
        """Compute column means without loading full matrix."""
        means = np.zeros(self.n_cols)
        chunk = min(self.n_rows, 1000)
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            means += np.sum(self.data[start:end], axis=0)
        means /= self.n_rows
        return means

    def row_sums(self) -> np.ndarray:
        """Compute row sums without loading full matrix."""
        sums = np.zeros(self.n_rows)
        chunk = min(self.n_rows, 1000)
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            sums[start:end] = np.sum(self.data[start:end], axis=1)
        return sums

    def normalize_columns(self, target: float = 1e6) -> None:
        """Normalize columns to a target sum (in-place)."""
        chunk = 1000
        col_sums = np.zeros(self.n_cols)
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            col_sums += np.sum(self.data[start:end], axis=0)
        col_sums[col_sums == 0] = 1.0
        factors = target / col_sums
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            self.data[start:end] *= factors
        self.flush()

    def apply_log2(self, pseudocount: float = 1.0) -> None:
        """Apply log2 transformation in-place."""
        chunk = 1000
        for start in range(0, self.n_rows, chunk):
            end = min(start + chunk, self.n_rows)
            self.data[start:end] = np.log2(self.data[start:end] + pseudocount)
        self.flush()

    def __repr__(self) -> str:
        return (f"MemmapMatrix('{self._filepath}', shape={self._shape}, "
                f"dtype='{self._dtype}')")

    def __del__(self):
        self.close()


class MemmapSpectraStore:
    """
    Memory-mapped storage for spectral data (m/z + intensity arrays).

    Stores spectra in a flat memory-mapped format with an index for
    random access. Efficient for large spectral libraries.

    Example:
        >>> store = MemmapSpectraStore.create("spectra.dat", max_peaks=500)
        >>> store.add_spectrum(mz_array, intensity_array)
        >>> mz, ints = store.get_spectrum(0)
    """

    def __init__(self, filepath: str, max_peaks: int, mode: str = "r+"):
        self._filepath = filepath
        self._max_peaks = max_peaks
        self._mode = mode
        self._index_path = filepath + ".index.json"
        self._mz_data: Optional[np.memmap] = None
        self._int_data: Optional[np.memmap] = None
        self._n_spectra = 0
        self._capacity = 0
        self._lengths: List[int] = []

    @classmethod
    def create(
        cls,
        filepath: str,
        max_peaks: int = 500,
        initial_capacity: int = 10000,
    ) -> "MemmapSpectraStore":
        """Create a new spectra store."""
        mz_path = filepath + ".mz.dat"
        int_path = filepath + ".int.dat"

        shape = (initial_capacity, max_peaks)
        mz_data = np.memmap(mz_path, dtype="float64", mode="w+", shape=shape)
        int_data = np.memmap(int_path, dtype="float32", mode="w+", shape=shape)
        mz_data.flush()
        int_data.flush()
        del mz_data, int_data

        store = cls(filepath, max_peaks, mode="r+")
        store._capacity = initial_capacity
        store._n_spectra = 0
        store._lengths = []
        store._save_index()
        store._open()
        return store

    @classmethod
    def open(cls, filepath: str) -> "MemmapSpectraStore":
        """Open an existing spectra store."""
        store = cls(filepath, 0, mode="r+")
        store._load_index()
        store._open()
        return store

    def _open(self):
        mz_path = self._filepath + ".mz.dat"
        int_path = self._filepath + ".int.dat"
        shape = (self._capacity, self._max_peaks)
        self._mz_data = np.memmap(mz_path, dtype="float64", mode=self._mode, shape=shape)
        self._int_data = np.memmap(int_path, dtype="float32", mode=self._mode, shape=shape)

    def _save_index(self):
        index = {
            "max_peaks": self._max_peaks,
            "capacity": self._capacity,
            "n_spectra": self._n_spectra,
            "lengths": self._lengths,
        }
        Path(self._index_path).write_text(json.dumps(index))

    def _load_index(self):
        data = json.loads(Path(self._index_path).read_text())
        self._max_peaks = data["max_peaks"]
        self._capacity = data["capacity"]
        self._n_spectra = data["n_spectra"]
        self._lengths = data["lengths"]

    @property
    def n_spectra(self) -> int:
        return self._n_spectra

    def add_spectrum(self, mz_array: np.ndarray, intensity_array: np.ndarray) -> int:
        """Add a spectrum and return its index."""
        n = min(len(mz_array), self._max_peaks)
        idx = self._n_spectra

        if idx >= self._capacity:
            raise RuntimeError("Store capacity exceeded. Create a larger store.")

        self._mz_data[idx, :n] = mz_array[:n]
        self._int_data[idx, :n] = intensity_array[:n]
        self._lengths.append(n)
        self._n_spectra += 1
        return idx

    def get_spectrum(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get m/z and intensity arrays for a spectrum."""
        if index >= self._n_spectra:
            raise IndexError(f"Spectrum index {index} out of range")
        n = self._lengths[index]
        mz = np.array(self._mz_data[index, :n])
        ints = np.array(self._int_data[index, :n])
        return mz, ints

    def flush(self):
        if self._mz_data is not None:
            self._mz_data.flush()
        if self._int_data is not None:
            self._int_data.flush()
        self._save_index()

    def close(self):
        self.flush()
        if self._mz_data is not None:
            del self._mz_data
            self._mz_data = None
        if self._int_data is not None:
            del self._int_data
            self._int_data = None

    def __repr__(self) -> str:
        return (f"MemmapSpectraStore('{self._filepath}', "
                f"n_spectra={self._n_spectra}, max_peaks={self._max_peaks})")

    def __del__(self):
        self.close()
