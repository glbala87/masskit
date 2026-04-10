"""Tests for memory-mapped storage module."""

import pytest
import numpy as np

from masskit.memmap import MemmapMatrix, MemmapSpectraStore


class TestMemmapMatrix:
    def test_create_and_read(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=10, n_cols=5)
        mm[0, 0] = 42.0
        mm[1, :] = np.ones(5) * 7.0
        mm.flush()
        assert mm.shape == (10, 5)
        assert mm.n_rows == 10
        assert mm.n_cols == 5
        assert mm[0, 0] == 42.0
        assert mm[1, 2] == 7.0
        mm.close()

    def test_open_existing(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=3, n_cols=4)
        mm[0, :] = [1.0, 2.0, 3.0, 4.0]
        mm.flush()
        mm.close()

        mm2 = MemmapMatrix.open(path)
        assert mm2[0, 0] == 1.0
        assert mm2[0, 3] == 4.0
        mm2.close()

    def test_to_array(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=3, n_cols=2)
        mm[:, :] = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        arr = mm.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (3, 2)
        np.testing.assert_array_equal(arr, [[1, 2], [3, 4], [5, 6]])
        mm.close()

    def test_column_means(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=4, n_cols=2)
        mm[:, :] = np.array([[2.0, 4.0], [4.0, 8.0], [6.0, 12.0], [8.0, 16.0]])
        means = mm.column_means()
        np.testing.assert_array_almost_equal(means, [5.0, 10.0])
        mm.close()

    def test_row_sums(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=3, n_cols=3)
        mm[:, :] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        sums = mm.row_sums()
        np.testing.assert_array_almost_equal(sums, [6.0, 15.0, 24.0])
        mm.close()

    def test_normalize_columns(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=2, n_cols=2)
        mm[:, :] = np.array([[100.0, 200.0], [300.0, 400.0]])
        mm.normalize_columns(target=1e6)
        # Column sums should be 1e6
        col_sums = np.sum(mm.to_array(), axis=0)
        np.testing.assert_allclose(col_sums, [1e6, 1e6], rtol=0.01)
        mm.close()

    def test_apply_log2(self, tmp_path):
        path = str(tmp_path / "test.mm")
        mm = MemmapMatrix.create(path, n_rows=2, n_cols=2)
        mm[:, :] = np.array([[8.0, 16.0], [32.0, 64.0]])
        mm.apply_log2(pseudocount=0.0)
        np.testing.assert_array_almost_equal(
            mm.to_array(),
            [[3.0, 4.0], [5.0, 6.0]],
        )
        mm.close()


class TestMemmapSpectraStore:
    def test_create_and_add(self, tmp_path):
        path = str(tmp_path / "spectra.mms")
        store = MemmapSpectraStore.create(path, max_peaks=100)
        idx = store.add_spectrum(
            np.array([100.0, 200.0, 300.0]),
            np.array([1000.0, 2000.0, 500.0]),
        )
        assert idx == 0
        assert store.n_spectra == 1
        store.flush()
        store.close()

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "spectra.mms")
        store = MemmapSpectraStore.create(path, max_peaks=100)
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1000.0, 2000.0, 500.0])
        store.add_spectrum(mz, intensity)
        store.flush()

        mz_out, int_out = store.get_spectrum(0)
        np.testing.assert_array_almost_equal(mz_out[:3], mz)
        np.testing.assert_array_almost_equal(int_out[:3], intensity)
        store.close()

    def test_multiple_spectra(self, tmp_path):
        path = str(tmp_path / "spectra.mms")
        store = MemmapSpectraStore.create(path, max_peaks=50)
        for i in range(5):
            mz = np.linspace(100, 200, 10) + i
            intensity = np.ones(10) * (i + 1) * 100
            store.add_spectrum(mz, intensity)
        assert store.n_spectra == 5
        store.close()

    def test_open_existing(self, tmp_path):
        path = str(tmp_path / "spectra.mms")
        store = MemmapSpectraStore.create(path, max_peaks=50)
        store.add_spectrum(np.array([100.0]), np.array([999.0]))
        store.flush()
        store.close()

        store2 = MemmapSpectraStore.open(path)
        assert store2.n_spectra == 1
        store2.close()
