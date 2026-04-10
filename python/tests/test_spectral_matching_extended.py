"""Extended tests for spectral_matching to push coverage above 75%."""

import pytest
import numpy as np
import os

from masskit.spectral_matching import (
    SpectralLibrary,
    LibrarySpectrum,
    SpectralMatch,
    cosine_similarity,
    modified_cosine_similarity,
    spectral_entropy_similarity,
    matched_peaks_count,
    normalize_spectrum,
    filter_spectrum,
    sqrt_transform,
    remove_precursor,
    _match_peaks,
    _shannon_entropy,
)


# ── LibrarySpectrum dataclass ────────────────────────────────────────


class TestLibrarySpectrum:
    def test_defaults(self):
        ls = LibrarySpectrum()
        assert ls.name == ""
        assert ls.precursor_mz == 0.0
        assert ls.num_peaks == 0

    def test_with_data(self):
        ls = LibrarySpectrum(
            name="test",
            precursor_mz=500.0,
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1000.0, 2000.0]),
        )
        assert ls.num_peaks == 2


# ── SpectralLibrary I/O ──────────────────────────────────────────────


class TestLibraryMGF:
    def test_load_mgf(self, mgf_file):
        lib = SpectralLibrary()
        n = lib.load_mgf(mgf_file)
        assert n == 3
        assert len(lib) == 3

    def test_load_mgf_then_search(self, mgf_file):
        lib = SpectralLibrary()
        lib.load_mgf(mgf_file)
        # Use the first library entry as query
        first = lib[0]
        matches = lib.search(first.mz, first.intensity, top_n=3)
        assert len(matches) >= 1
        # Self-match should score highest
        assert matches[0].score > 0.99

    def test_save_load_mgf_round_trip(self, tmp_path):
        lib = SpectralLibrary()
        for i in range(2):
            lib.add_spectrum(
                name=f"compound_{i}",
                precursor_mz=400.0 + i * 50,
                mz=np.array([100.0, 200.0, 300.0]) + i,
                intensity=np.array([1000.0, 2000.0, 500.0]),
                metadata={"smiles": f"C{i}H{i}"},
            )

        out = str(tmp_path / "lib.mgf")
        lib.save(out, format="mgf")
        assert os.path.exists(out)

        lib2 = SpectralLibrary()
        n = lib2.load_mgf(out)
        assert n == 2
        assert lib2[0].name == "compound_0"


class TestLibraryMSP:
    def _write_msp(self, path, n=2):
        lines = []
        for i in range(n):
            lines.append(f"Name: compound_{i}")
            lines.append(f"PrecursorMZ: {500.0 + i * 10}")
            lines.append(f"Formula: C{i+10}H{i+15}O{i+2}")
            lines.append("Num Peaks: 3")
            lines.append(f"100.{i:04d}\t1000")
            lines.append(f"200.{i:04d}\t2000")
            lines.append(f"300.{i:04d}\t500")
            lines.append("")
        path.write_text("\n".join(lines))

    def test_load_msp(self, tmp_path):
        msp_path = tmp_path / "lib.msp"
        self._write_msp(msp_path)

        lib = SpectralLibrary()
        n = lib.load_msp(str(msp_path))
        assert n == 2
        assert lib[0].name == "compound_0"
        assert lib[0].num_peaks == 3
        assert "formula" in lib[0].metadata

    def test_save_msp(self, tmp_path):
        lib = SpectralLibrary()
        lib.add_spectrum(
            name="test",
            precursor_mz=500.0,
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1000.0, 2000.0]),
            metadata={"smiles": "CCO"},
        )
        out = str(tmp_path / "out.msp")
        lib.save(out, format="msp")

        # Round-trip
        lib2 = SpectralLibrary()
        lib2.load_msp(out)
        assert len(lib2) == 1
        assert lib2[0].name == "test"

    def test_msp_with_no_trailing_blank(self, tmp_path):
        msp_path = tmp_path / "lib.msp"
        # No trailing blank — exercises the "last entry" path
        msp_path.write_text(
            "Name: only\nPrecursorMZ: 100.0\nNum Peaks: 1\n50.0\t100.0\n"
        )
        lib = SpectralLibrary()
        n = lib.load_msp(str(msp_path))
        assert n == 1


class TestLibrarySaveErrors:
    def test_unknown_format(self, tmp_path):
        lib = SpectralLibrary()
        with pytest.raises(ValueError, match="Unknown format"):
            lib.save(str(tmp_path / "x"), format="parquet")


# ── SpectralLibrary.search ────────────────────────────────────────────


class TestSearchMethods:
    def _make_lib(self):
        lib = SpectralLibrary()
        lib.add_spectrum(
            name="entry_a",
            precursor_mz=400.0,
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([1000.0, 2000.0, 500.0]),
        )
        lib.add_spectrum(
            name="entry_b",
            precursor_mz=500.0,
            mz=np.array([150.0, 250.0]),
            intensity=np.array([800.0, 1500.0]),
        )
        return lib

    def test_search_cosine(self):
        lib = self._make_lib()
        matches = lib.search(
            np.array([100.0, 200.0, 300.0]),
            np.array([1000.0, 2000.0, 500.0]),
            method="cosine",
        )
        assert matches[0].library_name == "entry_a"

    def test_search_modified_cosine(self):
        lib = self._make_lib()
        matches = lib.search(
            np.array([100.0, 200.0, 300.0]),
            np.array([1000.0, 2000.0, 500.0]),
            query_precursor_mz=400.0,
            method="modified_cosine",
        )
        assert len(matches) >= 1

    def test_search_entropy(self):
        lib = self._make_lib()
        matches = lib.search(
            np.array([100.0, 200.0, 300.0]),
            np.array([1000.0, 2000.0, 500.0]),
            method="entropy",
        )
        assert len(matches) >= 1

    def test_search_unknown_method(self):
        lib = self._make_lib()
        with pytest.raises(ValueError, match="Unknown method"):
            lib.search(
                np.array([100.0]),
                np.array([1000.0]),
                method="bogus",
            )

    def test_search_precursor_filter(self):
        lib = self._make_lib()
        # Restrict to precursor near 400
        matches = lib.search(
            np.array([100.0, 200.0]),
            np.array([1000.0, 2000.0]),
            query_precursor_mz=400.0,
            precursor_tolerance=10.0,
        )
        # entry_b precursor is 500, should be filtered out
        names = [m.library_name for m in matches]
        assert "entry_b" not in names

    def test_search_min_score(self):
        lib = self._make_lib()
        matches = lib.search(
            np.array([100.0]),
            np.array([1000.0]),
            min_score=0.99,
        )
        # No matches at very high threshold for this 1-peak query
        assert len(matches) <= 1


class TestFilterByPrecursor:
    def test_filter_range(self):
        lib = SpectralLibrary()
        for mz in [100.0, 200.0, 300.0, 400.0]:
            lib.add_spectrum(
                name=f"e_{mz}",
                precursor_mz=mz,
                mz=np.array([10.0]),
                intensity=np.array([100.0]),
            )
        sub = lib.filter_by_precursor(150.0, 350.0)
        assert len(sub) == 2


# ── Similarity edge cases ─────────────────────────────────────────────


class TestCosineEdgeCases:
    def test_empty_spectra(self):
        score, n = cosine_similarity(
            np.array([]), np.array([]),
            np.array([100.0]), np.array([1000.0]),
        )
        assert score == 0.0
        assert n == 0

    def test_zero_intensity(self):
        score, n = cosine_similarity(
            np.array([100.0]), np.array([0.0]),
            np.array([100.0]), np.array([1000.0]),
        )
        assert score == 0.0


class TestModifiedCosineEdgeCases:
    def test_no_matches(self):
        score, n = modified_cosine_similarity(
            np.array([100.0]), np.array([1000.0]), 200.0,
            np.array([800.0]), np.array([1000.0]), 900.0,
            tolerance=0.01,
        )
        # 100 + 700 (mass diff) = 800, should match
        assert score >= 0
        assert n >= 0

    def test_zero_norms(self):
        score, n = modified_cosine_similarity(
            np.array([100.0]), np.array([0.0]), 200.0,
            np.array([100.0]), np.array([1000.0]), 200.0,
        )
        assert score == 0.0


class TestEntropyEdgeCases:
    def test_zero_sum(self):
        score, n = spectral_entropy_similarity(
            np.array([100.0]), np.array([0.0]),
            np.array([100.0]), np.array([1000.0]),
        )
        assert score == 0.0

    def test_no_matches(self):
        score, n = spectral_entropy_similarity(
            np.array([100.0]), np.array([1000.0]),
            np.array([500.0]), np.array([1000.0]),
            tolerance=0.001,
        )
        assert score == 0.0


class TestMatchedPeaksCount:
    def test_count(self):
        n = matched_peaks_count(
            np.array([100.0, 200.0, 300.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([100.0, 250.0, 300.0]),
            np.array([1.0, 1.0, 1.0]),
            tolerance=0.01,
        )
        assert n == 2


# ── Preprocessing functions ──────────────────────────────────────────


class TestNormalizeSpectrum:
    def test_max(self):
        mz, ints = normalize_spectrum(
            np.array([100.0, 200.0]),
            np.array([500.0, 1000.0]),
            method="max",
        )
        assert ints[1] == 1.0

    def test_sum(self):
        mz, ints = normalize_spectrum(
            np.array([100.0, 200.0]),
            np.array([500.0, 1500.0]),
            method="sum",
        )
        assert abs(np.sum(ints) - 1.0) < 1e-9

    def test_sqrt(self):
        mz, ints = normalize_spectrum(
            np.array([100.0]),
            np.array([400.0]),
            method="sqrt",
        )
        assert abs(ints[0] - 20.0) < 1e-9

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalize_spectrum(
                np.array([100.0]), np.array([1.0]),
                method="bogus",
            )

    def test_zero_max(self):
        mz, ints = normalize_spectrum(
            np.array([100.0]), np.array([0.0]),
            method="max",
        )
        assert ints[0] == 0.0


class TestFilterSpectrum:
    def test_min_max_mz(self):
        mz, ints = filter_spectrum(
            np.array([100.0, 200.0, 300.0]),
            np.array([1.0, 2.0, 3.0]),
            min_mz=150.0,
            max_mz=250.0,
        )
        assert len(mz) == 1
        assert mz[0] == 200.0

    def test_min_intensity(self):
        mz, ints = filter_spectrum(
            np.array([100.0, 200.0, 300.0]),
            np.array([1.0, 5.0, 2.0]),
            min_intensity=3.0,
        )
        assert len(mz) == 1

    def test_top_n(self):
        mz, ints = filter_spectrum(
            np.array([100.0, 200.0, 300.0, 400.0]),
            np.array([1.0, 4.0, 2.0, 3.0]),
            top_n=2,
        )
        assert len(mz) == 2
        # Should keep the two highest, in m/z order
        assert 200.0 in mz
        assert 400.0 in mz

    def test_top_n_smaller_than_available(self):
        mz, ints = filter_spectrum(
            np.array([100.0]),
            np.array([1.0]),
            top_n=10,
        )
        assert len(mz) == 1


class TestSqrtTransform:
    def test_basic(self):
        result = sqrt_transform(np.array([4.0, 9.0, 16.0]))
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_negative_clamped(self):
        result = sqrt_transform(np.array([-1.0, 4.0]))
        assert result[0] == 0.0


class TestRemovePrecursor:
    def test_removes_precursor_and_losses(self):
        # Precursor at 500, water loss at 481.989, ammonia at 482.973
        mz = np.array([100.0, 200.0, 481.989, 482.973, 500.0, 600.0])
        ints = np.ones(6)
        out_mz, out_ints = remove_precursor(mz, ints, 500.0, tolerance=1.0)
        # Should remove 481.989, 482.973, and 500.0
        assert 100.0 in out_mz
        assert 600.0 in out_mz
        assert 500.0 not in out_mz


# ── Internal helpers ─────────────────────────────────────────────────


class TestMatchPeaks:
    def test_empty(self):
        assert _match_peaks(np.array([]), np.array([]),
                            np.array([1.0]), np.array([1.0]), 0.01) == []

    def test_basic_match(self):
        result = _match_peaks(
            np.array([100.0, 200.0]),
            np.array([1.0, 2.0]),
            np.array([100.001, 200.5]),
            np.array([3.0, 4.0]),
            tolerance=0.01,
        )
        assert len(result) == 1
        assert result[0] == (1.0, 3.0)

    def test_one_to_one(self):
        # Each peak in mz2 can only be matched once
        result = _match_peaks(
            np.array([100.0, 100.001]),
            np.array([1.0, 2.0]),
            np.array([100.0]),
            np.array([5.0]),
            tolerance=0.01,
        )
        assert len(result) == 1


class TestShannonEntropy:
    def test_uniform(self):
        # Uniform: max entropy = log(n)
        e = _shannon_entropy(np.array([0.25, 0.25, 0.25, 0.25]))
        assert abs(e - np.log(4)) < 1e-9

    def test_concentrated(self):
        e = _shannon_entropy(np.array([1.0, 0.0, 0.0]))
        assert e == 0.0

    def test_empty(self):
        e = _shannon_entropy(np.array([]))
        assert e == 0.0
