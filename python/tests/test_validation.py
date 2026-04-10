"""Tests for the validation module."""

import pytest
import tempfile
import os
import numpy as np

from masskit.validation import (
    validate_file_path,
    validate_output_path,
    validate_mz_intensity,
    validate_positive,
    validate_non_negative,
    validate_range,
    validate_peptide_sequence,
)
from masskit.exceptions import (
    ValidationError,
    PathTraversalError,
    SpectrumError,
)


class TestFilePathValidation:
    def test_existing_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mzML", delete=False) as f:
            try:
                path = validate_file_path(f.name)
                assert path.exists()
            finally:
                os.unlink(f.name)

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            validate_file_path("/nonexistent/file.mzML")

    def test_path_traversal(self):
        with pytest.raises(PathTraversalError):
            validate_file_path("../../etc/passwd", must_exist=False)

    def test_allowed_extensions(self):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            try:
                with pytest.raises(ValidationError, match="Unsupported"):
                    validate_file_path(f.name, allowed_extensions=[".mzML", ".mzXML"])
            finally:
                os.unlink(f.name)

    def test_valid_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".mzML", delete=False) as f:
            try:
                path = validate_file_path(f.name, allowed_extensions=[".mzML"])
                assert path.exists()
            finally:
                os.unlink(f.name)

    def test_no_must_exist(self):
        path = validate_file_path("/some/future/file.mzML", must_exist=False)
        assert path is not None


class TestOutputPathValidation:
    def test_basic(self):
        with tempfile.TemporaryDirectory() as td:
            path = validate_output_path(os.path.join(td, "output.csv"))
            assert path is not None

    def test_creates_parent(self):
        with tempfile.TemporaryDirectory() as td:
            nested = os.path.join(td, "sub", "dir", "output.csv")
            path = validate_output_path(nested)
            assert path.parent.exists()

    def test_traversal_blocked(self):
        with pytest.raises(PathTraversalError):
            validate_output_path("../../evil/path.csv")


class TestMzIntensityValidation:
    def test_valid_arrays(self):
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([1000.0, 2000.0, 500.0])
        validate_mz_intensity(mz, intensity)  # No exception

    def test_length_mismatch(self):
        with pytest.raises(SpectrumError, match="same length"):
            validate_mz_intensity([1.0, 2.0], [1.0])

    def test_nan_mz(self):
        with pytest.raises(SpectrumError, match="NaN"):
            validate_mz_intensity([float("nan"), 2.0], [1.0, 2.0])

    def test_negative_mz(self):
        with pytest.raises(SpectrumError, match="non-negative"):
            validate_mz_intensity([-1.0, 2.0], [1.0, 2.0])

    def test_negative_intensity(self):
        with pytest.raises(SpectrumError, match="non-negative"):
            validate_mz_intensity([1.0, 2.0], [-1.0, 2.0])

    def test_empty_arrays(self):
        validate_mz_intensity([], [])  # No exception

    def test_multidimensional(self):
        with pytest.raises(SpectrumError, match="1-dimensional"):
            validate_mz_intensity([[1.0]], [[2.0]])


class TestNumericValidation:
    def test_validate_positive(self):
        assert validate_positive(5.0, "test") == 5.0

    def test_validate_positive_zero(self):
        with pytest.raises(ValidationError, match="positive"):
            validate_positive(0.0, "test")

    def test_validate_non_negative(self):
        assert validate_non_negative(0.0, "test") == 0.0

    def test_validate_non_negative_fails(self):
        with pytest.raises(ValidationError, match="non-negative"):
            validate_non_negative(-1.0, "test")

    def test_validate_range(self):
        assert validate_range(0.5, "test", 0.0, 1.0) == 0.5

    def test_validate_range_fails(self):
        with pytest.raises(ValidationError, match="between"):
            validate_range(2.0, "test", 0.0, 1.0)


class TestPeptideValidation:
    def test_valid_sequence(self):
        assert validate_peptide_sequence("PEPTIDE") == "PEPTIDE"

    def test_lowercase(self):
        assert validate_peptide_sequence("peptide") == "PEPTIDE"

    def test_invalid_chars(self):
        with pytest.raises(ValidationError, match="Invalid amino acid"):
            validate_peptide_sequence("PEPTXDE")

    def test_numbers_invalid(self):
        with pytest.raises(ValidationError):
            validate_peptide_sequence("PEP1TIDE")
