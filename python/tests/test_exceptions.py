"""Tests for the exception hierarchy."""

import pytest
from masskit.exceptions import (
    LCMSError,
    FileFormatError,
    ValidationError,
    SpectrumError,
    ConfigurationError,
    DependencyError,
    PathTraversalError,
    SearchEngineError,
    QuantificationError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        with pytest.raises(LCMSError):
            raise LCMSError("test")

    def test_file_format_error(self):
        err = FileFormatError("test.raw", "unsupported format")
        assert "test.raw" in str(err)
        assert "unsupported format" in str(err)
        assert isinstance(err, LCMSError)

    def test_file_format_no_detail(self):
        err = FileFormatError("test.raw")
        assert "test.raw" in str(err)

    def test_validation_error(self):
        assert issubclass(ValidationError, LCMSError)
        assert issubclass(SpectrumError, ValidationError)

    def test_configuration_error(self):
        assert issubclass(ConfigurationError, LCMSError)

    def test_dependency_error(self):
        err = DependencyError("dask", "distributed processing")
        assert "dask" in str(err)
        assert "distributed processing" in str(err)
        assert "pip install" in str(err)

    def test_dependency_error_no_feature(self):
        err = DependencyError("matplotlib")
        assert "matplotlib" in str(err)

    def test_path_traversal_error(self):
        err = PathTraversalError("../../etc/passwd")
        assert "traversal" in str(err).lower()
        assert isinstance(err, LCMSError)

    def test_search_engine_error(self):
        assert issubclass(SearchEngineError, LCMSError)

    def test_quantification_error(self):
        assert issubclass(QuantificationError, LCMSError)

    def test_catch_all_lcms(self):
        """All custom exceptions should be catchable as LCMSError."""
        errors = [
            FileFormatError("f"),
            ValidationError("v"),
            SpectrumError("s"),
            ConfigurationError("c"),
            DependencyError("d"),
            PathTraversalError("p"),
            SearchEngineError("se"),
            QuantificationError("q"),
        ]
        for err in errors:
            assert isinstance(err, LCMSError)
