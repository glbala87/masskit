"""
Custom exception hierarchy for MassKit.

Provides specific exception types for different error categories,
enabling precise error handling and informative error messages.
"""


class LCMSError(Exception):
    """Base exception for all MassKit errors."""
    pass


class FileFormatError(LCMSError):
    """Raised when a file cannot be parsed or has an unsupported format."""
    def __init__(self, filepath: str, detail: str = ""):
        self.filepath = filepath
        self.detail = detail
        msg = f"Cannot parse file '{filepath}'"
        if detail:
            msg += f": {detail}"
        super().__init__(msg)


class FileNotFoundError(LCMSError):
    """Raised when a required input file does not exist."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        super().__init__(f"File not found: '{filepath}'")


class ValidationError(LCMSError):
    """Raised when input data fails validation checks."""
    pass


class SpectrumError(ValidationError):
    """Raised for invalid spectrum data."""
    pass


class PrecursorError(ValidationError):
    """Raised for invalid precursor data."""
    pass


class ConfigurationError(LCMSError):
    """Raised for invalid configuration parameters."""
    pass


class DependencyError(LCMSError):
    """Raised when an optional dependency is required but not installed."""
    def __init__(self, package: str, feature: str = ""):
        self.package = package
        self.feature = feature
        msg = f"Required package '{package}' is not installed"
        if feature:
            msg += f" (needed for {feature})"
        msg += f". Install with: pip install {package}"
        super().__init__(msg)


class SearchEngineError(LCMSError):
    """Raised when an external search engine fails."""
    pass


class QuantificationError(LCMSError):
    """Raised for errors during quantification workflows."""
    pass


class PathTraversalError(LCMSError):
    """Raised when a file path attempts directory traversal."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        super().__init__(f"Path traversal detected in: '{filepath}'")
