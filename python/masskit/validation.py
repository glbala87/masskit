"""
Input validation and path sanitization utilities for MassKit.

Provides validation for file paths, numeric parameters, and data arrays
to catch errors early and prevent security issues.
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .exceptions import (
    ValidationError,
    PathTraversalError,
    SpectrumError,
)


def validate_file_path(
    filepath: str,
    must_exist: bool = True,
    allowed_extensions: Optional[Sequence[str]] = None,
    max_size_mb: float = 0,
) -> Path:
    """
    Validate and sanitize a file path.

    Args:
        filepath: Path to validate
        must_exist: Whether the file must exist
        allowed_extensions: Allowed file extensions (e.g., ['.mzML', '.mzXML'])
        max_size_mb: Maximum file size in MB (0 = no limit)

    Returns:
        Resolved Path object

    Raises:
        PathTraversalError: If path contains traversal sequences
        FileNotFoundError: If must_exist and file doesn't exist
        ValidationError: If extension or size check fails
    """
    path = Path(filepath)

    # Check for path traversal
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as e:
        raise ValidationError(f"Invalid file path: {filepath}") from e

    # Detect traversal attempts in the original path string
    normalized = os.path.normpath(filepath)
    if ".." in normalized.split(os.sep):
        raise PathTraversalError(filepath)

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    if must_exist and not resolved.is_file():
        raise ValidationError(f"Not a file: {filepath}")

    if allowed_extensions:
        lower_exts = [e.lower() for e in allowed_extensions]
        if resolved.suffix.lower() not in lower_exts:
            raise ValidationError(
                f"Unsupported file type '{resolved.suffix}'. "
                f"Allowed: {', '.join(allowed_extensions)}"
            )

    if max_size_mb > 0 and must_exist:
        size_mb = resolved.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            raise ValidationError(
                f"File too large: {size_mb:.1f} MB (max: {max_size_mb:.1f} MB)"
            )

    return resolved


def validate_output_path(filepath: str) -> Path:
    """
    Validate an output file path and ensure parent directory exists.

    Args:
        filepath: Output path to validate

    Returns:
        Resolved Path object
    """
    path = Path(filepath)

    normalized = os.path.normpath(filepath)
    if ".." in normalized.split(os.sep):
        raise PathTraversalError(filepath)

    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def validate_mz_intensity(
    mz: Union[np.ndarray, list],
    intensity: Union[np.ndarray, list],
) -> None:
    """
    Validate m/z and intensity arrays.

    Raises:
        SpectrumError: If arrays are invalid
    """
    mz = np.asarray(mz)
    intensity = np.asarray(intensity)

    if mz.ndim != 1 or intensity.ndim != 1:
        raise SpectrumError("m/z and intensity must be 1-dimensional arrays")

    if len(mz) != len(intensity):
        raise SpectrumError(
            f"m/z ({len(mz)}) and intensity ({len(intensity)}) "
            "arrays must have same length"
        )

    if len(mz) > 0:
        if np.any(np.isnan(mz)):
            raise SpectrumError("m/z array contains NaN values")
        if np.any(np.isnan(intensity)):
            raise SpectrumError("Intensity array contains NaN values")
        if np.any(mz < 0):
            raise SpectrumError("m/z values must be non-negative")
        if np.any(intensity < 0):
            raise SpectrumError("Intensity values must be non-negative")


def validate_positive(value: float, name: str) -> float:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
    return value


def validate_non_negative(value: float, name: str) -> float:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValidationError(f"{name} must be non-negative, got {value}")
    return value


def validate_range(
    value: float, name: str, min_val: float, max_val: float
) -> float:
    """Validate that a value is within a range."""
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )
    return value


def validate_peptide_sequence(sequence: str) -> str:
    """
    Validate a peptide sequence.

    Args:
        sequence: Amino acid sequence

    Returns:
        Uppercased sequence

    Raises:
        ValidationError: If sequence contains invalid characters
    """
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    upper = sequence.upper()
    invalid = set(upper) - valid_aa
    if invalid:
        raise ValidationError(
            f"Invalid amino acid(s) in sequence: {', '.join(sorted(invalid))}"
        )
    return upper
