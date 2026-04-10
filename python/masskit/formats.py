"""
Additional file format support for LC-MS data.

Supports reading and writing:
- MGF (Mascot Generic Format) for MS/MS spectra
- imzML (Imaging MS) basic support
- mzIdentML for identification results
- CSV/TSV feature tables
"""

from typing import Optional, List, Dict, Tuple, Any
from pathlib import Path
import numpy as np
import csv
import re

from .spectrum import Spectrum, SpectrumType, Polarity
from .experiment import MSExperiment
from .peak import Peak, PeakList
from .feature import Feature, FeatureMap


# =========================================================================
# MGF (Mascot Generic Format)
# =========================================================================

def load_mgf(filepath: str) -> MSExperiment:
    """
    Load MS/MS spectra from an MGF file.

    Args:
        filepath: Path to MGF file

    Returns:
        MSExperiment containing the spectra

    Example:
        >>> exp = load_mgf("tandem_spectra.mgf")
        >>> print(f"Loaded {exp.num_spectra} MS/MS spectra")
    """
    from .validation import validate_file_path
    validate_file_path(filepath, must_exist=True)
    exp = MSExperiment()
    exp.metadata["source_file"] = filepath
    exp.metadata["format"] = "MGF"

    with open(filepath, "r") as f:
        in_ions = False
        mz_list: List[float] = []
        int_list: List[float] = []
        metadata: Dict[str, str] = {}
        scan_index = 0

        for line in f:
            line = line.strip()

            if line == "BEGIN IONS":
                in_ions = True
                mz_list = []
                int_list = []
                metadata = {}
            elif line == "END IONS":
                if mz_list:
                    spec = Spectrum(
                        mz=np.array(mz_list, dtype=np.float64),
                        intensity=np.array(int_list, dtype=np.float64),
                        ms_level=2,
                    )
                    spec.index = scan_index
                    spec.spectrum_type = SpectrumType.CENTROID

                    if "rt" in metadata:
                        try:
                            spec.rt = float(metadata["rt"])
                        except ValueError:
                            pass

                    if "pepmass" in metadata:
                        parts = metadata["pepmass"].split()
                        precursor = {"mz": float(parts[0])}
                        if len(parts) > 1:
                            precursor["intensity"] = float(parts[1])
                        if "charge" in metadata:
                            charge_str = metadata["charge"].rstrip("+-")
                            try:
                                precursor["charge"] = int(charge_str)
                            except ValueError:
                                pass
                        spec.precursors = [precursor]

                    spec.native_id = metadata.get("title", f"scan={scan_index}")
                    spec.metadata = {k: v for k, v in metadata.items()
                                     if k not in ("pepmass", "charge", "rt")}

                    exp.add_spectrum(spec)
                    scan_index += 1

                in_ions = False
            elif in_ions:
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.strip().lower()] = value.strip()
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz_list.append(float(parts[0]))
                            int_list.append(float(parts[1]))
                        except ValueError:
                            pass

    return exp


def save_mgf(
    filepath: str,
    experiment: MSExperiment,
    ms_level: int = 2,
) -> int:
    """
    Save spectra to MGF format.

    Args:
        filepath: Output file path
        experiment: MSExperiment to save
        ms_level: MS level to export (default: 2 for MS/MS)

    Returns:
        Number of spectra written
    """
    from .validation import validate_output_path
    validate_output_path(filepath)
    count = 0
    with open(filepath, "w") as f:
        for spec in experiment.spectra:
            if ms_level > 0 and spec.ms_level != ms_level:
                continue

            f.write("BEGIN IONS\n")

            if spec.native_id:
                f.write(f"TITLE={spec.native_id}\n")
            else:
                f.write(f"TITLE=scan={spec.index}\n")

            if spec.rt > 0:
                f.write(f"RTINSECONDS={spec.rt:.4f}\n")

            if spec.precursors:
                pre = spec.precursors[0]
                mz = pre.get("mz", 0)
                intensity = pre.get("intensity", 0)
                if intensity > 0:
                    f.write(f"PEPMASS={mz:.6f} {intensity:.2f}\n")
                else:
                    f.write(f"PEPMASS={mz:.6f}\n")

                charge = pre.get("charge", 0)
                if charge:
                    f.write(f"CHARGE={charge}+\n")

            for key, value in spec.metadata.items():
                f.write(f"{key.upper()}={value}\n")

            for mz, intensity in zip(spec.mz, spec.intensity):
                f.write(f"{mz:.6f} {intensity:.4f}\n")

            f.write("END IONS\n\n")
            count += 1

    return count


# =========================================================================
# CSV/TSV Feature Tables
# =========================================================================

def save_feature_table(
    filepath: str,
    features: FeatureMap,
    delimiter: str = "\t",
    include_header: bool = True,
) -> int:
    """
    Save features to a CSV/TSV table.

    Args:
        filepath: Output file path
        features: FeatureMap to export
        delimiter: Column delimiter
        include_header: Write header row

    Returns:
        Number of features written
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)

        if include_header:
            writer.writerow([
                "feature_id", "mz", "rt", "intensity", "volume",
                "charge", "quality", "rt_min", "rt_max",
                "mz_min", "mz_max", "neutral_mass",
            ])

        for i, feat in enumerate(features):
            neutral_mass = feat.neutral_mass() if feat.charge > 0 else 0.0
            writer.writerow([
                f"F{i:06d}",
                f"{feat.mz:.6f}",
                f"{feat.rt:.2f}",
                f"{feat.intensity:.2f}",
                f"{feat.volume:.2f}",
                feat.charge,
                f"{feat.quality:.4f}",
                f"{feat.rt_min:.2f}",
                f"{feat.rt_max:.2f}",
                f"{feat.mz_min:.6f}",
                f"{feat.mz_max:.6f}",
                f"{neutral_mass:.4f}" if neutral_mass > 0 else "",
            ])

    return len(features)


def load_feature_table(
    filepath: str,
    delimiter: str = "\t",
) -> FeatureMap:
    """
    Load features from a CSV/TSV table.

    Args:
        filepath: Input file path
        delimiter: Column delimiter

    Returns:
        FeatureMap with loaded features
    """
    features = FeatureMap()

    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)

        for row in reader:
            feat = Feature()
            feat.mz = float(row.get("mz", 0))
            feat.rt = float(row.get("rt", 0))
            feat.intensity = float(row.get("intensity", 0))
            # Support both old "area" and new "volume" headers
            feat.volume = float(row.get("volume", row.get("area", 0)) or 0)
            feat.charge = int(row.get("charge", 0))
            feat.quality = float(row.get("quality", 0))

            rt_min = row.get("rt_min", row.get("rt_start"))
            rt_max = row.get("rt_max", row.get("rt_end"))
            if rt_min and rt_max:
                feat.rt_min = float(rt_min)
                feat.rt_max = float(rt_max)

            mz_min = row.get("mz_min", row.get("mz_start"))
            mz_max = row.get("mz_max", row.get("mz_end"))
            if mz_min and mz_max:
                feat.mz_min = float(mz_min)
                feat.mz_max = float(mz_max)

            features.add(feat)

    return features


# =========================================================================
# Peak List Formats
# =========================================================================

def save_peak_list(
    filepath: str,
    peaks: PeakList,
    delimiter: str = "\t",
) -> int:
    """Save peaks to a CSV/TSV file."""
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["mz", "rt", "intensity", "area", "snr", "fwhm", "charge"])

        for peak in peaks:
            writer.writerow([
                f"{peak.mz:.6f}",
                f"{peak.rt:.2f}",
                f"{peak.intensity:.2f}",
                f"{peak.area:.2f}",
                f"{peak.snr:.2f}",
                f"{peak.fwhm_mz:.6f}",
                peak.charge,
            ])

    return len(peaks)


def load_peak_list(
    filepath: str,
    delimiter: str = "\t",
) -> PeakList:
    """Load peaks from a CSV/TSV file."""
    peaks = PeakList()

    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            peak = Peak(
                mz=float(row.get("mz", 0)),
                rt=float(row.get("rt", 0)),
                intensity=float(row.get("intensity", 0)),
                area=float(row.get("area", 0)),
                snr=float(row.get("snr", 0)),
                fwhm_mz=float(row.get("fwhm", 0)),
                charge=int(row.get("charge", 0)),
            )
            peaks.add(peak)

    return peaks


# =========================================================================
# imzML (Imaging Mass Spectrometry)
# =========================================================================

def load_imzml_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse imzML metadata (XML part only, not the binary .ibd file).

    Args:
        filepath: Path to .imzML file

    Returns:
        Dict with imaging metadata (dimensions, pixel size, etc.)

    Note:
        Full imzML support requires reading the associated .ibd binary file.
        This function provides metadata extraction only.
    """
    import xml.etree.ElementTree as ET

    metadata: Dict[str, Any] = {
        "format": "imzML",
        "filepath": filepath,
        "spectra_count": 0,
        "pixel_coordinates": [],
        "max_x": 0,
        "max_y": 0,
    }

    try:
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Handle namespace
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0] + "}"

        # Count spectra
        spectrum_list = root.find(f".//{ns}spectrumList")
        if spectrum_list is not None:
            count = spectrum_list.get("count", "0")
            metadata["spectra_count"] = int(count)

        # Extract pixel coordinates
        for spectrum in root.iter(f"{ns}spectrum"):
            x, y, z = 0, 0, 0
            for param in spectrum.iter(f"{ns}cvParam"):
                name = param.get("name", "")
                if "position x" in name.lower():
                    x = int(param.get("value", 0))
                elif "position y" in name.lower():
                    y = int(param.get("value", 0))
                elif "position z" in name.lower():
                    z = int(param.get("value", 0))
            metadata["pixel_coordinates"].append((x, y, z))

        if metadata["pixel_coordinates"]:
            xs = [c[0] for c in metadata["pixel_coordinates"]]
            ys = [c[1] for c in metadata["pixel_coordinates"]]
            metadata["max_x"] = max(xs)
            metadata["max_y"] = max(ys)
            metadata["dimensions"] = (max(xs), max(ys))

    except ET.ParseError as e:
        metadata["error"] = str(e)

    return metadata


# =========================================================================
# mzIdentML (Identification Results)
# =========================================================================

def save_mzidentml(
    filepath: str,
    identifications: List[Dict[str, Any]],
    search_params: Optional[Dict[str, str]] = None,
) -> int:
    """
    Save identification results in a simplified mzIdentML-like TSV format.

    Args:
        filepath: Output file path
        identifications: List of identification dicts with keys:
            scan, peptide, protein, score, charge, mz, rt, modifications
        search_params: Optional search parameters

    Returns:
        Number of identifications written
    """
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")

        # Header
        if search_params:
            for key, value in search_params.items():
                f.write(f"# {key}={value}\n")

        writer.writerow([
            "scan", "rt", "charge", "precursor_mz",
            "peptide", "protein", "score", "q_value",
            "modifications", "mass_error_ppm",
        ])

        for ident in identifications:
            writer.writerow([
                ident.get("scan", ""),
                f"{ident.get('rt', 0):.2f}",
                ident.get("charge", ""),
                f"{ident.get('mz', 0):.6f}",
                ident.get("peptide", ""),
                ident.get("protein", ""),
                f"{ident.get('score', 0):.4f}",
                f"{ident.get('q_value', 1.0):.6f}",
                ident.get("modifications", ""),
                f"{ident.get('mass_error_ppm', 0):.2f}",
            ])

    return len(identifications)


def load_identification_table(
    filepath: str,
    delimiter: str = "\t",
) -> List[Dict[str, Any]]:
    """
    Load identification results from a TSV file.

    Args:
        filepath: Input file path
        delimiter: Column delimiter

    Returns:
        List of identification dicts
    """
    results = []

    with open(filepath, "r") as f:
        # Skip comment lines
        lines = [line for line in f if not line.startswith("#")]

    reader = csv.DictReader(lines, delimiter=delimiter)
    for row in reader:
        ident = {
            "scan": row.get("scan", ""),
            "peptide": row.get("peptide", ""),
            "protein": row.get("protein", ""),
        }

        for field in ("rt", "precursor_mz", "score", "q_value", "mass_error_ppm"):
            try:
                ident[field] = float(row.get(field, 0))
            except (ValueError, TypeError):
                ident[field] = 0.0

        try:
            ident["charge"] = int(row.get("charge", 0))
        except (ValueError, TypeError):
            ident["charge"] = 0

        ident["modifications"] = row.get("modifications", "")
        results.append(ident)

    return results
