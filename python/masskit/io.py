"""
I/O functions for reading and writing LC-MS data files.
"""

import base64
import binascii
import struct
import zlib
import logging
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


def _find_first(*candidates):
    """Return the first non-None Element from candidates (avoids truthy checks)."""
    for c in candidates:
        if c is not None:
            return c
    return None


def _findall_any(elem, ns_path: str, plain_path: str, ns: Dict[str, str]) -> List[ET.Element]:
    """findall with namespace fallback (avoids truthy 'or' on lists)."""
    result = elem.findall(ns_path, ns)
    if result:
        return result
    return elem.findall(plain_path)


def _iter_descendants_any(elem, tag: str, ns: Dict[str, str]):
    """
    Iterate over all descendants matching a local tag, regardless of namespace.

    Used to locate cvParam/scan/binaryDataArray elements which may live inside
    different parent hierarchies across mzML versions (1.1 uses scanList/
    binaryDataArrayList wrappers, 0.99 uses spectrumDescription wrapping and
    direct binaryDataArray children).
    """
    yielded = set()
    if ns:
        for e in elem.iter(f"{{{list(ns.values())[0]}}}{tag}"):
            yielded.add(id(e))
            yield e
    for e in elem.iter(tag):
        if id(e) not in yielded:
            yield e


def _auto_detect_precision(decoded_bytes: int, array_length: int, declared_64bit: bool) -> bool:
    """
    Infer actual binary precision from byte count, correcting for mislabeled
    mzML 0.99 files that declare 32-bit but actually contain 64-bit data.

    Returns True for 64-bit, False for 32-bit.
    """
    if array_length <= 0:
        return declared_64bit
    per_element = decoded_bytes / array_length
    if abs(per_element - 8.0) < 0.01:
        return True
    if abs(per_element - 4.0) < 0.01:
        return False
    return declared_64bit


def _decode_binary_robust(
    decoded_bytes: bytes,
    is_64bit: bool,
    array_length: int = 0,
) -> List[float]:
    """
    Decode a binary blob into floats, auto-detecting endianness when the
    declared layout produces non-physical values (extreme magnitudes or NaN).

    mzML 1.1 mandates little-endian but some mzML 0.99 files use big-endian.
    This helper tries little-endian first and falls back to big-endian if
    the decoded values look corrupt.
    """
    import math

    size = 8 if is_64bit else 4
    fmt = "d" if is_64bit else "f"
    count = len(decoded_bytes) // size
    if count == 0:
        return []

    # First: little-endian (spec-compliant)
    le_values = list(
        struct.unpack(f"<{count}{fmt}", decoded_bytes[: count * size])
    )

    def _looks_sensible(values: List[float]) -> bool:
        # Quick heuristic: sample up to 16 values, reject if any is NaN, inf,
        # extreme magnitude (> 1e10 or non-zero < 1e-20), or negative.
        # Real m/z and intensity values are always positive and in a
        # tractable range.
        sample = values[: min(16, len(values))]
        for v in sample:
            if math.isnan(v) or math.isinf(v):
                return False
            if v < 0:
                return False
            if v > 1e10:
                return False
            if 0 < v < 1e-20:
                return False  # denormalized: likely wrong endianness
        return True

    if _looks_sensible(le_values):
        return le_values

    # Fall back to big-endian
    be_values = list(
        struct.unpack(f">{count}{fmt}", decoded_bytes[: count * size])
    )
    if _looks_sensible(be_values):
        return be_values

    # Neither worked — return little-endian anyway and let the caller decide
    return le_values

from .spectrum import Spectrum, SpectrumType, Polarity, Precursor
from .chromatogram import Chromatogram, ChromatogramType
from .experiment import MSExperiment


# CV term accessions
CV_MS_LEVEL = "MS:1000511"
CV_PROFILE = "MS:1000128"
CV_CENTROID = "MS:1000127"
CV_POSITIVE = "MS:1000130"
CV_NEGATIVE = "MS:1000129"
CV_MZ_ARRAY = "MS:1000514"
CV_INTENSITY_ARRAY = "MS:1000515"
CV_TIME_ARRAY = "MS:1000595"
CV_FLOAT64 = "MS:1000523"
CV_FLOAT32 = "MS:1000521"
CV_ZLIB = "MS:1000574"
CV_SCAN_TIME = "MS:1000016"
CV_TIC = "MS:1000285"
CV_SELECTED_MZ = "MS:1000744"
CV_CHARGE = "MS:1000041"
CV_COLLISION_ENERGY = "MS:1000045"
CV_MINUTE = "UO:0000031"
CV_MINUTE_MZML099 = "MS:1000038"  # older mzML 0.99 minute unit


def decode_binary(
    data: str,
    is_64bit: bool = True,
    is_compressed: bool = False,
    is_little_endian: bool = True,
) -> List[float]:
    """
    Decode Base64 encoded binary data.

    Args:
        data: Base64 encoded string
        is_64bit: True for 64-bit floats, False for 32-bit
        is_compressed: True if zlib compressed
        is_little_endian: True for little-endian byte order

    Returns:
        List of decoded values
    """
    # Remove whitespace
    data = "".join(data.split())
    if not data:
        return []

    # Decode Base64
    binary = base64.b64decode(data)

    # Decompress if needed
    if is_compressed:
        binary = zlib.decompress(binary)

    # Unpack floats
    if is_64bit:
        fmt = "<d" if is_little_endian else ">d"
        size = 8
    else:
        fmt = "<f" if is_little_endian else ">f"
        size = 4

    count = len(binary) // size
    values = []
    for i in range(count):
        value = struct.unpack(fmt, binary[i * size : (i + 1) * size])[0]
        values.append(value)

    return values


def load_mzml(
    filename: str,
    ms_levels: Optional[List[int]] = None,
    rt_range: Optional[tuple] = None,
    max_spectra: int = 0,
    skip_chromatograms: bool = False,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> MSExperiment:
    """
    Load an mzML file.

    Args:
        filename: Path to mzML file
        ms_levels: Only load spectra at these MS levels (None = all)
        rt_range: Only load spectra in (min, max) RT range
        max_spectra: Maximum number of spectra to load (0 = unlimited)
        skip_chromatograms: Skip loading chromatograms
        progress_callback: Callback(current, total) -> continue

    Returns:
        Loaded MSExperiment

    Example:
        >>> exp = load_mzml("sample.mzML")
        >>> exp = load_mzml("sample.mzML", ms_levels=[1], rt_range=(60, 300))
    """
    from .exceptions import FileFormatError
    from .validation import validate_file_path

    validate_file_path(filename, must_exist=True, allowed_extensions=[".mzML", ".mzml"])

    logger.info("Loading mzML file: %s", filename)
    exp = MSExperiment()
    exp.source_file = filename

    # Parse XML
    try:
        tree = ET.parse(filename)
    except ET.ParseError as e:
        raise FileFormatError(filename, f"Invalid XML: {e}") from e
    root = tree.getroot()

    # Handle namespace
    ns = {}
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0][1:]
        ns = {"mzml": ns_uri}

    # Find mzML element (may be inside indexedmzML)
    mzml = _find_first(
        root.find(".//mzml:mzML", ns),
        root.find(".//mzML", ns),
        root,
    )

    # Find run element
    run = _find_first(
        mzml.find("mzml:run", ns),
        mzml.find("run", ns),
        mzml.find(".//run"),
    )
    if run is None:
        run = mzml

    # Parse spectra
    spectrum_list = _find_first(
        run.find("mzml:spectrumList", ns),
        run.find("spectrumList", ns),
        run.find(".//spectrumList"),
    )

    if spectrum_list is not None:
        total = int(spectrum_list.get("count", 0))
        loaded = 0

        for spec_elem in _findall_any(spectrum_list, "mzml:spectrum", "spectrum", ns):
            if max_spectra > 0 and loaded >= max_spectra:
                break

            spec = _parse_spectrum(spec_elem, ns)

            # Apply filters
            if ms_levels is not None and spec.ms_level not in ms_levels:
                continue
            if rt_range is not None:
                if spec.rt < rt_range[0] or spec.rt > rt_range[1]:
                    continue

            exp.add_spectrum(spec)
            loaded += 1

            if progress_callback:
                if not progress_callback(loaded, total):
                    break

    logger.debug("Loaded %d spectra from mzML", exp.spectrum_count)

    # Parse chromatograms
    if not skip_chromatograms:
        chrom_list = _find_first(
            run.find("mzml:chromatogramList", ns),
            run.find("chromatogramList", ns),
            run.find(".//chromatogramList"),
        )

        if chrom_list is not None:
            for chrom_elem in _findall_any(chrom_list, "mzml:chromatogram", "chromatogram", ns):
                chrom = _parse_chromatogram(chrom_elem, ns)
                exp.add_chromatogram(chrom)

    return exp


def _parse_spectrum(elem: ET.Element, ns: Dict[str, str]) -> Spectrum:
    """
    Parse a spectrum element.

    Supports both mzML 1.1.x (scanList, binaryDataArrayList wrappers) and
    mzML 0.99.x (spectrumDescription wrapper, direct binaryDataArray children,
    msLevel as spectrum attribute, and often-mislabeled binary precision).
    """
    spec = Spectrum()
    spec.native_id = elem.get("id", "")

    # mzML 0.99: msLevel is an attribute of <spectrum>
    if elem.get("msLevel"):
        try:
            spec.ms_level = int(elem.get("msLevel"))
        except ValueError:
            pass

    # Collect cvParams from the spectrum itself AND from spectrumDescription (0.99)
    cvparams: List[ET.Element] = list(_findall_any(elem, "mzml:cvParam", "cvParam", ns))
    spec_desc = _find_first(
        elem.find("mzml:spectrumDescription", ns),
        elem.find("spectrumDescription"),
    )
    if spec_desc is not None:
        cvparams.extend(_findall_any(spec_desc, "mzml:cvParam", "cvParam", ns))

    for cv in cvparams:
        acc = cv.get("accession", "")
        value = cv.get("value", "")

        if acc == CV_MS_LEVEL:
            try:
                spec.ms_level = int(value)
            except ValueError:
                pass
        elif acc == CV_PROFILE:
            spec.spectrum_type = SpectrumType.PROFILE
        elif acc == CV_CENTROID:
            spec.spectrum_type = SpectrumType.CENTROID
        elif acc == CV_POSITIVE:
            spec.polarity = Polarity.POSITIVE
        elif acc == CV_NEGATIVE:
            spec.polarity = Polarity.NEGATIVE

    # Parse scan — may be inside scanList (1.1) or spectrumDescription (0.99)
    scan = None
    scan_list = _find_first(
        elem.find("mzml:scanList", ns),
        elem.find("scanList"),
    )
    if scan_list is not None:
        scan = _find_first(scan_list.find("mzml:scan", ns), scan_list.find("scan"))
    if scan is None and spec_desc is not None:
        scan = _find_first(
            spec_desc.find("mzml:scan", ns),
            spec_desc.find("scan"),
        )

    if scan is not None:
        for cv in _findall_any(scan, "mzml:cvParam", "cvParam", ns):
            if cv.get("accession") == CV_SCAN_TIME:
                try:
                    rt = float(cv.get("value", 0))
                except ValueError:
                    continue
                unit = cv.get("unitAccession", "")
                unit_name = (cv.get("unitName") or "").lower()
                if unit == CV_MINUTE or unit == CV_MINUTE_MZML099 or unit_name == "minute":
                    rt *= 60.0
                spec.rt = rt

    # Parse precursors — same structure in 0.99 and 1.1
    prec_list = _find_first(
        elem.find("mzml:precursorList", ns),
        elem.find("precursorList"),
    )
    if prec_list is not None:
        for prec_elem in _findall_any(prec_list, "mzml:precursor", "precursor", ns):
            prec = _parse_precursor(prec_elem, ns)
            spec.precursors.append(prec)

    # Parse binary data:
    #   mzML 1.1: spectrum > binaryDataArrayList > binaryDataArray
    #   mzML 0.99: spectrum > binaryDataArray (direct children)
    binary_list = _find_first(
        elem.find("mzml:binaryDataArrayList", ns),
        elem.find("binaryDataArrayList"),
    )
    if binary_list is not None:
        binary_arrays = _findall_any(
            binary_list, "mzml:binaryDataArray", "binaryDataArray", ns
        )
    else:
        binary_arrays = _findall_any(
            elem, "mzml:binaryDataArray", "binaryDataArray", ns
        )

    mz_data: List[float] = []
    intensity_data: List[float] = []

    for binary in binary_arrays:
        # arrayLength lets us auto-detect the actual precision if the CV
        # terms are wrong (common in mzML 0.99 files)
        array_length = 0
        try:
            array_length = int(binary.get("arrayLength", "0"))
        except ValueError:
            array_length = 0

        is_mz = False
        is_intensity = False
        is_64bit = True
        is_compressed = False

        for cv in _findall_any(binary, "mzml:cvParam", "cvParam", ns):
            acc = cv.get("accession", "")
            if acc == CV_MZ_ARRAY:
                is_mz = True
            elif acc == CV_INTENSITY_ARRAY:
                is_intensity = True
            elif acc == CV_FLOAT64:
                is_64bit = True
            elif acc == CV_FLOAT32:
                is_64bit = False
            elif acc == CV_ZLIB:
                is_compressed = True

        data_elem = _find_first(binary.find("mzml:binary", ns), binary.find("binary"))
        if data_elem is None or not data_elem.text:
            continue

        # Decode base64 + (maybe) zlib, then auto-detect precision
        raw = "".join(data_elem.text.split())
        if not raw:
            continue
        try:
            decoded_bytes = base64.b64decode(raw)
        except (ValueError, binascii.Error):
            continue
        if is_compressed:
            try:
                decoded_bytes = zlib.decompress(decoded_bytes)
            except zlib.error:
                continue

        is_64bit = _auto_detect_precision(len(decoded_bytes), array_length, is_64bit)
        values = _decode_binary_robust(decoded_bytes, is_64bit, array_length)
        if not values:
            continue

        if is_mz:
            mz_data = values
        elif is_intensity:
            intensity_data = values

    if mz_data:
        if not intensity_data:
            intensity_data = [0.0] * len(mz_data)
        # Atomic assignment via re-construction to avoid inconsistent cache
        if len(mz_data) == len(intensity_data):
            saved_precursors = spec.precursors
            saved_rt = spec.rt
            saved_ms = spec.ms_level
            saved_id = spec.native_id
            saved_type = spec.spectrum_type
            saved_polarity = spec.polarity
            spec = Spectrum(
                mz=mz_data,
                intensity=intensity_data,
                ms_level=saved_ms,
                rt=saved_rt,
                spectrum_type=saved_type,
                polarity=saved_polarity,
            )
            spec.native_id = saved_id
            spec.precursors = saved_precursors

    return spec


def _parse_precursor(elem: ET.Element, ns: Dict[str, str]) -> Precursor:
    """Parse a precursor element."""
    prec = Precursor()

    # Parse isolation window
    isolation = _find_first(elem.find("mzml:isolationWindow", ns), elem.find("isolationWindow"))
    if isolation is not None:
        for cv in _findall_any(isolation, "mzml:cvParam", "cvParam", ns):
            acc = cv.get("accession", "")
            value = float(cv.get("value", 0))
            if "1000827" in acc:  # target
                prec.mz = value
            elif "1000828" in acc:  # lower offset
                prec.isolation_window_lower = value
            elif "1000829" in acc:  # upper offset
                prec.isolation_window_upper = value

    # Parse selected ion
    selected_list = _find_first(elem.find("mzml:selectedIonList", ns), elem.find("selectedIonList"))
    if selected_list is not None:
        selected = _find_first(selected_list.find("mzml:selectedIon", ns), selected_list.find("selectedIon"))
        if selected is not None:
            for cv in _findall_any(selected, "mzml:cvParam", "cvParam", ns):
                acc = cv.get("accession", "")
                value = cv.get("value", "0")
                if acc == CV_SELECTED_MZ:
                    prec.mz = float(value)
                elif acc == CV_CHARGE:
                    prec.charge = int(value)
                elif "1000042" in acc:  # intensity
                    prec.intensity = float(value)

    # Parse activation
    activation = _find_first(elem.find("mzml:activation", ns), elem.find("activation"))
    if activation is not None:
        for cv in _findall_any(activation, "mzml:cvParam", "cvParam", ns):
            acc = cv.get("accession", "")
            if acc == CV_COLLISION_ENERGY:
                prec.collision_energy = float(cv.get("value", 0))
            elif "1000133" in acc:
                prec.activation_method = "CID"
            elif "1000422" in acc:
                prec.activation_method = "HCD"
            elif "1000598" in acc:
                prec.activation_method = "ETD"

    return prec


def _parse_chromatogram(elem: ET.Element, ns: Dict[str, str]) -> Chromatogram:
    """Parse a chromatogram element."""
    chrom = Chromatogram()
    chrom.native_id = elem.get("id", "")

    # Parse cvParams
    for cv in _findall_any(elem, "mzml:cvParam", "cvParam", ns):
        acc = cv.get("accession", "")
        if "1000235" in acc:  # TIC
            chrom.chrom_type = ChromatogramType.TIC
        elif "1000628" in acc:  # BPC
            chrom.chrom_type = ChromatogramType.BPC
        elif "1001473" in acc:  # SRM
            chrom.chrom_type = ChromatogramType.SRM

    # Parse binary data
    binary_list = _find_first(elem.find("mzml:binaryDataArrayList", ns), elem.find("binaryDataArrayList"))
    if binary_list is not None:
        rt_data = []
        intensity_data = []

        for binary in _findall_any(binary_list, "mzml:binaryDataArray", "binaryDataArray", ns):
            is_time = False
            is_intensity = False
            is_64bit = True
            is_compressed = False
            is_minutes = False

            for cv in _findall_any(binary, "mzml:cvParam", "cvParam", ns):
                acc = cv.get("accession", "")
                unit = cv.get("unitAccession", "")
                if acc == CV_TIME_ARRAY:
                    is_time = True
                elif acc == CV_INTENSITY_ARRAY:
                    is_intensity = True
                elif acc == CV_FLOAT64:
                    is_64bit = True
                elif acc == CV_FLOAT32:
                    is_64bit = False
                elif acc == CV_ZLIB:
                    is_compressed = True
                if unit == CV_MINUTE:
                    is_minutes = True

            data_elem = _find_first(binary.find("mzml:binary", ns), binary.find("binary"))
            if data_elem is not None and data_elem.text:
                values = decode_binary(data_elem.text, is_64bit, is_compressed)
                if is_time:
                    if is_minutes:
                        values = [v * 60.0 for v in values]
                    rt_data = values
                elif is_intensity:
                    intensity_data = values

        if rt_data and intensity_data:
            chrom.rt = rt_data
            chrom.intensity = intensity_data

    return chrom


def load_mzxml(
    filename: str,
    ms_levels: Optional[List[int]] = None,
    rt_range: Optional[tuple] = None,
    max_spectra: int = 0,
    progress_callback: Optional[Callable[[int, int], bool]] = None,
) -> MSExperiment:
    """
    Load an mzXML file.

    Args:
        filename: Path to mzXML file
        ms_levels: Only load spectra at these MS levels
        rt_range: Only load spectra in (min, max) RT range
        max_spectra: Maximum number of spectra to load
        progress_callback: Callback for progress updates

    Returns:
        Loaded MSExperiment
    """
    from .exceptions import FileFormatError
    from .validation import validate_file_path

    validate_file_path(filename, must_exist=True, allowed_extensions=[".mzXML", ".mzxml"])

    logger.info("Loading mzXML file: %s", filename)
    exp = MSExperiment()
    exp.source_file = filename

    try:
        tree = ET.parse(filename)
    except ET.ParseError as e:
        raise FileFormatError(filename, f"Invalid XML: {e}") from e
    root = tree.getroot()

    # Handle namespace
    ns = {}
    if root.tag.startswith("{"):
        ns_uri = root.tag.split("}")[0][1:]
        ns = {"mzxml": ns_uri}

    # Find msRun
    ms_run = _find_first(
        root.find(".//mzxml:msRun", ns),
        root.find(".//msRun"),
        root,
    )

    loaded = 0

    def parse_scans(parent):
        nonlocal loaded
        for scan in _findall_any(parent, "mzxml:scan", "scan", ns):
            if max_spectra > 0 and loaded >= max_spectra:
                return

            spec = _parse_mzxml_scan(scan, ns)

            # Apply filters
            if ms_levels is not None and spec.ms_level not in ms_levels:
                parse_scans(scan)
                continue
            if rt_range is not None:
                if spec.rt < rt_range[0] or spec.rt > rt_range[1]:
                    parse_scans(scan)
                    continue

            exp.add_spectrum(spec)
            loaded += 1

            if progress_callback:
                if not progress_callback(loaded, -1):
                    return

            # Parse nested scans (MS/MS)
            parse_scans(scan)

    parse_scans(ms_run)
    return exp


def _parse_mzxml_scan(elem: ET.Element, ns: Dict[str, str]) -> Spectrum:
    """Parse an mzXML scan element."""
    spec = Spectrum()

    spec.ms_level = int(elem.get("msLevel", 1))

    # Parse RT
    rt_str = elem.get("retentionTime", "0")
    if rt_str.startswith("PT") and rt_str.endswith("S"):
        spec.rt = float(rt_str[2:-1])
    elif rt_str.startswith("PT") and rt_str.endswith("M"):
        spec.rt = float(rt_str[2:-1]) * 60.0
    else:
        spec.rt = float(rt_str)

    # Polarity
    polarity = elem.get("polarity", "")
    if polarity in ("+", "positive"):
        spec.polarity = Polarity.POSITIVE
    elif polarity in ("-", "negative"):
        spec.polarity = Polarity.NEGATIVE

    # Centroided
    if elem.get("centroided", "0") in ("1", "true"):
        spec.spectrum_type = SpectrumType.CENTROID
    else:
        spec.spectrum_type = SpectrumType.PROFILE

    # Parse precursors
    for prec_elem in _findall_any(elem, "mzxml:precursorMz", "precursorMz", ns):
        prec = Precursor()
        prec.mz = float(prec_elem.text or 0)
        prec.intensity = float(prec_elem.get("precursorIntensity", 0))
        prec.charge = int(prec_elem.get("precursorCharge", 0))
        spec.precursors.append(prec)

    # Parse peaks
    peaks_elem = _find_first(elem.find("mzxml:peaks", ns), elem.find("peaks"))
    if peaks_elem is not None and peaks_elem.text:
        precision = int(peaks_elem.get("precision", 32))
        byte_order = peaks_elem.get("byteOrder", "network")
        compression = peaks_elem.get("compressionType", "")
        pair_order = peaks_elem.get("pairOrder", "m/z-int")

        is_64bit = (precision == 64)
        is_little_endian = (byte_order != "network")
        is_compressed = (compression == "zlib")

        values = decode_binary(peaks_elem.text, is_64bit, is_compressed, is_little_endian)

        # Values are interleaved
        mz_data = []
        intensity_data = []
        mz_first = (pair_order != "int-mz")

        for i in range(0, len(values) - 1, 2):
            if mz_first:
                mz_data.append(values[i])
                intensity_data.append(values[i + 1])
            else:
                intensity_data.append(values[i])
                mz_data.append(values[i + 1])

        spec.mz = mz_data
        spec.intensity = intensity_data

    return spec


def save_mztab(
    features: Any,
    filename: str,
    study_id: str = "lcms_study",
    run_id: str = "run_1",
) -> None:
    """
    Save features to mzTab format.

    Args:
        features: FeatureMap or similar with features
        filename: Output filename
        study_id: Study identifier
        run_id: MS run identifier
    """
    from .feature import FeatureMap
    from .validation import validate_output_path

    logger.info("Saving mzTab: %s", filename)
    validate_output_path(filename)
    with open(filename, "w") as f:
        # Metadata section
        f.write("MTD\tmzTab-version\t1.0.0\n")
        f.write("MTD\tmzTab-mode\tSummary\n")
        f.write("MTD\tmzTab-type\tQuantification\n")
        f.write(f"MTD\tdescription\t{study_id}\n")
        f.write(f"MTD\tms_run[1]-location\t{features.source_file if hasattr(features, 'source_file') else 'unknown'}\n")

        # Small molecule header
        f.write("\n")
        f.write("SMH\tSML_ID\tidentifier\tsmiles\tmass\tcharge\tretention_time\topt_peak_intensity\topt_peak_area\n")

        # Small molecule data
        for i, feat in enumerate(features):
            f.write(f"SML\t{i+1}\t")
            f.write(f"feature_{i+1}\t")
            f.write("\t")  # SMILES (empty)
            f.write(f"{feat.mz:.6f}\t")
            f.write(f"{feat.charge if feat.charge else ''}\t")
            f.write(f"{feat.rt:.2f}\t")
            f.write(f"{feat.intensity:.0f}\t")
            f.write(f"{feat.volume:.0f}\n")

    print(f"Saved {len(list(features))} features to {filename}")
