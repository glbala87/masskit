"""
Streaming and indexed file access for LC-MS data.

Provides memory-efficient random access to spectra in large mzML/mzXML files
without loading the entire file into memory.
"""

from typing import Optional, List, Iterator, Tuple, Dict, Callable
from dataclasses import dataclass, field
from pathlib import Path
import struct
import base64
import zlib
import re
import json
import logging
import numpy as np
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

from .spectrum import Spectrum, SpectrumType, Polarity


@dataclass
class SpectrumIndex:
    """Index entry for a single spectrum."""
    scan_number: int = 0
    offset: int = 0
    length: int = 0
    ms_level: int = 1
    rt: float = 0.0
    precursor_mz: float = 0.0


class FileIndex:
    """
    Standalone index for fast random access to spectra.

    Can be saved to and loaded from disk to avoid re-indexing.

    Example:
        >>> index = FileIndex.build("large_file.mzML")
        >>> index.save("large_file.mzML.idx")
        >>> # Later:
        >>> index = FileIndex.load("large_file.mzML.idx")
    """

    def __init__(self):
        self.entries: List[SpectrumIndex] = []
        self.source_file: str = ""
        self._scan_map: Dict[int, int] = {}

    @property
    def spectrum_count(self) -> int:
        return len(self.entries)

    @property
    def rt_range(self) -> Tuple[float, float]:
        if not self.entries:
            return (0.0, 0.0)
        rts = [e.rt for e in self.entries]
        return (min(rts), max(rts))

    @property
    def ms_levels(self) -> List[int]:
        return sorted(set(e.ms_level for e in self.entries))

    def add_entry(self, entry: SpectrumIndex) -> None:
        idx = len(self.entries)
        self.entries.append(entry)
        self._scan_map[entry.scan_number] = idx

    def get_by_scan(self, scan_number: int) -> Optional[SpectrumIndex]:
        idx = self._scan_map.get(scan_number)
        if idx is not None:
            return self.entries[idx]
        return None

    def get_by_index(self, index: int) -> Optional[SpectrumIndex]:
        if 0 <= index < len(self.entries):
            return self.entries[index]
        return None

    def get_by_rt(self, rt: float, tolerance: float = 0.01) -> Optional[SpectrumIndex]:
        best = None
        best_diff = float("inf")
        for entry in self.entries:
            diff = abs(entry.rt - rt)
            if diff <= tolerance and diff < best_diff:
                best = entry
                best_diff = diff
        return best

    def save(self, filepath: str) -> None:
        """Save index to file."""
        data = {
            "source_file": self.source_file,
            "entries": [
                {
                    "scan": e.scan_number,
                    "offset": e.offset,
                    "length": e.length,
                    "ms_level": e.ms_level,
                    "rt": e.rt,
                    "precursor_mz": e.precursor_mz,
                }
                for e in self.entries
            ],
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "FileIndex":
        """Load index from file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        index = cls()
        index.source_file = data.get("source_file", "")
        for entry_data in data.get("entries", []):
            entry = SpectrumIndex(
                scan_number=entry_data["scan"],
                offset=entry_data["offset"],
                length=entry_data["length"],
                ms_level=entry_data["ms_level"],
                rt=entry_data["rt"],
                precursor_mz=entry_data.get("precursor_mz", 0.0),
            )
            index.add_entry(entry)
        return index

    @classmethod
    def build(cls, filepath: str) -> "FileIndex":
        """Build index by scanning an mzML or mzXML file."""
        filepath_lower = filepath.lower()
        if filepath_lower.endswith(".mzxml"):
            return cls._build_mzxml_index(filepath)
        return cls._build_mzml_index(filepath)

    @classmethod
    def _build_mzml_index(cls, filepath: str) -> "FileIndex":
        """Build index for mzML file."""
        index = cls()
        index.source_file = filepath

        scan_number = 0
        with open(filepath, "rb") as f:
            content = f.read()

        # Find spectrum elements
        text = content.decode("utf-8", errors="replace")
        pattern = re.compile(r"<spectrum\s[^>]*>", re.DOTALL)

        for match in pattern.finditer(text):
            tag = match.group()
            offset = match.start()

            # Find end of spectrum
            end_tag = "</spectrum>"
            end_pos = text.find(end_tag, offset)
            length = (end_pos + len(end_tag) - offset) if end_pos > 0 else 0

            # Extract attributes
            scan_number += 1
            ms_level = 1
            rt = 0.0

            # Try to get index
            idx_match = re.search(r'index="(\d+)"', tag)
            if idx_match:
                scan_number = int(idx_match.group(1))

            # Try to get native ID
            id_match = re.search(r'id="([^"]*)"', tag)

            # Look for ms level and RT in the spectrum block
            if end_pos > 0:
                block = text[offset:end_pos + len(end_tag)]

                ms_match = re.search(
                    r'name="ms level"[^>]*value="(\d+)"', block
                )
                if ms_match:
                    ms_level = int(ms_match.group(1))

                rt_match = re.search(
                    r'name="scan start time"[^>]*value="([^"]+)"', block
                )
                if rt_match:
                    rt = float(rt_match.group(1))

                precursor_mz = 0.0
                pre_match = re.search(
                    r'name="selected ion m/z"[^>]*value="([^"]+)"', block
                )
                if pre_match:
                    precursor_mz = float(pre_match.group(1))

                entry = SpectrumIndex(
                    scan_number=scan_number,
                    offset=offset,
                    length=length,
                    ms_level=ms_level,
                    rt=rt,
                    precursor_mz=precursor_mz,
                )
                index.add_entry(entry)

        return index

    @classmethod
    def _build_mzxml_index(cls, filepath: str) -> "FileIndex":
        """Build index for mzXML file."""
        index = cls()
        index.source_file = filepath

        with open(filepath, "rb") as f:
            content = f.read()

        text = content.decode("utf-8", errors="replace")
        pattern = re.compile(r"<scan\s[^>]*>", re.DOTALL)

        for match in pattern.finditer(text):
            tag = match.group()
            offset = match.start()

            # Find end of scan
            end_tag = "</scan>"
            end_pos = text.find(end_tag, offset)
            length = (end_pos + len(end_tag) - offset) if end_pos > 0 else 0

            scan_number = 0
            ms_level = 1
            rt = 0.0

            num_match = re.search(r'num="(\d+)"', tag)
            if num_match:
                scan_number = int(num_match.group(1))

            level_match = re.search(r'msLevel="(\d+)"', tag)
            if level_match:
                ms_level = int(level_match.group(1))

            rt_match = re.search(r'retentionTime="PT([\d.]+)S"', tag)
            if rt_match:
                rt = float(rt_match.group(1))

            precursor_mz = 0.0
            if end_pos > 0:
                block = text[offset:end_pos + len(end_tag)]
                pre_match = re.search(
                    r"<precursorMz[^>]*>([\d.]+)</precursorMz>", block
                )
                if pre_match:
                    precursor_mz = float(pre_match.group(1))

            entry = SpectrumIndex(
                scan_number=scan_number,
                offset=offset,
                length=length,
                ms_level=ms_level,
                rt=rt,
                precursor_mz=precursor_mz,
            )
            index.add_entry(entry)

        return index


class IndexedMzMLReader:
    """
    Indexed mzML reader for random access to spectra.

    Builds an offset index on open, then reads individual spectra
    on demand without loading the entire file.

    Example:
        >>> with IndexedMzMLReader("large.mzML") as reader:
        ...     spec = reader.get_spectrum(0)
        ...     for spec in reader.iter_spectra(ms_level=1):
        ...         process(spec)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._index: Optional[FileIndex] = None
        self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self) -> None:
        """Open file and build index."""
        logger.info("Opening indexed mzML reader: %s", self.filepath)
        self._file = open(self.filepath, "rb")
        idx_path = self.filepath + ".idx"
        if Path(idx_path).exists():
            logger.debug("Loading existing index from %s", idx_path)
            self._index = FileIndex.load(idx_path)
        else:
            logger.debug("Building index for %s", self.filepath)
            self._index = FileIndex.build(self.filepath)
        logger.info("Indexed %d spectra", self._index.spectrum_count)

    def close(self) -> None:
        """Close the file."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def spectrum_count(self) -> int:
        return self._index.spectrum_count if self._index else 0

    @property
    def rt_range(self) -> Tuple[float, float]:
        return self._index.rt_range if self._index else (0.0, 0.0)

    @property
    def ms_levels(self) -> List[int]:
        return self._index.ms_levels if self._index else []

    def save_index(self, filepath: Optional[str] = None) -> None:
        """Save the index for future fast loading."""
        if self._index:
            path = filepath or (self.filepath + ".idx")
            self._index.save(path)

    def get_spectrum(self, index: int) -> Optional[Spectrum]:
        """Get spectrum by sequential index."""
        if self._index is None or self._file is None:
            return None

        entry = self._index.get_by_index(index)
        if entry is None:
            return None

        return self._read_spectrum_at(entry)

    def get_spectrum_by_scan(self, scan_number: int) -> Optional[Spectrum]:
        """Get spectrum by scan number."""
        if self._index is None or self._file is None:
            return None

        entry = self._index.get_by_scan(scan_number)
        if entry is None:
            return None

        return self._read_spectrum_at(entry)

    def get_spectrum_by_rt(
        self, rt: float, tolerance: float = 0.01
    ) -> Optional[Spectrum]:
        """Get spectrum closest to given retention time."""
        if self._index is None or self._file is None:
            return None

        entry = self._index.get_by_rt(rt, tolerance)
        if entry is None:
            return None

        return self._read_spectrum_at(entry)

    def iter_spectra(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ms_level: Optional[int] = None,
    ) -> Iterator[Spectrum]:
        """
        Lazily iterate over spectra.

        Args:
            start: Starting index (None = 0)
            end: Ending index (None = last)
            ms_level: Filter by MS level (None = all)
        """
        if self._index is None:
            return

        start = start or 0
        end = end or self._index.spectrum_count

        for i in range(start, end):
            entry = self._index.get_by_index(i)
            if entry is None:
                continue
            if ms_level is not None and entry.ms_level != ms_level:
                continue

            spec = self._read_spectrum_at(entry)
            if spec is not None:
                yield spec

    def _read_spectrum_at(self, entry: SpectrumIndex) -> Optional[Spectrum]:
        """Read and parse a spectrum at the given file offset."""
        if self._file is None or entry.length == 0:
            return None

        self._file.seek(entry.offset)
        data = self._file.read(entry.length)
        xml_str = data.decode("utf-8", errors="replace")

        try:
            # Wrap in root element for valid XML
            xml_str_clean = re.sub(r'xmlns="[^"]*"', '', xml_str)
            root = ET.fromstring(f"<root>{xml_str_clean}</root>")

            spec = Spectrum(ms_level=entry.ms_level, rt=entry.rt)
            spec.index = entry.scan_number

            # Parse binary data arrays
            mz_data = None
            int_data = None

            for binary_array in root.iter("binaryDataArray"):
                is_mz = False
                is_intensity = False
                is_64bit = False
                is_compressed = False

                for param in binary_array.iter("cvParam"):
                    # Match by accession (stable across mzML versions) with
                    # name fallback for older / non-standard files.
                    acc = param.get("accession", "")
                    name = param.get("name", "").lower()
                    if acc == "MS:1000514" or "m/z array" in name or "mzarray" in name:
                        is_mz = True
                    elif acc == "MS:1000515" or "intensity array" in name:
                        is_intensity = True
                    elif acc == "MS:1000523" or "64-bit float" in name:
                        is_64bit = True
                    elif acc == "MS:1000521" or "32-bit float" in name:
                        is_64bit = False
                    elif acc == "MS:1000574" or "zlib" in name:
                        is_compressed = True

                # Also check arrayLength for auto-detecting precision
                array_length = 0
                try:
                    array_length = int(binary_array.get("arrayLength", "0"))
                except (ValueError, TypeError):
                    pass

                binary_elem = binary_array.find("binary")
                if binary_elem is not None and binary_elem.text:
                    decoded = base64.b64decode(binary_elem.text.strip())
                    if is_compressed:
                        decoded = zlib.decompress(decoded)

                    # Auto-detect precision and endianness (handles mzML 0.99 quirks)
                    from .io import _auto_detect_precision, _decode_binary_robust
                    is_64bit = _auto_detect_precision(len(decoded), array_length, is_64bit)
                    values = np.array(
                        _decode_binary_robust(decoded, is_64bit, array_length),
                        dtype=np.float64,
                    )

                    if is_mz:
                        mz_data = values
                    elif is_intensity:
                        int_data = values

            if mz_data is not None and int_data is not None:
                if len(mz_data) != len(int_data):
                    logger.warning(
                        "spectrum %s: m/z and intensity arrays have different "
                        "lengths (%d vs %d), skipping data",
                        entry.scan_number, len(mz_data), len(int_data),
                    )
                else:
                    # Re-create spectrum atomically to avoid inconsistent cache
                    spec = Spectrum(
                        mz=mz_data,
                        intensity=int_data,
                        ms_level=entry.ms_level,
                        rt=entry.rt,
                    )
                    spec.index = entry.scan_number

            return spec

        except ET.ParseError:
            return None


class IndexedMzXMLReader:
    """
    Indexed mzXML reader for random access to spectra.

    Same interface as IndexedMzMLReader but for mzXML files.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._index: Optional[FileIndex] = None
        self._file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self) -> None:
        self._file = open(self.filepath, "rb")
        idx_path = self.filepath + ".idx"
        if Path(idx_path).exists():
            self._index = FileIndex.load(idx_path)
        else:
            self._index = FileIndex.build(self.filepath)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    @property
    def spectrum_count(self) -> int:
        return self._index.spectrum_count if self._index else 0

    @property
    def rt_range(self) -> Tuple[float, float]:
        return self._index.rt_range if self._index else (0.0, 0.0)

    @property
    def ms_levels(self) -> List[int]:
        return self._index.ms_levels if self._index else []

    def save_index(self, filepath: Optional[str] = None) -> None:
        if self._index:
            path = filepath or (self.filepath + ".idx")
            self._index.save(path)

    def get_spectrum(self, index: int) -> Optional[Spectrum]:
        if self._index is None or self._file is None:
            return None
        entry = self._index.get_by_index(index)
        if entry is None:
            return None
        return self._read_spectrum_at(entry)

    def get_spectrum_by_scan(self, scan_number: int) -> Optional[Spectrum]:
        if self._index is None or self._file is None:
            return None
        entry = self._index.get_by_scan(scan_number)
        if entry is None:
            return None
        return self._read_spectrum_at(entry)

    def get_spectrum_by_rt(
        self, rt: float, tolerance: float = 0.01
    ) -> Optional[Spectrum]:
        if self._index is None or self._file is None:
            return None
        entry = self._index.get_by_rt(rt, tolerance)
        if entry is None:
            return None
        return self._read_spectrum_at(entry)

    def iter_spectra(
        self,
        start: Optional[int] = None,
        end: Optional[int] = None,
        ms_level: Optional[int] = None,
    ) -> Iterator[Spectrum]:
        if self._index is None:
            return
        start = start or 0
        end = end or self._index.spectrum_count
        for i in range(start, end):
            entry = self._index.get_by_index(i)
            if entry is None:
                continue
            if ms_level is not None and entry.ms_level != ms_level:
                continue
            spec = self._read_spectrum_at(entry)
            if spec is not None:
                yield spec

    def _read_spectrum_at(self, entry: SpectrumIndex) -> Optional[Spectrum]:
        if self._file is None or entry.length == 0:
            return None

        self._file.seek(entry.offset)
        data = self._file.read(entry.length)
        xml_str = data.decode("utf-8", errors="replace")

        try:
            xml_str_clean = re.sub(r'xmlns="[^"]*"', '', xml_str)
            root = ET.fromstring(f"<root>{xml_str_clean}</root>")

            spec = Spectrum(ms_level=entry.ms_level, rt=entry.rt)
            spec.index = entry.scan_number

            scan = root.find(".//scan")
            if scan is None:
                return spec

            peaks_elem = scan.find("peaks")
            if peaks_elem is not None and peaks_elem.text:
                precision = int(peaks_elem.get("precision", "32"))
                byte_order = peaks_elem.get("byteOrder", "network")
                compressed = peaks_elem.get("compressionType", "none") != "none"

                decoded = base64.b64decode(peaks_elem.text.strip())
                if compressed:
                    decoded = zlib.decompress(decoded)

                fmt_char = "d" if precision == 64 else "f"
                endian = ">" if byte_order == "network" else "<"
                n_values = len(decoded) // struct.calcsize(fmt_char)
                values = struct.unpack(f"{endian}{n_values}{fmt_char}", decoded)

                # mzXML interleaves m/z and intensity
                n_peaks = n_values // 2
                mz_data = np.array(values[0::2], dtype=np.float64)
                int_data = np.array(values[1::2], dtype=np.float64)

                # Re-create spectrum atomically to avoid inconsistent cache
                spec = Spectrum(
                    mz=mz_data,
                    intensity=int_data,
                    ms_level=entry.ms_level,
                    rt=entry.rt,
                )
                spec.index = entry.scan_number

            return spec

        except ET.ParseError:
            return None


class StreamingExperiment:
    """
    MSExperiment-like interface backed by indexed file access.

    Provides the same interface as MSExperiment but loads spectra
    on demand, keeping memory usage constant regardless of file size.

    Example:
        >>> with StreamingExperiment("large.mzML") as exp:
        ...     print(f"{len(exp)} spectra")
        ...     for spec in exp.filter(ms_level=1):
        ...         process(spec)
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._reader = None

        if filepath.lower().endswith(".mzxml"):
            self._reader = IndexedMzXMLReader(filepath)
        else:
            self._reader = IndexedMzMLReader(filepath)

    def __enter__(self):
        self._reader.open()
        return self

    def __exit__(self, *args):
        self._reader.close()

    def __len__(self) -> int:
        return self._reader.spectrum_count

    def __iter__(self) -> Iterator[Spectrum]:
        return self._reader.iter_spectra()

    def spectrum(self, index: int) -> Optional[Spectrum]:
        """Get spectrum by index (on-demand loading)."""
        return self._reader.get_spectrum(index)

    def get_tic(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute TIC by streaming through all spectra.

        Returns:
            Tuple of (rt_array, intensity_array)
        """
        rts = []
        tics = []
        for spec in self._reader.iter_spectra(ms_level=1):
            rts.append(spec.rt)
            tics.append(np.sum(spec.intensity) if len(spec.intensity) > 0 else 0.0)
        return (np.array(rts), np.array(tics))

    def get_xic(
        self, mz: float, tolerance: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute XIC by streaming through all spectra.

        Args:
            mz: Target m/z
            tolerance: m/z tolerance

        Returns:
            Tuple of (rt_array, intensity_array)
        """
        rts = []
        intensities = []
        for spec in self._reader.iter_spectra(ms_level=1):
            rts.append(spec.rt)
            mask = np.abs(spec.mz - mz) <= tolerance
            intensity = np.sum(spec.intensity[mask]) if np.any(mask) else 0.0
            intensities.append(intensity)
        return (np.array(rts), np.array(intensities))

    def filter(
        self,
        ms_level: Optional[int] = None,
        rt_range: Optional[Tuple[float, float]] = None,
    ) -> Iterator[Spectrum]:
        """
        Filtered lazy iterator over spectra.

        Args:
            ms_level: Filter by MS level
            rt_range: Filter by RT range (min, max)
        """
        for spec in self._reader.iter_spectra(ms_level=ms_level):
            if rt_range is not None:
                if spec.rt < rt_range[0] or spec.rt > rt_range[1]:
                    continue
            yield spec


class ChunkedProcessor:
    """
    Process large files in chunks to limit memory usage.

    Example:
        >>> processor = ChunkedProcessor("large.mzML", chunk_size=100)
        >>> results = processor.process_chunks(analyze_chunk)
    """

    def __init__(self, filepath: str, chunk_size: int = 100):
        self.filepath = filepath
        self.chunk_size = chunk_size

    def process_chunks(
        self,
        func: Callable,
        ms_level: Optional[int] = None,
        **kwargs,
    ) -> List:
        """
        Process file in chunks.

        Args:
            func: Function that takes a list of Spectrum objects and returns a result
            ms_level: Filter by MS level
            **kwargs: Additional arguments passed to func

        Returns:
            List of results from each chunk
        """
        results = []
        chunk: List[Spectrum] = []

        with StreamingExperiment(self.filepath) as exp:
            for spec in exp.filter(ms_level=ms_level):
                chunk.append(spec)
                if len(chunk) >= self.chunk_size:
                    results.append(func(chunk, **kwargs))
                    chunk = []

            if chunk:
                results.append(func(chunk, **kwargs))

        return results
