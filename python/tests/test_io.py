"""Tests for I/O module."""

import pytest
import numpy as np
import tempfile
import os

from masskit.io import decode_binary, load_mzml, load_mzxml
from masskit.exceptions import FileFormatError


class TestDecodeBinary:
    def test_empty_data(self):
        result = decode_binary("")
        assert result == []

    def test_whitespace_only(self):
        result = decode_binary("   \n  ")
        assert result == []

    def test_decode_64bit(self):
        import base64
        import struct
        # Encode two float64 values
        values = [100.5, 200.5]
        binary = struct.pack("<2d", *values)
        encoded = base64.b64encode(binary).decode("ascii")
        result = decode_binary(encoded, is_64bit=True, is_compressed=False)
        assert len(result) == 2
        assert abs(result[0] - 100.5) < 0.001
        assert abs(result[1] - 200.5) < 0.001

    def test_decode_32bit(self):
        import base64
        import struct
        values = [100.5, 200.5]
        binary = struct.pack("<2f", *values)
        encoded = base64.b64encode(binary).decode("ascii")
        result = decode_binary(encoded, is_64bit=False, is_compressed=False)
        assert len(result) == 2
        assert abs(result[0] - 100.5) < 0.1

    def test_decode_compressed(self):
        import base64
        import struct
        import zlib
        values = [100.5, 200.5, 300.5]
        binary = struct.pack("<3d", *values)
        compressed = zlib.compress(binary)
        encoded = base64.b64encode(compressed).decode("ascii")
        result = decode_binary(encoded, is_64bit=True, is_compressed=True)
        assert len(result) == 3
        assert abs(result[2] - 300.5) < 0.001


class TestLoadMzML:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_mzml("/nonexistent/file.mzML")

    def test_invalid_xml(self):
        with tempfile.NamedTemporaryFile(suffix=".mzML", mode="w", delete=False) as f:
            f.write("not valid xml <<<<")
            f.flush()
            try:
                with pytest.raises(FileFormatError):
                    load_mzml(f.name)
            finally:
                os.unlink(f.name)

    def test_minimal_mzml(self):
        """Test loading a minimal valid mzML structure."""
        mzml_content = """<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml">
  <run>
    <spectrumList count="0">
    </spectrumList>
  </run>
</mzML>"""
        with tempfile.NamedTemporaryFile(suffix=".mzML", mode="w", delete=False) as f:
            f.write(mzml_content)
            f.flush()
            try:
                exp = load_mzml(f.name)
                assert exp.spectrum_count == 0
            finally:
                os.unlink(f.name)


class TestLoadMzXML:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_mzxml("/nonexistent/file.mzXML")

    def test_invalid_xml(self):
        with tempfile.NamedTemporaryFile(suffix=".mzXML", mode="w", delete=False) as f:
            f.write("not valid xml <<<<")
            f.flush()
            try:
                with pytest.raises(FileFormatError):
                    load_mzxml(f.name)
            finally:
                os.unlink(f.name)
