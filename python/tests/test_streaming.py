"""Tests for the streaming/indexed reader module."""

import pytest
import os

from masskit.streaming import (
    FileIndex,
    SpectrumIndex,
    IndexedMzMLReader,
    IndexedMzXMLReader,
    StreamingExperiment,
    ChunkedProcessor,
)


class TestFileIndex:
    def test_build_mzml(self, mzml_file):
        index = FileIndex.build(mzml_file)
        assert index.spectrum_count == 10
        assert index.source_file == mzml_file

    def test_build_mzxml(self, mzxml_file):
        index = FileIndex.build(mzxml_file)
        assert index.spectrum_count == 5

    def test_save_and_load(self, mzml_file, tmp_path):
        index = FileIndex.build(mzml_file)
        idx_path = str(tmp_path / "test.idx")
        index.save(idx_path)

        loaded = FileIndex.load(idx_path)
        assert loaded.spectrum_count == index.spectrum_count

    def test_rt_range(self, mzml_file):
        index = FileIndex.build(mzml_file)
        rmin, rmax = index.rt_range
        assert rmin >= 0
        assert rmax >= rmin

    def test_ms_levels(self, mzml_file):
        index = FileIndex.build(mzml_file)
        levels = index.ms_levels
        assert 1 in levels
        assert 2 in levels

    def test_get_by_index(self, mzml_file):
        index = FileIndex.build(mzml_file)
        entry = index.get_by_index(0)
        assert entry is not None
        assert entry.ms_level in (1, 2)

    def test_get_by_index_out_of_range(self, mzml_file):
        index = FileIndex.build(mzml_file)
        assert index.get_by_index(9999) is None

    def test_get_by_rt(self, mzml_file):
        index = FileIndex.build(mzml_file)
        entry = index.get_by_rt(60.0, tolerance=1.0)
        assert entry is not None


class TestIndexedMzMLReader:
    def test_open_close(self, mzml_file):
        reader = IndexedMzMLReader(mzml_file)
        reader.open()
        assert reader.spectrum_count == 10
        reader.close()

    def test_context_manager(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            assert reader.spectrum_count == 10

    def test_get_spectrum(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            spec = reader.get_spectrum(0)
            assert spec is not None
            assert len(spec.mz) > 0

    def test_iter_spectra(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            specs = list(reader.iter_spectra())
            assert len(specs) == 10

    def test_iter_filtered_by_ms_level(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            ms1_specs = list(reader.iter_spectra(ms_level=1))
            assert all(s.ms_level == 1 for s in ms1_specs if s is not None)

    def test_save_index(self, mzml_file, tmp_path):
        idx_path = str(tmp_path / "saved.idx")
        with IndexedMzMLReader(mzml_file) as reader:
            reader.save_index(idx_path)
        assert os.path.exists(idx_path)

    def test_rt_range(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            rmin, rmax = reader.rt_range
            assert rmax >= rmin

    def test_ms_levels(self, mzml_file):
        with IndexedMzMLReader(mzml_file) as reader:
            levels = reader.ms_levels
            assert len(levels) >= 1


class TestIndexedMzXMLReader:
    def test_basic(self, mzxml_file):
        with IndexedMzXMLReader(mzxml_file) as reader:
            assert reader.spectrum_count == 5
            spec = reader.get_spectrum(0)
            assert spec is not None

    def test_iter(self, mzxml_file):
        with IndexedMzXMLReader(mzxml_file) as reader:
            specs = list(reader.iter_spectra())
            assert len(specs) > 0


class TestStreamingExperiment:
    def test_basic(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            assert len(exp) == 10
            spec = exp.spectrum(0)
            assert spec is not None

    def test_iter(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            count = sum(1 for _ in exp)
            assert count == 10

    def test_filter_by_ms_level(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            ms1 = list(exp.filter(ms_level=1))
            assert all(s.ms_level == 1 for s in ms1)

    def test_filter_by_rt(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            in_range = list(exp.filter(rt_range=(60.0, 70.0)))
            for s in in_range:
                assert 60.0 <= s.rt <= 70.0

    def test_get_tic(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            rts, tics = exp.get_tic()
            assert len(rts) == len(tics)
            assert len(rts) > 0

    def test_get_xic(self, mzml_file):
        with StreamingExperiment(mzml_file) as exp:
            rts, ints = exp.get_xic(mz=500.0, tolerance=50.0)
            assert len(rts) == len(ints)

    def test_mzxml_streaming(self, mzxml_file):
        with StreamingExperiment(mzxml_file) as exp:
            assert len(exp) == 5


class TestChunkedProcessor:
    def test_process_chunks(self, mzml_file):
        processor = ChunkedProcessor(mzml_file, chunk_size=3)
        results = processor.process_chunks(lambda chunk: len(chunk))
        # 10 spectra / 3 chunks per call = 4 chunks (3, 3, 3, 1)
        assert sum(results) == 10

    def test_filtered_chunks(self, mzml_file):
        processor = ChunkedProcessor(mzml_file, chunk_size=10)
        results = processor.process_chunks(
            lambda chunk: len(chunk),
            ms_level=1,
        )
        assert sum(results) > 0
