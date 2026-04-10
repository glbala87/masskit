"""Tests for the formats module (MGF, feature tables, peak lists, etc.)."""

import pytest
import os
import csv
import numpy as np

from masskit.formats import (
    load_mgf,
    save_mgf,
    save_feature_table,
    load_feature_table,
    save_peak_list,
    load_peak_list,
    load_imzml_metadata,
    save_mzidentml,
    load_identification_table,
)
from masskit.experiment import MSExperiment
from masskit.spectrum import Spectrum
from masskit.feature import Feature, FeatureMap
from masskit.peak import Peak, PeakList


class TestMGF:
    def test_load_mgf(self, mgf_file):
        exp = load_mgf(mgf_file)
        assert exp.spectrum_count == 3
        assert exp.spectra[0].ms_level == 2
        assert len(exp.spectra[0].mz) == 20

    def test_save_mgf(self, tmp_path):
        exp = MSExperiment()
        spec = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([1000.0, 2000.0, 500.0]),
            ms_level=2,
            rt=60.0,
        )
        spec.precursors = [{"mz": 500.5, "intensity": 5000.0, "charge": 2}]
        spec.native_id = "test_scan_1"
        exp.add_spectrum(spec)

        out = str(tmp_path / "out.mgf")
        n = save_mgf(out, exp, ms_level=2)
        assert n == 1

        content = open(out).read()
        assert "BEGIN IONS" in content
        assert "END IONS" in content
        assert "PEPMASS" in content
        assert "test_scan_1" in content

    def test_save_mgf_filters_ms_level(self, tmp_path):
        exp = MSExperiment()
        ms1 = Spectrum(mz=[100.0], intensity=[500.0], ms_level=1)
        ms2 = Spectrum(mz=[200.0], intensity=[1000.0], ms_level=2)
        ms2.precursors = [{"mz": 400.0}]
        exp.add_spectrum(ms1)
        exp.add_spectrum(ms2)

        out = str(tmp_path / "out.mgf")
        n = save_mgf(out, exp, ms_level=2)
        assert n == 1

    def test_round_trip_mgf(self, tmp_path):
        # Create and save
        exp = MSExperiment()
        for i in range(2):
            spec = Spectrum(
                mz=np.array([100.0, 200.0, 300.0]) + i,
                intensity=np.array([1000.0, 2000.0, 500.0]),
                ms_level=2,
                rt=60.0 + i * 5,
            )
            spec.precursors = [{"mz": 500.5 + i, "charge": 2}]
            exp.add_spectrum(spec)

        out = str(tmp_path / "rt.mgf")
        save_mgf(out, exp)

        loaded = load_mgf(out)
        assert loaded.spectrum_count == 2


class TestFeatureTable:
    def _make_feature_map(self, n=3):
        fmap = FeatureMap()
        for i in range(n):
            f = Feature()
            f.mz = 100.0 + i * 50
            f.rt = 60.0 + i * 30
            f.intensity = 1000.0 + i * 100
            f.volume = 5000.0 + i * 200
            f.charge = 2
            f.quality = 0.8
            f.rt_min = f.rt - 5
            f.rt_max = f.rt + 5
            f.mz_min = f.mz - 0.01
            f.mz_max = f.mz + 0.01
            fmap.add(f)
        return fmap

    def test_save_and_load(self, tmp_path):
        fmap = self._make_feature_map()
        out = str(tmp_path / "features.tsv")
        n = save_feature_table(out, fmap)
        assert n == 3

        loaded = load_feature_table(out)
        assert len(loaded) == 3

    def test_save_csv(self, tmp_path):
        fmap = self._make_feature_map()
        out = str(tmp_path / "features.csv")
        save_feature_table(out, fmap, delimiter=",")
        content = open(out).read()
        assert "feature_id" in content
        assert "," in content


class TestPeakList:
    def test_save_and_load(self, tmp_path):
        peaks = PeakList()
        for i in range(5):
            p = Peak(
                mz=100.0 + i * 50,
                rt=60.0 + i * 10,
                intensity=1000.0 + i * 100,
                area=2000.0,
                snr=10.0,
                fwhm_mz=0.01,
                charge=1,
            )
            peaks.add(p)

        out = str(tmp_path / "peaks.tsv")
        n = save_peak_list(out, peaks)
        assert n == 5

        loaded = load_peak_list(out)
        assert len(loaded) == 5


class TestIdentificationTable:
    def test_save_and_load(self, tmp_path):
        idents = [
            {
                "scan": 1,
                "rt": 60.0,
                "charge": 2,
                "mz": 500.5,
                "peptide": "PEPTIDE",
                "protein": "ProteinA",
                "score": 25.5,
                "q_value": 0.01,
                "modifications": "",
                "mass_error_ppm": 1.5,
            },
            {
                "scan": 2,
                "rt": 65.0,
                "charge": 3,
                "mz": 333.7,
                "peptide": "ANOTHERPEP",
                "protein": "ProteinB",
                "score": 30.2,
                "q_value": 0.005,
                "modifications": "",
                "mass_error_ppm": -0.8,
            },
        ]

        out = str(tmp_path / "idents.tsv")
        n = save_mzidentml(out, idents, search_params={"enzyme": "trypsin"})
        assert n == 2

        loaded = load_identification_table(out)
        assert len(loaded) == 2
        assert loaded[0]["peptide"] == "PEPTIDE"
        assert loaded[0]["score"] == 25.5

    def test_save_with_search_params(self, tmp_path):
        out = str(tmp_path / "out.tsv")
        save_mzidentml(out, [], search_params={"db": "human.fasta", "fdr": "0.01"})
        content = open(out).read()
        assert "# db=human.fasta" in content


class TestImzMLMetadata:
    def test_metadata_invalid_file(self, tmp_path):
        bad = tmp_path / "bad.imzML"
        bad.write_text("not xml")
        meta = load_imzml_metadata(str(bad))
        assert "error" in meta
