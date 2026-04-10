"""Tests for isotope labeling quantification."""

import pytest
import numpy as np

from masskit.labeling import (
    LabelingStrategy,
    get_reporter_ions,
    extract_reporter_ions,
    normalize_reporter_intensities,
    compute_dimethyl_shift,
    aggregate_protein_ratios,
    isotope_correction,
    ReporterIonQuantification,
    TMT6_REPORTERS,
)
from masskit.spectrum import Spectrum


class TestReporterIons:
    def test_get_tmt6(self):
        reporters = get_reporter_ions(LabelingStrategy.TMT6)
        assert len(reporters) == 6
        assert "126" in reporters

    def test_get_tmt10(self):
        reporters = get_reporter_ions(LabelingStrategy.TMT10)
        assert len(reporters) == 10

    def test_get_itraq4(self):
        reporters = get_reporter_ions(LabelingStrategy.ITRAQ4)
        assert len(reporters) == 4


class TestExtractReporters:
    def test_extract_tmt6(self):
        # Create spectrum with TMT reporter ions
        mz_list = list(TMT6_REPORTERS.values())
        int_list = [1000, 1200, 800, 1100, 900, 1050]
        # Add some other peaks
        mz_list.extend([200.0, 300.0, 400.0])
        int_list.extend([5000, 3000, 4000])
        idx = np.argsort(mz_list)

        spec = Spectrum(
            mz=np.array(mz_list)[idx],
            intensity=np.array(int_list, dtype=float)[idx],
        )

        result = extract_reporter_ions(spec, LabelingStrategy.TMT6)
        assert len(result.channel_intensities) == 6
        assert result.total_intensity > 0


class TestNormalization:
    def test_median_normalization(self):
        quants = []
        for _ in range(10):
            q = ReporterIonQuantification(
                channel_intensities={
                    "126": np.random.lognormal(7, 0.5),
                    "127": np.random.lognormal(7, 0.5),
                    "128": np.random.lognormal(7, 0.5),
                },
                total_intensity=1000,
            )
            quants.append(q)

        normalized = normalize_reporter_intensities(quants, method="median")
        assert len(normalized) == 10


class TestDimethyl:
    def test_light_shift(self):
        shift = compute_dimethyl_shift("PEPTIDEK", "light")
        # N-term + 1 K = 2 sites
        assert shift > 0

    def test_no_lysine(self):
        shift = compute_dimethyl_shift("PEPTIDE", "light")
        # Just N-term = 1 site
        assert shift > 0


class TestProteinRatios:
    def test_aggregate(self):
        peptide_ratios = {
            "ProteinA": [1.5, 1.8, 1.6, 1.7],
            "ProteinB": [0.5, 0.6, 0.4],
        }
        result = aggregate_protein_ratios(peptide_ratios)
        assert "ProteinA" in result
        assert "ProteinB" in result
        assert result["ProteinA"]["n_peptides"] == 4


class TestIsotopeCorrection:
    def test_no_correction(self):
        intensities = {"126": 1000, "127": 1200}
        corrected = isotope_correction(intensities, correction_matrix=None)
        assert corrected == intensities

    def test_with_identity_matrix(self):
        intensities = {"126": 1000, "127": 1200}
        identity = np.eye(2)
        corrected = isotope_correction(intensities, identity)
        assert abs(corrected["126"] - 1000) < 1
        assert abs(corrected["127"] - 1200) < 1
