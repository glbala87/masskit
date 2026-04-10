"""Tests for peptide identification module."""

import pytest
import numpy as np

from masskit.identification import (
    calculate_peptide_mass,
    generate_theoretical_fragments,
    target_decoy_fdr,
    protein_inference,
    mass_error_ppm,
    calculate_mz,
    PeptideSpectrumMatch,
    ProteinGroup,
)


class TestPeptideMass:
    def test_known_mass(self):
        # Glycine (G) monoisotopic mass = 57.021464
        # Water = 18.010565
        mass = calculate_peptide_mass("G")
        assert abs(mass - (57.02146 + 18.01056)) < 0.01

    def test_longer_peptide(self):
        mass = calculate_peptide_mass("PEPTIDE")
        assert mass > 700  # Should be around 799.36

    def test_empty(self):
        mass = calculate_peptide_mass("")
        assert abs(mass - 18.01056) < 0.01  # Just water


class TestMzCalculation:
    def test_calculate_mz(self):
        mass = 1000.0
        mz = calculate_mz(mass, charge=2)
        expected = (mass + 2 * 1.007276) / 2
        assert abs(mz - expected) < 0.001

    def test_zero_charge(self):
        mz = calculate_mz(1000.0, charge=0)
        assert mz == 1000.0

    def test_mass_error_ppm(self):
        error = mass_error_ppm(1000.0, 1000.001)
        assert abs(error - 1.0) < 0.1


class TestFragments:
    def test_generate_fragments(self):
        fragments = generate_theoretical_fragments("PEPTIDE", charge=1)
        assert isinstance(fragments, dict)
        assert "b" in fragments
        assert "y" in fragments

    def test_fragment_count(self):
        # For a 7-residue peptide, expect 6 b-ions and 6 y-ions
        fragments = generate_theoretical_fragments("PEPTIDE", charge=1)
        assert len(fragments["b"]) == 6
        assert len(fragments["y"]) == 6

    def test_fragment_labels(self):
        fragments = generate_theoretical_fragments("PEP", charge=1)
        b_labels = [label for _, label in fragments["b"]]
        assert "b1" in b_labels
        assert "b2" in b_labels


class TestFDR:
    def test_target_decoy(self):
        # Create mock PSMs
        psms = []
        # 10 good target hits
        for i in range(10):
            psms.append(PeptideSpectrumMatch(
                peptide=f"PEPTIDE{i}",
                score=50.0 + i,
                is_decoy=False,
            ))
        # 2 decoy hits
        for i in range(2):
            psms.append(PeptideSpectrumMatch(
                peptide=f"DECOY{i}",
                score=20.0 + i,
                is_decoy=True,
            ))

        filtered = target_decoy_fdr(psms, fdr_threshold=0.05)
        assert len(filtered) <= len(psms)
        # All filtered should be targets at this FDR
        assert all(not p.is_decoy for p in filtered)

    def test_empty_input(self):
        assert target_decoy_fdr([]) == []

    def test_all_decoys(self):
        psms = [PeptideSpectrumMatch(peptide="D", score=10.0, is_decoy=True)]
        filtered = target_decoy_fdr(psms, fdr_threshold=0.01)
        assert len(filtered) == 0


class TestProteinInference:
    def test_basic_inference(self):
        psms = [
            PeptideSpectrumMatch(peptide="PEPTIDEK", protein="ProtA", score=50.0),
            PeptideSpectrumMatch(peptide="ANOTHERK", protein="ProtA", score=40.0),
            PeptideSpectrumMatch(peptide="UNIQUEPEPTIDE", protein="ProtB", score=30.0),
        ]
        groups = protein_inference(psms, fdr_threshold=1.0)
        assert len(groups) >= 1
        assert all(isinstance(g, ProteinGroup) for g in groups)


class TestPSMDataclass:
    def test_theoretical_mass(self):
        psm = PeptideSpectrumMatch(peptide="PEPTIDE", charge=2)
        mass = psm.theoretical_mass()
        assert mass > 700

    def test_repr(self):
        psm = PeptideSpectrumMatch(peptide="PEPTIDE", score=10.5, q_value=0.01)
        r = repr(psm)
        assert "PEPTIDE" in r
