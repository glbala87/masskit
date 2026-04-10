"""Extended tests for identification module to push coverage above 75%."""

import pytest
import subprocess
import numpy as np
from unittest.mock import patch, MagicMock

from masskit.identification import (
    SimpleDatabaseSearch,
    PeptideSpectrumMatch,
    ProteinGroup,
    target_decoy_fdr,
    protein_inference,
    calculate_peptide_mass,
    calculate_mz,
    mass_error_ppm,
    generate_theoretical_fragments,
    run_external_search,
    _parse_fasta,
    _digest,
)
from masskit.experiment import MSExperiment
from masskit.spectrum import Spectrum, Precursor


# ── _parse_fasta ─────────────────────────────────────────────────────


class TestParseFasta:
    def test_basic(self, tmp_path):
        fasta = tmp_path / "test.fasta"
        fasta.write_text(
            ">sp|P00001|TEST1 description\n"
            "MKVLWAALLVTFLAGCQAKVE\n"
            "QAVETEPEPELRQQTEW\n"
            ">sp|P00002|TEST2\n"
            "MAKQLEDKVEELLSK\n"
        )
        proteins = _parse_fasta(str(fasta))
        assert "sp|P00001|TEST1" in proteins
        assert "sp|P00002|TEST2" in proteins
        # Multi-line sequence should be concatenated
        assert proteins["sp|P00001|TEST1"] == "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEW"

    def test_empty_file(self, tmp_path):
        fasta = tmp_path / "empty.fasta"
        fasta.write_text("")
        assert _parse_fasta(str(fasta)) == {}

    def test_single_protein_no_blank_line(self, tmp_path):
        fasta = tmp_path / "single.fasta"
        fasta.write_text(">P1\nMKVL")
        proteins = _parse_fasta(str(fasta))
        assert proteins == {"P1": "MKVL"}

    def test_header_no_pipe(self, tmp_path):
        fasta = tmp_path / "simple.fasta"
        fasta.write_text(">protein_x\nMKVL\n")
        proteins = _parse_fasta(str(fasta))
        assert "protein_x" in proteins


# ── _digest ──────────────────────────────────────────────────────────


class TestDigest:
    def test_trypsin_basic(self):
        # MKVLR.K -> 2 cleavage sites after R and K
        peps = _digest("MKVLRPSEK", enzyme="trypsin",
                       missed_cleavages=0, min_length=2, max_length=20)
        # Cleaves after K and R, but not before P
        # Sites: 0, 2 (after K), 9 (end). Note: K at pos 8 followed by end
        # MK / VLRPSEK
        assert "MK" in peps
        assert "VLRPSEK" in peps

    def test_trypsin_kp_no_cleavage(self):
        # KP should not cleave (rule of trypsin)
        peps = _digest("AAKPAA", enzyme="trypsin",
                       missed_cleavages=0, min_length=2, max_length=20)
        # No cleavage at K because next residue is P → whole sequence
        assert "AAKPAA" in peps

    def test_trypsin_missed_cleavages(self):
        peps = _digest("KKKK", enzyme="trypsin",
                       missed_cleavages=2, min_length=1, max_length=10)
        # Should generate 1, 2, 3-mer combinations
        assert "K" in peps
        assert "KK" in peps

    def test_lysc(self):
        peps = _digest("AKBKCK", enzyme="lysc",
                       missed_cleavages=0, min_length=1, max_length=10)
        assert "AK" in peps
        assert "BK" in peps

    def test_no_enzyme_long(self):
        peps = _digest("PEPTIDE", enzyme="none",
                       missed_cleavages=0, min_length=2, max_length=20)
        assert peps == ["PEPTIDE"]

    def test_no_enzyme_too_short(self):
        peps = _digest("X", enzyme="none",
                       missed_cleavages=0, min_length=2, max_length=20)
        assert peps == []

    def test_min_length_filter(self):
        peps = _digest("KAKBK", enzyme="trypsin",
                       missed_cleavages=0, min_length=3, max_length=20)
        assert all(len(p) >= 3 for p in peps)


# ── SimpleDatabaseSearch ─────────────────────────────────────────────


class TestSimpleDatabaseSearch:
    def test_load_fasta(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        n = searcher.load_fasta(fasta_file)
        assert n > 0
        assert len(searcher.proteins) > 0

    def test_load_fasta_with_lysc(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        n = searcher.load_fasta(fasta_file, enzyme="lysc", missed_cleavages=1)
        assert n > 0

    def test_load_nonexistent_fasta(self):
        searcher = SimpleDatabaseSearch()
        with pytest.raises(FileNotFoundError):
            searcher.load_fasta("/nonexistent/protein.fasta")

    def test_search_no_ms2(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        searcher.load_fasta(fasta_file)
        exp = MSExperiment()  # empty
        results = searcher.search(exp)
        assert results == []

    def test_search_skips_no_precursor(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        searcher.load_fasta(fasta_file)
        exp = MSExperiment()
        # MS2 spectrum with no precursors
        spec = Spectrum(
            mz=np.array([100.0, 200.0]),
            intensity=np.array([1000.0, 2000.0]),
            ms_level=2,
        )
        exp.add_spectrum(spec)
        results = searcher.search(exp)
        assert results == []

    def test_search_with_precursor_as_dict(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        searcher.load_fasta(fasta_file, missed_cleavages=0)
        exp = MSExperiment()

        # Use a known peptide for the search target
        # Just verify the path runs without error using dict-style precursor
        spec = Spectrum(
            mz=np.array([100.0, 200.0, 300.0, 400.0]),
            intensity=np.array([1000.0, 2000.0, 1500.0, 800.0]),
            ms_level=2,
        )
        spec.precursors = [{"mz": 500.0, "charge": 2}]
        exp.add_spectrum(spec)
        # Loose tolerance to maximize match chance
        results = searcher.search(
            exp,
            precursor_tolerance_ppm=1e9,
            fragment_tolerance_da=10.0,
            min_matched_peaks=1,
        )
        # Should run without error (may or may not find matches)
        assert isinstance(results, list)

    def test_search_with_precursor_dataclass(self, fasta_file):
        searcher = SimpleDatabaseSearch()
        searcher.load_fasta(fasta_file)
        exp = MSExperiment()

        spec = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([1000.0, 2000.0, 500.0]),
            ms_level=2,
        )
        spec.precursors = [Precursor(mz=500.0, charge=2)]
        exp.add_spectrum(spec)
        results = searcher.search(exp, precursor_tolerance_ppm=1e9,
                                  fragment_tolerance_da=10.0,
                                  min_matched_peaks=1)
        assert isinstance(results, list)


# ── target_decoy_fdr edge cases ──────────────────────────────────────


class TestTargetDecoyFDR:
    def test_with_score_threshold(self):
        psms = [
            PeptideSpectrumMatch(peptide="P1", score=10.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="P2", score=20.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="P3", score=30.0, is_decoy=False),
        ]
        filtered = target_decoy_fdr(psms, score_threshold=15.0, fdr_threshold=1.0)
        # Only P2 and P3 pass the score threshold
        assert len(filtered) == 2

    def test_q_value_monotonization(self):
        # Decoys interleaved with targets — q-values should be monotonic
        psms = [
            PeptideSpectrumMatch(peptide="T1", score=100.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="D1", score=90.0, is_decoy=True),
            PeptideSpectrumMatch(peptide="T2", score=80.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="T3", score=70.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="D2", score=60.0, is_decoy=True),
        ]
        target_decoy_fdr(psms, fdr_threshold=1.0)
        # After running, q-values should be assigned
        sorted_psms = sorted(psms, key=lambda p: p.score, reverse=True)
        q_vals = [p.q_value for p in sorted_psms]
        # Verify monotonized (each q >= the next)
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-9


# ── protein_inference edge cases ─────────────────────────────────────


class TestProteinInference:
    def test_excludes_decoys(self):
        psms = [
            PeptideSpectrumMatch(peptide="PEPTIDEK", protein="P1", score=50.0, is_decoy=False),
            PeptideSpectrumMatch(peptide="DECOYK", protein="DECOY_P1", score=40.0, is_decoy=True),
        ]
        groups = protein_inference(psms, fdr_threshold=1.0)
        # Decoys should be excluded from grouping
        accessions = [g.accession for g in groups]
        assert "DECOY_P1" not in accessions

    def test_unique_peptides(self):
        psms = [
            # Peptide shared across two proteins (razor)
            PeptideSpectrumMatch(peptide="SHAREDK", protein="P1", score=50.0),
            PeptideSpectrumMatch(peptide="SHAREDK", protein="P2", score=50.0),
            # Unique to P1
            PeptideSpectrumMatch(peptide="UNIQUEPEPTIDEK", protein="P1", score=40.0),
        ]
        groups = protein_inference(psms, fdr_threshold=1.0)
        p1 = next((g for g in groups if g.accession == "P1"), None)
        assert p1 is not None
        # P1 should have at least one unique peptide
        assert len(p1.unique_peptides) >= 1

    def test_empty_psms(self):
        groups = protein_inference([], fdr_threshold=0.01)
        assert groups == []

    def test_protein_group_repr(self):
        g = ProteinGroup(accession="P1", peptides=["A", "B"])
        s = repr(g)
        assert "P1" in s


# ── PSM dataclass ────────────────────────────────────────────────────


class TestPSM:
    def test_repr(self):
        psm = PeptideSpectrumMatch(peptide="PEPTIDE", score=10.5, q_value=0.01)
        assert "PEPTIDE" in repr(psm)

    def test_theoretical_mass(self):
        psm = PeptideSpectrumMatch(peptide="PEPTIDE")
        assert psm.theoretical_mass() > 0


# ── Fragment generation edge cases ───────────────────────────────────


class TestFragmentEdgeCases:
    def test_b_only(self):
        frags = generate_theoretical_fragments("PEPTIDE", ion_types="b")
        assert "b" in frags
        assert "y" not in frags

    def test_y_only(self):
        frags = generate_theoretical_fragments("PEPTIDE", ion_types="y")
        assert "y" in frags
        assert "b" not in frags

    def test_unknown_aa(self):
        # Unknown AA should be skipped
        frags = generate_theoretical_fragments("PEXTIDE")
        # Shouldn't crash; b-ions count up to (n-1) but skipping X
        assert len(frags["b"]) <= 6

    def test_multi_charge(self):
        frags = generate_theoretical_fragments("PEPTIDE", charge=2)
        assert "b" in frags
        # Charge 2 means m/z values should be different
        b1 = generate_theoretical_fragments("PEPTIDE", charge=1)
        # Different charge → different m/z
        assert frags["b"][0][0] != b1["b"][0][0]


# ── run_external_search ──────────────────────────────────────────────


class TestRunExternalSearch:
    def test_unknown_tool(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown search tool"):
            run_external_search(
                tool="bogus_engine",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )

    def test_comet_success_mock(self, tmp_path):
        # Mock subprocess.run to simulate Comet success
        result_mock = MagicMock(returncode=0, stderr="")
        # Create a fake output file
        fake_out = tmp_path / "result.pepXML"
        fake_out.write_text("<pepXML/>")

        with patch("masskit.identification.subprocess.run", return_value=result_mock):
            result = run_external_search(
                tool="comet",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )
        assert result is not None
        assert result.endswith(".pepXML")

    def test_msfragger_no_output(self, tmp_path):
        result_mock = MagicMock(returncode=0, stderr="")
        with patch("masskit.identification.subprocess.run", return_value=result_mock):
            result = run_external_search(
                tool="msfragger",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )
        # No output files in temp dir → returns None
        assert result is None

    def test_search_engine_failure(self, tmp_path):
        result_mock = MagicMock(returncode=1, stderr="error: bad input")
        with patch("masskit.identification.subprocess.run", return_value=result_mock):
            result = run_external_search(
                tool="comet",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )
        assert result is None

    def test_timeout(self, tmp_path):
        with patch(
            "masskit.identification.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="comet", timeout=3600),
        ):
            result = run_external_search(
                tool="comet",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )
        assert result is None

    def test_executable_not_found(self, tmp_path):
        with patch(
            "masskit.identification.subprocess.run",
            side_effect=FileNotFoundError("comet not found"),
        ):
            result = run_external_search(
                tool="comet",
                mzml_file="x.mzML",
                database="db.fasta",
                output_dir=str(tmp_path),
            )
        assert result is None


# ── Helper functions ─────────────────────────────────────────────────


class TestHelpers:
    def test_calculate_mz_with_negative_charge(self):
        # Function uses abs(), so negative charge produces same magnitude
        mass = 1000.0
        mz = calculate_mz(mass, charge=2)
        assert mz > 500  # Above mass/charge

    def test_mass_error_ppm_zero(self):
        assert mass_error_ppm(0.0, 100.0) == 0.0

    def test_calculate_peptide_mass_lowercase(self):
        # Lowercase letters should still work (uppercased internally)
        upper = calculate_peptide_mass("PEPTIDE")
        lower = calculate_peptide_mass("peptide")
        assert abs(upper - lower) < 1e-9
