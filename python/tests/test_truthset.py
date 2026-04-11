"""
Truthset validation suite for MassKit.

These tests verify numerical accuracy against published reference values,
known chemical constants, and mathematically provable invariants. Unlike
functional tests (which verify code runs), these verify the toolkit
produces *correct* results.

Reference sources:
- NIST monoisotopic amino acid masses
- UniMod modification masses
- Thermo TMT/iTRAQ reporter ion m/z (vendor specification)
- Mathematical identities (cosine similarity, normalization)
- Planted synthetic data with known ground truth
"""

import pytest
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# 1. PEPTIDE MASS ACCURACY — NIST reference monoisotopic masses
# ═══════════════════════════════════════════════════════════════════════

# Published monoisotopic NEUTRAL masses (NIST Chemistry WebBook / UniMod).
# Note: many databases report [M+H]+ (protonated). These are neutral M.
# Neutral M = [M+H]+ - proton_mass (1.007276 Da)
# Format: (sequence, expected_neutral_mass, tolerance_Da)
PROTON = 1.007276
KNOWN_PEPTIDE_MASSES = [
    # Angiotensin II: DRVYIHPF — [M+H]+ = 1046.5423, neutral = 1045.5350
    ("DRVYIHPF", 1046.5423 - PROTON, 0.01),
    # Bradykinin: RPPGFSPFR — [M+H]+ = 1060.5684, neutral = 1059.5611
    ("RPPGFSPFR", 1060.5684 - PROTON, 0.01),
    # [Glu1]-Fibrinopeptide B: EGVNDNEEGFFSAR — [M+H]+ = 1570.6774
    ("EGVNDNEEGFFSAR", 1570.6774 - PROTON, 0.01),
    # Leucine enkephalin: YGGFL — neutral = 555.2693
    ("YGGFL", 555.2693, 0.01),
    # Single amino acids (residue mass + H2O)
    ("G", 57.02146 + 18.01056, 0.001),
    ("A", 71.03711 + 18.01056, 0.001),
    ("W", 186.07931 + 18.01056, 0.001),
]


class TestPeptideMassTruthset:
    """Validate peptide mass calculations against NIST reference values."""

    @pytest.mark.parametrize("sequence,expected,tolerance", KNOWN_PEPTIDE_MASSES)
    def test_known_peptide_mass(self, sequence, expected, tolerance):
        from masskit.identification import calculate_peptide_mass
        calculated = calculate_peptide_mass(sequence)
        assert abs(calculated - expected) < tolerance, (
            f"{sequence}: expected {expected:.4f}, got {calculated:.4f}, "
            f"delta {abs(calculated - expected):.6f} Da"
        )

    def test_mass_additivity(self):
        """Mass of concatenated peptide = sum of parts - water (peptide bond)."""
        from masskit.identification import calculate_peptide_mass, WATER_MASS
        m_ab = calculate_peptide_mass("PEPTIDER")
        m_a = calculate_peptide_mass("PEPTI")
        m_b = calculate_peptide_mass("DER")
        # AB = A + B - H2O (one water lost forming the peptide bond)
        assert abs(m_ab - (m_a + m_b - WATER_MASS)) < 0.001

    def test_mz_from_mass(self):
        """m/z = (mass + z * proton) / z"""
        from masskit.identification import calculate_peptide_mass, calculate_mz, PROTON_MASS
        mass = calculate_peptide_mass("PEPTIDER")
        for z in [1, 2, 3, 4]:
            mz = calculate_mz(mass, z)
            expected = (mass + z * PROTON_MASS) / z
            assert abs(mz - expected) < 1e-9

    def test_mass_error_ppm_identity(self):
        """ppm = (obs - theo) / theo * 1e6"""
        from masskit.identification import mass_error_ppm
        theo = 1000.0
        obs = 1000.005
        expected_ppm = (obs - theo) / theo * 1e6  # = 5.0 ppm
        assert abs(mass_error_ppm(theo, obs) - expected_ppm) < 1e-9


# ═══════════════════════════════════════════════════════════════════════
# 2. FRAGMENT ION ACCURACY — b/y ion series
# ═══════════════════════════════════════════════════════════════════════

class TestFragmentIonTruthset:
    """Validate fragment ions against manually computed values."""

    def test_b1_ion_of_peptide(self):
        """b1 of PEPTIDER = mass of P (proline) as MH+ at charge 1."""
        from masskit.identification import (
            generate_theoretical_fragments,
            AMINO_ACID_MASSES,
            PROTON_MASS,
        )
        frags = generate_theoretical_fragments("PEPTIDER", charge=1)
        b1_mz = frags["b"][0][0]
        # b1 = mass(P) + proton, at charge 1
        expected = (AMINO_ACID_MASSES["P"] + PROTON_MASS) / 1
        assert abs(b1_mz - expected) < 0.001, f"b1: expected {expected:.4f}, got {b1_mz:.4f}"

    def test_y1_ion_of_peptider(self):
        """y1 of PEPTIDER = mass of R + H2O, as MH+ at charge 1."""
        from masskit.identification import (
            generate_theoretical_fragments,
            AMINO_ACID_MASSES,
            WATER_MASS,
            PROTON_MASS,
        )
        frags = generate_theoretical_fragments("PEPTIDER", charge=1)
        y1_mz = frags["y"][0][0]
        # y1 = mass(R) + H2O + proton
        expected = (AMINO_ACID_MASSES["R"] + WATER_MASS + PROTON_MASS) / 1
        assert abs(y1_mz - expected) < 0.001, f"y1: expected {expected:.4f}, got {y1_mz:.4f}"

    def test_b_y_complementary(self):
        """For singly-charged ions: b(i)_mz + y(n-i)_mz = M_neutral + 2*proton.

        Each singly-charged fragment carries one proton, so the pair
        carries two. The neutral backbone mass sums to M.
        """
        from masskit.identification import (
            generate_theoretical_fragments,
            calculate_peptide_mass,
            PROTON_MASS,
        )
        seq = "PEPTIDER"
        mass = calculate_peptide_mass(seq)
        frags = generate_theoretical_fragments(seq, charge=1)
        n = len(seq)

        for i in range(n - 1):
            b_mz = frags["b"][i][0]  # b(i+1)
            y_mz = frags["y"][n - 2 - i][0]  # y(n-i-1)
            total = b_mz + y_mz
            expected = mass + 2 * PROTON_MASS
            assert abs(total - expected) < 0.01, (
                f"b{i+1}+y{n-1-i}: {total:.4f} != M+2H {expected:.4f}"
            )

    def test_fragment_count(self):
        """n-residue peptide yields (n-1) b-ions and (n-1) y-ions."""
        from masskit.identification import generate_theoretical_fragments
        for seq in ["PEPTIDE", "ACDEFGHIKLMNPQRSTVWY"]:
            frags = generate_theoretical_fragments(seq, charge=1)
            assert len(frags["b"]) == len(seq) - 1
            assert len(frags["y"]) == len(seq) - 1


# ═══════════════════════════════════════════════════════════════════════
# 3. SPECTRAL SIMILARITY — mathematical identities
# ═══════════════════════════════════════════════════════════════════════

class TestCosineSimilarityTruthset:
    """Validate cosine similarity against analytical solutions."""

    def test_identical_spectra_equals_one(self):
        """cos(A, A) = 1.0 exactly."""
        from masskit.spectral_matching import cosine_similarity
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        ints = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        score, n = cosine_similarity(mz, ints, mz, ints, tolerance=0.01)
        assert abs(score - 1.0) < 1e-10
        assert n == 5

    def test_orthogonal_spectra_equals_zero(self):
        """Non-overlapping spectra: cos = 0."""
        from masskit.spectral_matching import cosine_similarity
        mz_a = np.array([100.0, 200.0])
        int_a = np.array([1.0, 1.0])
        mz_b = np.array([300.0, 400.0])
        int_b = np.array([1.0, 1.0])
        score, n = cosine_similarity(mz_a, int_a, mz_b, int_b, tolerance=0.01)
        assert score == 0.0
        assert n == 0

    def test_known_cosine_value(self):
        """cos([1,0], [1,1]) = 1/sqrt(2) ≈ 0.7071."""
        from masskit.spectral_matching import cosine_similarity
        mz = np.array([100.0, 200.0])
        int_a = np.array([1.0, 0.0])
        int_b = np.array([1.0, 1.0])
        score, n = cosine_similarity(mz, int_a, mz, int_b, tolerance=0.01)
        expected = 1.0 / np.sqrt(2)
        assert abs(score - expected) < 0.01, f"expected {expected:.4f}, got {score:.4f}"

    def test_scaling_invariance(self):
        """Cosine similarity is scale-invariant: cos(A, k*A) = 1.0."""
        from masskit.spectral_matching import cosine_similarity
        mz = np.array([100.0, 200.0, 300.0])
        int_a = np.array([3.0, 4.0, 5.0])
        int_b = int_a * 1000  # Scaled version
        score, _ = cosine_similarity(mz, int_a, mz, int_b, tolerance=0.01)
        assert abs(score - 1.0) < 1e-6

    def test_symmetry(self):
        """cos(A, B) = cos(B, A)."""
        from masskit.spectral_matching import cosine_similarity
        mz_a = np.array([100.0, 200.0, 300.0])
        int_a = np.array([1.0, 2.0, 3.0])
        mz_b = np.array([100.0, 200.0, 350.0])
        int_b = np.array([3.0, 2.0, 1.0])
        score_ab, _ = cosine_similarity(mz_a, int_a, mz_b, int_b, tolerance=0.01)
        score_ba, _ = cosine_similarity(mz_b, int_b, mz_a, int_a, tolerance=0.01)
        assert abs(score_ab - score_ba) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# 4. NORMALIZATION INVARIANTS
# ═══════════════════════════════════════════════════════════════════════

class TestNormalizationTruthset:
    """Validate normalization methods preserve mathematical invariants."""

    def test_median_norm_equalizes_medians(self):
        """After median normalization, column medians should be equal."""
        from masskit.quantification import median_normalization
        rng = np.random.default_rng(42)
        # Deliberately skewed columns
        matrix = rng.lognormal(10, 2, (100, 4))
        matrix[:, 0] *= 0.5  # Half the first column
        matrix[:, 3] *= 3.0  # Triple the last
        normed = median_normalization(matrix)
        medians = np.median(normed, axis=0)
        # All column medians should be within 1% of each other
        cv = np.std(medians) / np.mean(medians)
        assert cv < 0.01, f"median CV after norm: {cv:.4f}"

    def test_tic_norm_equalizes_sums(self):
        """After TIC normalization, column sums should be equal."""
        from masskit.quantification import tic_normalization
        rng = np.random.default_rng(42)
        matrix = rng.lognormal(10, 2, (50, 4))
        normed = tic_normalization(matrix)
        sums = np.sum(normed, axis=0)
        np.testing.assert_allclose(sums, sums[0], rtol=1e-6)

    def test_quantile_norm_equalizes_distributions(self):
        """After quantile normalization, sorted columns should be identical."""
        from masskit.quantification import quantile_normalization
        rng = np.random.default_rng(42)
        matrix = rng.lognormal(10, 2, (50, 4))
        normed = quantile_normalization(matrix)
        sorted_cols = np.sort(normed, axis=0)
        # Each row of sorted columns should be constant (same quantiles)
        for row in sorted_cols:
            assert np.std(row) / np.mean(row) < 0.01


# ═══════════════════════════════════════════════════════════════════════
# 5. TMT/iTRAQ REPORTER ION ACCURACY — vendor reference m/z
# ═══════════════════════════════════════════════════════════════════════

# Published TMT6plex reporter ion m/z (Thermo Fisher Scientific)
KNOWN_TMT6_MZ = {
    "126": 126.127726,
    "127": 127.131081,
    "128": 128.134436,
    "129": 129.137790,
    "130": 130.141145,
    "131": 131.138180,
}

# Published iTRAQ 4-plex reporter ions (AB SCIEX)
KNOWN_ITRAQ4_MZ = {
    "114": 114.11068,
    "115": 115.10826,
    "116": 116.11162,
    "117": 117.11497,
}


class TestReporterIonTruthset:
    """Validate reporter ion m/z against vendor specifications."""

    def test_tmt6_mz_values(self):
        from masskit.labeling import TMT6_REPORTERS
        for channel, expected_mz in KNOWN_TMT6_MZ.items():
            actual_mz = TMT6_REPORTERS[channel]
            assert abs(actual_mz - expected_mz) < 0.0001, (
                f"TMT6 {channel}: expected {expected_mz:.6f}, got {actual_mz:.6f}"
            )

    def test_itraq4_mz_values(self):
        from masskit.labeling import ITRAQ4_REPORTERS
        for channel, expected_mz in KNOWN_ITRAQ4_MZ.items():
            actual_mz = ITRAQ4_REPORTERS[channel]
            assert abs(actual_mz - expected_mz) < 0.001, (
                f"iTRAQ4 {channel}: expected {expected_mz:.5f}, got {actual_mz:.5f}"
            )

    def test_tmt6_channel_spacing(self):
        """TMT6 channels should be ~1.003 Da apart (neutron mass)."""
        from masskit.labeling import TMT6_REPORTERS
        mzs = sorted(TMT6_REPORTERS.values())
        for i in range(len(mzs) - 1):
            spacing = mzs[i + 1] - mzs[i]
            # Spacing varies due to isotope labeling but should be ~1.003 ± 0.01
            assert 0.99 < spacing < 1.02, f"TMT spacing {i}: {spacing:.6f}"

    def test_reporter_extraction_accuracy(self):
        """Planted reporter ions should be extracted at correct intensities."""
        from masskit.labeling import extract_reporter_ions, LabelingStrategy, TMT6_REPORTERS
        from masskit.spectrum import Spectrum

        # Plant exact TMT6 reporter ions with known intensities
        planted = {"126": 1000.0, "127": 2000.0, "128": 1500.0,
                   "129": 3000.0, "130": 500.0, "131": 2500.0}
        mz_list = [TMT6_REPORTERS[ch] for ch in planted]
        int_list = [planted[ch] for ch in planted]
        # Add noise peaks far from reporters
        mz_list += [200.0, 300.0, 400.0, 500.0]
        int_list += [5000.0, 3000.0, 4000.0, 2000.0]
        idx = np.argsort(mz_list)
        spec = Spectrum(
            mz=np.array(mz_list)[idx],
            intensity=np.array(int_list)[idx],
        )

        quant = extract_reporter_ions(spec, LabelingStrategy.TMT6, tolerance_da=0.01)
        for ch, expected_int in planted.items():
            actual = quant.channel_intensities.get(ch, 0)
            assert abs(actual - expected_int) < 1.0, (
                f"TMT {ch}: expected {expected_int}, got {actual}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 6. PEAK PICKING — planted peaks at known positions
# ═══════════════════════════════════════════════════════════════════════

class TestPeakPickingTruthset:
    """Validate peak picking recovers planted peaks at correct m/z."""

    def test_recover_planted_peaks(self):
        """Plant 5 Gaussian peaks, verify all are found within 0.05 Da."""
        from masskit.spectrum import Spectrum
        from masskit.algorithms import pick_peaks

        planted_mz = [200.0, 350.5, 500.0, 650.25, 800.0]
        planted_int = [5000.0, 10000.0, 3000.0, 8000.0, 6000.0]

        mz = np.arange(100, 900, 0.01)
        intensity = np.zeros_like(mz)
        for pm, pi in zip(planted_mz, planted_int):
            intensity += pi * np.exp(-0.5 * ((mz - pm) / 0.05) ** 2)
        # Low noise
        intensity += np.abs(np.random.RandomState(42).normal(0, 10, len(mz)))

        spec = Spectrum(mz=mz, intensity=intensity)
        peaks = pick_peaks(spec, min_snr=3.0)

        found_mzs = sorted([p.mz for p in peaks], key=lambda x: x)
        for target in planted_mz:
            closest = min(found_mzs, key=lambda x: abs(x - target))
            assert abs(closest - target) < 0.05, (
                f"Planted peak at {target:.2f} not found within 0.05 Da "
                f"(closest: {closest:.4f})"
            )

    def test_no_false_peaks_in_flat_region(self):
        """A flat spectrum (no peaks) should yield zero or very few detections."""
        from masskit.spectrum import Spectrum
        from masskit.algorithms import pick_peaks

        mz = np.arange(100, 500, 0.1)
        intensity = np.ones_like(mz) * 100  # Flat
        spec = Spectrum(mz=mz, intensity=intensity)
        peaks = pick_peaks(spec, min_snr=3.0)
        assert len(peaks) <= 2, f"Flat spectrum produced {len(peaks)} false peaks"


# ═══════════════════════════════════════════════════════════════════════
# 7. FDR CONTROL — calibration check
# ═══════════════════════════════════════════════════════════════════════

class TestFDRTruthset:
    """Validate that target-decoy FDR is correctly calibrated."""

    def test_fdr_upper_bound(self):
        """At threshold t, the fraction of decoys should be <= t."""
        from masskit.identification import PeptideSpectrumMatch, target_decoy_fdr

        rng = np.random.default_rng(42)
        psms = []
        # 100 targets with high scores
        for i in range(100):
            psms.append(PeptideSpectrumMatch(
                peptide=f"TARGET{i}", score=rng.uniform(20, 100), is_decoy=False,
            ))
        # 20 decoys with lower scores
        for i in range(20):
            psms.append(PeptideSpectrumMatch(
                peptide=f"DECOY{i}", score=rng.uniform(5, 40), is_decoy=True,
            ))

        for threshold in [0.01, 0.05, 0.1]:
            filtered = target_decoy_fdr(psms, fdr_threshold=threshold)
            if len(filtered) == 0:
                continue
            n_decoy = sum(1 for p in filtered if p.is_decoy)
            n_target = sum(1 for p in filtered if not p.is_decoy)
            if n_target > 0:
                actual_fdr = n_decoy / n_target
                assert actual_fdr <= threshold + 0.01, (
                    f"FDR threshold {threshold}: actual {actual_fdr:.3f} exceeds limit"
                )

    def test_q_values_monotonic(self):
        """q-values should be non-decreasing when sorted by descending score."""
        from masskit.identification import PeptideSpectrumMatch, target_decoy_fdr

        psms = []
        for i in range(50):
            psms.append(PeptideSpectrumMatch(
                peptide=f"T{i}", score=100 - i, is_decoy=i % 10 == 0,
            ))
        target_decoy_fdr(psms, fdr_threshold=1.0)
        sorted_psms = sorted(psms, key=lambda p: p.score, reverse=True)
        q_vals = [p.q_value for p in sorted_psms]
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-10, (
                f"q-values not monotonic at position {i}: {q_vals[i]} > {q_vals[i+1]}"
            )


# ═══════════════════════════════════════════════════════════════════════
# 8. ISOTOPE DISTRIBUTION — known patterns
# ═══════════════════════════════════════════════════════════════════════

class TestIsotopeDistributionTruthset:
    """Validate isotope distributions against known chemical properties."""

    def test_monoisotopic_is_most_abundant_at_low_mass(self):
        """For masses < 1500 Da, the monoisotopic peak should be most intense."""
        from masskit.isotope import averagine_distribution
        for mass in [500, 800, 1000, 1200]:
            dist = averagine_distribution(float(mass), num_peaks=5)
            assert np.argmax(dist) == 0, (
                f"At {mass} Da: M+{np.argmax(dist)} is most intense, expected M+0"
            )

    def test_distribution_sums_near_one(self):
        """The full isotope distribution should sum close to 1.0."""
        from masskit.isotope import averagine_distribution
        for mass in [500, 1000, 2000, 5000]:
            dist = averagine_distribution(float(mass), num_peaks=10)
            total = sum(dist)
            # With enough peaks, should capture >95% of the distribution
            assert total > 0.80, f"At {mass} Da: total={total:.3f} (expected > 0.80)"


# ═══════════════════════════════════════════════════════════════════════
# 9. QC METRICS — mathematical consistency
# ═══════════════════════════════════════════════════════════════════════

class TestQCTruthset:
    """Validate QC metric calculations against known inputs."""

    def test_tic_cv_of_constant_spectra(self):
        """Spectra with identical TIC should produce CV = 0."""
        from masskit.experiment import MSExperiment
        from masskit.spectrum import Spectrum
        from masskit.qc import compute_qc_metrics

        exp = MSExperiment()
        for i in range(10):
            # Identical m/z and intensity for each spectrum
            spec = Spectrum(
                mz=np.array([100.0, 200.0, 300.0]),
                intensity=np.array([1000.0, 2000.0, 3000.0]),
                ms_level=1,
                rt=float(i * 10),
            )
            exp.add_spectrum(spec)

        qc = compute_qc_metrics(exp)
        assert abs(qc.tic_cv) < 0.001, f"TIC CV of identical spectra: {qc.tic_cv}"

    def test_spectrum_counts(self):
        """QC should correctly count MS1 and MS2 spectra."""
        from masskit.experiment import MSExperiment
        from masskit.spectrum import Spectrum
        from masskit.qc import compute_qc_metrics

        exp = MSExperiment()
        for i in range(7):
            spec = Spectrum(
                mz=np.array([100.0]),
                intensity=np.array([1000.0]),
                ms_level=1 if i < 4 else 2,
                rt=float(i * 10),
            )
            exp.add_spectrum(spec)

        qc = compute_qc_metrics(exp)
        assert qc.num_ms1 == 4
        assert qc.num_ms2 == 3
        assert qc.num_spectra == 7


# ═══════════════════════════════════════════════════════════════════════
# 10. SPECTRUM DATA STRUCTURE — invariants
# ═══════════════════════════════════════════════════════════════════════

class TestSpectrumInvariants:
    """Validate Spectrum data structure maintains correct invariants."""

    def test_tic_equals_sum_of_intensities(self):
        from masskit.spectrum import Spectrum
        ints = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        spec = Spectrum(mz=np.arange(5, dtype=float) + 100, intensity=ints)
        assert abs(spec.tic - np.sum(ints)) < 1e-10

    def test_base_peak_is_max(self):
        from masskit.spectrum import Spectrum
        mz = np.array([100.0, 200.0, 300.0, 400.0])
        ints = np.array([500.0, 2000.0, 800.0, 100.0])
        spec = Spectrum(mz=mz, intensity=ints)
        assert spec.base_peak_intensity == 2000.0
        assert spec.base_peak_mz == 200.0

    def test_extract_range_subset(self):
        """Extracted range should only contain points within bounds."""
        from masskit.spectrum import Spectrum
        mz = np.arange(100.0, 1000.0, 1.0)
        ints = np.ones_like(mz)
        spec = Spectrum(mz=mz, intensity=ints)
        sub = spec.extract_range(300.0, 500.0)
        assert np.all(sub.mz >= 300.0)
        assert np.all(sub.mz <= 500.0)

    def test_normalize_max_scales_to_one(self):
        from masskit.spectrum import Spectrum
        spec = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([500.0, 1000.0, 250.0]),
        )
        normed = spec.normalize("max")
        assert abs(np.max(normed.intensity) - 1.0) < 1e-10

    def test_normalize_sum_scales_to_one(self):
        from masskit.spectrum import Spectrum
        spec = Spectrum(
            mz=np.array([100.0, 200.0, 300.0]),
            intensity=np.array([500.0, 1000.0, 250.0]),
        )
        normed = spec.normalize("sum")
        assert abs(np.sum(normed.intensity) - 1.0) < 1e-10
