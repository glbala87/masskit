"""Tests for spectrum annotation module."""

import pytest
import numpy as np

from masskit.annotation import (
    IonType, NeutralLoss,
    compute_fragment_ions,
    compute_immonium_ions,
    annotate_spectrum,
    format_annotation_table,
    AA_MASSES,
)
from masskit.spectrum import Spectrum


class TestFragmentIons:
    def test_b_y_ions(self):
        fragments = compute_fragment_ions("PEPTIDE")
        assert len(fragments) > 0
        ion_types = set(f[2] for f in fragments)
        assert IonType.B in ion_types
        assert IonType.Y in ion_types

    def test_fragment_count(self):
        # 7 residues -> 6 b-ions + 6 y-ions = 12
        fragments = compute_fragment_ions(
            "PEPTIDE",
            ion_types=[IonType.B, IonType.Y],
            charge_states=[1],
            neutral_losses=[NeutralLoss.NONE],
        )
        assert len(fragments) == 12

    def test_multiple_charges(self):
        fragments_z1 = compute_fragment_ions(
            "PEPTIDE", charge_states=[1],
            neutral_losses=[NeutralLoss.NONE],
        )
        fragments_z12 = compute_fragment_ions(
            "PEPTIDE", charge_states=[1, 2],
            neutral_losses=[NeutralLoss.NONE],
        )
        assert len(fragments_z12) == 2 * len(fragments_z1)

    def test_neutral_losses(self):
        fragments = compute_fragment_ions(
            "PEPTIDE",
            neutral_losses=[NeutralLoss.NONE, NeutralLoss.WATER],
        )
        # Should have double the fragments
        base = compute_fragment_ions(
            "PEPTIDE",
            neutral_losses=[NeutralLoss.NONE],
        )
        assert len(fragments) == 2 * len(base)


class TestImmonium:
    def test_immonium_ions(self):
        ions = compute_immonium_ions("PEPTIDE")
        # Should have immonium ions for unique AAs: P, E, T, I, D
        assert len(ions) >= 4


class TestAnnotateSpectrum:
    def test_annotate(self):
        # Create a spectrum with some b/y ion peaks
        fragments = compute_fragment_ions(
            "PEPTIDE",
            ion_types=[IonType.B, IonType.Y],
            charge_states=[1],
            neutral_losses=[NeutralLoss.NONE],
        )
        # Use theoretical m/z as observed peaks
        mz_values = [f[0] for f in fragments]
        int_values = [1000.0] * len(mz_values)
        # Add noise peaks
        mz_values.extend([150.5, 250.5, 350.5])
        int_values.extend([100.0, 100.0, 100.0])

        idx = np.argsort(mz_values)
        spec = Spectrum(
            mz=np.array(mz_values)[idx],
            intensity=np.array(int_values)[idx],
        )

        result = annotate_spectrum(spec, "PEPTIDE", tolerance_da=0.05)
        assert result.n_matched > 0
        assert result.coverage > 0
        assert len(result.annotations) > 0

    def test_format_table(self):
        fragments = compute_fragment_ions("PEP", charge_states=[1],
                                          neutral_losses=[NeutralLoss.NONE])
        mz = np.array([f[0] for f in fragments])
        ints = np.ones(len(mz)) * 1000
        spec = Spectrum(mz=mz, intensity=ints)
        ann = annotate_spectrum(spec, "PEP", tolerance_da=0.05)
        table = format_annotation_table(ann)
        assert "PEP" in table
        assert "Matched" in table
