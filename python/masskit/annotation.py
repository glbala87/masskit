"""
Spectrum annotation module for LC-MS/MS data.

Provides b/y ion annotation, neutral loss labeling, and
fragment ion matching for peptide mass spectra.
"""

from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from .spectrum import Spectrum


class IonType(Enum):
    """Fragment ion types."""
    B = "b"
    Y = "y"
    A = "a"
    C = "c"
    X = "x"
    Z = "z"
    PRECURSOR = "precursor"
    IMMONIUM = "immonium"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class NeutralLoss(Enum):
    """Common neutral losses."""
    NONE = ("", 0.0)
    WATER = ("-H2O", 18.010565)
    AMMONIA = ("-NH3", 17.026549)
    CO = ("-CO", 27.994915)
    CO2 = ("-CO2", 43.989829)
    H3PO4 = ("-H3PO4", 97.976896)
    CH3SOH = ("-CH3SOH", 63.998285)

    def __init__(self, label: str, mass: float):
        self.label = label
        self.mass_loss = mass


# Amino acid monoisotopic masses
AA_MASSES = {
    "G": 57.021464, "A": 71.037114, "V": 99.068414, "L": 113.084064,
    "I": 113.084064, "P": 97.052764, "F": 147.068414, "W": 186.079313,
    "M": 131.040485, "S": 87.032028, "T": 101.047678, "C": 103.009185,
    "Y": 163.063329, "H": 137.058912, "D": 115.026943, "E": 129.042593,
    "N": 114.042927, "Q": 128.058578, "K": 128.094963, "R": 156.101111,
}

# Immonium ion masses (amino acid mass - CO = 27.9949)
IMMONIUM_IONS = {
    aa: mass - 27.994915 for aa, mass in AA_MASSES.items()
}

PROTON_MASS = 1.007276
WATER_MASS = 18.010565


@dataclass
class FragmentAnnotation:
    """Annotation for a single fragment peak."""
    mz_observed: float = 0.0
    mz_theoretical: float = 0.0
    intensity: float = 0.0
    ion_type: IonType = IonType.UNKNOWN
    ion_number: int = 0  # e.g., 3 for b3
    charge: int = 1
    neutral_loss: NeutralLoss = NeutralLoss.NONE
    error_da: float = 0.0
    error_ppm: float = 0.0
    label: str = ""

    def __post_init__(self):
        if not self.label and self.ion_type != IonType.UNKNOWN:
            self.label = self._make_label()

    def _make_label(self) -> str:
        label = f"{self.ion_type.value}{self.ion_number}"
        if self.neutral_loss != NeutralLoss.NONE:
            label += self.neutral_loss.label
        if self.charge > 1:
            label += f"({self.charge}+)"
        return label


@dataclass
class SpectrumAnnotation:
    """Complete annotation for a spectrum."""
    sequence: str = ""
    precursor_mz: float = 0.0
    charge: int = 0
    annotations: List[FragmentAnnotation] = field(default_factory=list)
    coverage: float = 0.0
    n_matched: int = 0
    n_total_peaks: int = 0
    ion_series_coverage: Dict[str, List[bool]] = field(default_factory=dict)

    @property
    def matched_fraction(self) -> float:
        if self.n_total_peaks == 0:
            return 0.0
        return self.n_matched / self.n_total_peaks

    def get_ions(self, ion_type: IonType) -> List[FragmentAnnotation]:
        """Get all annotations of a specific ion type."""
        return [a for a in self.annotations if a.ion_type == ion_type]

    def sequence_coverage_string(self) -> str:
        """Generate a sequence coverage visualization."""
        n = len(self.sequence)
        b_found = self.ion_series_coverage.get("b", [False] * n)
        y_found = self.ion_series_coverage.get("y", [False] * n)

        top = "  " + " ".join("─" if b else " " for b in b_found[:n-1]) + " "
        seq = "  " + " ".join(self.sequence)
        bot = "  " + " ".join("─" if y else " " for y in y_found[:n-1]) + " "

        return f"b: {top}\n   {seq}\ny: {bot}"


def compute_fragment_ions(
    sequence: str,
    ion_types: Optional[List[IonType]] = None,
    charge_states: Optional[List[int]] = None,
    neutral_losses: Optional[List[NeutralLoss]] = None,
) -> List[Tuple[float, str, IonType, int, int, NeutralLoss]]:
    """
    Compute theoretical fragment ion m/z values.

    Args:
        sequence: Peptide sequence
        ion_types: Ion types to compute (default: [B, Y])
        charge_states: Charge states (default: [1])
        neutral_losses: Neutral losses to consider (default: [NONE])

    Returns:
        List of (mz, label, ion_type, ion_number, charge, neutral_loss)
    """
    if ion_types is None:
        ion_types = [IonType.B, IonType.Y]
    if charge_states is None:
        charge_states = [1]
    if neutral_losses is None:
        neutral_losses = [NeutralLoss.NONE]

    seq = sequence.upper()
    n = len(seq)
    fragments = []

    # Cumulative residue masses from N-term
    cumulative_n = np.zeros(n)
    cumulative_n[0] = AA_MASSES.get(seq[0], 0.0)
    for i in range(1, n):
        cumulative_n[i] = cumulative_n[i-1] + AA_MASSES.get(seq[i], 0.0)

    # Cumulative residue masses from C-term
    cumulative_c = np.zeros(n)
    cumulative_c[n-1] = AA_MASSES.get(seq[n-1], 0.0)
    for i in range(n-2, -1, -1):
        cumulative_c[i] = cumulative_c[i+1] + AA_MASSES.get(seq[i], 0.0)

    for ion_type in ion_types:
        for i in range(1, n):  # Fragment position
            if ion_type == IonType.B:
                # b ions: sum of N-terminal residues
                base_mass = cumulative_n[i-1]
                ion_number = i
            elif ion_type == IonType.Y:
                # y ions: sum of C-terminal residues + H2O
                base_mass = cumulative_c[n-i] + WATER_MASS
                ion_number = i
            elif ion_type == IonType.A:
                # a ions: b - CO
                base_mass = cumulative_n[i-1] - 27.994915
                ion_number = i
            elif ion_type == IonType.C:
                # c ions: b + NH3
                base_mass = cumulative_n[i-1] + 17.026549
                ion_number = i
            elif ion_type == IonType.X:
                # x ions: y + CO - H2
                base_mass = cumulative_c[n-i] + WATER_MASS + 27.994915 - 2.01565
                ion_number = i
            elif ion_type == IonType.Z:
                # z ions: y - NH3
                base_mass = cumulative_c[n-i] + WATER_MASS - 17.026549
                ion_number = i
            else:
                continue

            for z in charge_states:
                for nl in neutral_losses:
                    mz = (base_mass - nl.mass_loss + z * PROTON_MASS) / z
                    label = f"{ion_type.value}{ion_number}"
                    if nl != NeutralLoss.NONE:
                        label += nl.label
                    if z > 1:
                        label += f"({z}+)"
                    fragments.append((mz, label, ion_type, ion_number, z, nl))

    return fragments


def compute_immonium_ions(sequence: str) -> List[Tuple[float, str]]:
    """Compute immonium ions for amino acids present in the sequence."""
    seq = set(sequence.upper())
    ions = []
    for aa in sorted(seq):
        if aa in IMMONIUM_IONS:
            ions.append((IMMONIUM_IONS[aa], f"I({aa})"))
    return ions


def annotate_spectrum(
    spectrum: Spectrum,
    sequence: str,
    precursor_charge: int = 2,
    tolerance_da: float = 0.02,
    tolerance_ppm: Optional[float] = None,
    ion_types: Optional[List[IonType]] = None,
    max_fragment_charge: Optional[int] = None,
    neutral_losses: Optional[List[NeutralLoss]] = None,
    include_immonium: bool = False,
) -> SpectrumAnnotation:
    """
    Annotate an MS2 spectrum with fragment ion assignments.

    Args:
        spectrum: MS2 spectrum to annotate
        sequence: Peptide sequence
        precursor_charge: Precursor charge state
        tolerance_da: Mass tolerance in Da (used if tolerance_ppm is None)
        tolerance_ppm: Mass tolerance in ppm (overrides Da if set)
        ion_types: Ion types to consider
        max_fragment_charge: Max fragment charge (default: precursor_charge - 1)
        neutral_losses: Neutral losses to include
        include_immonium: Include immonium ion annotation

    Returns:
        SpectrumAnnotation with matched peaks
    """
    if ion_types is None:
        ion_types = [IonType.B, IonType.Y]
    if max_fragment_charge is None:
        max_fragment_charge = max(1, precursor_charge - 1)
    if neutral_losses is None:
        neutral_losses = [NeutralLoss.NONE, NeutralLoss.WATER, NeutralLoss.AMMONIA]

    charge_states = list(range(1, max_fragment_charge + 1))

    # Compute theoretical fragments
    theoretical = compute_fragment_ions(
        sequence, ion_types, charge_states, neutral_losses
    )

    if include_immonium:
        immonium = compute_immonium_ions(sequence)
        for mz, label in immonium:
            theoretical.append((mz, label, IonType.IMMONIUM, 0, 1, NeutralLoss.NONE))

    mz_array = spectrum.mz
    int_array = spectrum.intensity
    n_peaks = len(mz_array)

    annotations = []
    matched_indices = set()
    seq_len = len(sequence)

    # Track ion series coverage
    b_coverage = [False] * (seq_len - 1)
    y_coverage = [False] * (seq_len - 1)

    for theo_mz, label, ion_type, ion_num, charge, nl in theoretical:
        if tolerance_ppm is not None:
            tol = theo_mz * tolerance_ppm * 1e-6
        else:
            tol = tolerance_da

        # Find closest peak within tolerance
        diffs = np.abs(mz_array - theo_mz)
        min_idx = np.argmin(diffs)
        min_diff = diffs[min_idx]

        if min_diff <= tol:
            error_ppm = (mz_array[min_idx] - theo_mz) / theo_mz * 1e6
            ann = FragmentAnnotation(
                mz_observed=float(mz_array[min_idx]),
                mz_theoretical=theo_mz,
                intensity=float(int_array[min_idx]),
                ion_type=ion_type,
                ion_number=ion_num,
                charge=charge,
                neutral_loss=nl,
                error_da=float(mz_array[min_idx] - theo_mz),
                error_ppm=float(error_ppm),
                label=label,
            )
            annotations.append(ann)
            matched_indices.add(int(min_idx))

            # Update coverage
            if ion_type == IonType.B and 0 < ion_num <= seq_len - 1:
                b_coverage[ion_num - 1] = True
            elif ion_type == IonType.Y and 0 < ion_num <= seq_len - 1:
                y_coverage[seq_len - 1 - ion_num] = True

    # Compute sequence coverage
    n_positions = seq_len - 1
    covered = sum(1 for b, y in zip(b_coverage, y_coverage) if b or y)
    coverage = covered / n_positions if n_positions > 0 else 0.0

    return SpectrumAnnotation(
        sequence=sequence,
        precursor_mz=getattr(spectrum, "precursor_mz", 0.0),
        charge=precursor_charge,
        annotations=annotations,
        coverage=coverage,
        n_matched=len(matched_indices),
        n_total_peaks=n_peaks,
        ion_series_coverage={
            "b": b_coverage,
            "y": y_coverage,
        },
    )


def format_annotation_table(annotation: SpectrumAnnotation) -> str:
    """
    Format spectrum annotation as a text table.

    Returns:
        Formatted string table of annotations
    """
    lines = [
        f"Spectrum Annotation: {annotation.sequence}",
        f"Precursor: m/z {annotation.precursor_mz:.4f} ({annotation.charge}+)",
        f"Matched: {annotation.n_matched}/{annotation.n_total_peaks} peaks "
        f"({annotation.matched_fraction:.1%})",
        f"Sequence Coverage: {annotation.coverage:.1%}",
        "",
        f"{'Label':<15} {'Observed':<12} {'Theoretical':<12} "
        f"{'Error(Da)':<10} {'Error(ppm)':<10} {'Intensity':<12}",
        "-" * 75,
    ]

    sorted_anns = sorted(annotation.annotations, key=lambda a: a.mz_observed)
    for ann in sorted_anns:
        lines.append(
            f"{ann.label:<15} {ann.mz_observed:<12.4f} {ann.mz_theoretical:<12.4f} "
            f"{ann.error_da:<10.4f} {ann.error_ppm:<10.1f} {ann.intensity:<12.0f}"
        )

    return "\n".join(lines)


def compute_ion_coverage(
    annotations: List[SpectrumAnnotation],
) -> Dict[str, float]:
    """
    Compute average ion series coverage across multiple annotated spectra.

    Returns:
        Dict with 'mean_coverage', 'median_coverage', 'mean_b_coverage',
        'mean_y_coverage', 'mean_matched_fraction'
    """
    if not annotations:
        return {}

    coverages = [a.coverage for a in annotations]
    matched = [a.matched_fraction for a in annotations]

    b_cov = []
    y_cov = []
    for a in annotations:
        b_series = a.ion_series_coverage.get("b", [])
        y_series = a.ion_series_coverage.get("y", [])
        if b_series:
            b_cov.append(sum(b_series) / len(b_series))
        if y_series:
            y_cov.append(sum(y_series) / len(y_series))

    return {
        "mean_coverage": float(np.mean(coverages)),
        "median_coverage": float(np.median(coverages)),
        "mean_b_coverage": float(np.mean(b_cov)) if b_cov else 0.0,
        "mean_y_coverage": float(np.mean(y_cov)) if y_cov else 0.0,
        "mean_matched_fraction": float(np.mean(matched)),
        "n_spectra": len(annotations),
    }
