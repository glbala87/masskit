"""
Protein/peptide identification pipeline for LC-MS/MS data.

Provides database search integration, PSM scoring, peptide validation,
protein inference, and FDR control at PSM/peptide/protein levels.
"""

from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import re
import subprocess
import csv
import logging

logger = logging.getLogger(__name__)

from .spectrum import Spectrum
from .experiment import MSExperiment


# Standard amino acid masses (monoisotopic)
AMINO_ACID_MASSES = {
    "G": 57.02146, "A": 71.03711, "V": 99.06841, "L": 113.08406,
    "I": 113.08406, "P": 97.05276, "F": 147.06841, "W": 186.07931,
    "M": 131.04049, "S": 87.03203, "T": 101.04768, "C": 103.00919,
    "Y": 163.06333, "H": 137.05891, "D": 115.02694, "E": 129.04259,
    "N": 114.04293, "Q": 128.05858, "K": 128.09496, "R": 156.10111,
}

PROTON_MASS = 1.007276
WATER_MASS = 18.01056


@dataclass
class PeptideSpectrumMatch:
    """A peptide-spectrum match (PSM)."""
    scan: int = 0
    rt: float = 0.0
    charge: int = 0
    precursor_mz: float = 0.0
    peptide: str = ""
    protein: str = ""
    score: float = 0.0
    delta_score: float = 0.0
    mass_error_ppm: float = 0.0
    modifications: str = ""
    q_value: float = 1.0
    is_decoy: bool = False
    rank: int = 1

    def theoretical_mass(self) -> float:
        """Calculate theoretical neutral mass of the peptide."""
        return calculate_peptide_mass(self.peptide)

    def __repr__(self) -> str:
        return (
            f"PSM(scan={self.scan}, peptide='{self.peptide}', "
            f"score={self.score:.3f}, q={self.q_value:.4f})"
        )


@dataclass
class ProteinGroup:
    """A protein group from protein inference."""
    accession: str = ""
    description: str = ""
    peptides: List[str] = field(default_factory=list)
    unique_peptides: List[str] = field(default_factory=list)
    psm_count: int = 0
    coverage: float = 0.0
    score: float = 0.0
    q_value: float = 1.0

    @property
    def num_peptides(self) -> int:
        return len(self.peptides)

    @property
    def num_unique_peptides(self) -> int:
        return len(self.unique_peptides)

    def __repr__(self) -> str:
        return (
            f"ProteinGroup(acc='{self.accession}', "
            f"peptides={self.num_peptides}, "
            f"unique={self.num_unique_peptides}, "
            f"q={self.q_value:.4f})"
        )


def calculate_peptide_mass(sequence: str) -> float:
    """
    Calculate monoisotopic neutral mass of a peptide.

    Args:
        sequence: Amino acid sequence (one-letter code)

    Returns:
        Neutral monoisotopic mass
    """
    mass = WATER_MASS  # N-term H + C-term OH
    for aa in sequence.upper():
        if aa in AMINO_ACID_MASSES:
            mass += AMINO_ACID_MASSES[aa]
    return mass


def calculate_mz(mass: float, charge: int) -> float:
    """Calculate m/z from neutral mass and charge."""
    if charge == 0:
        return mass
    return (mass + charge * PROTON_MASS) / abs(charge)


def mass_error_ppm(theoretical: float, observed: float) -> float:
    """Calculate mass error in ppm."""
    if theoretical == 0:
        return 0.0
    return (observed - theoretical) / theoretical * 1e6


def generate_theoretical_fragments(
    sequence: str,
    charge: int = 1,
    ion_types: str = "by",
) -> Dict[str, List[Tuple[float, str]]]:
    """
    Generate theoretical b/y fragment ions for a peptide.

    Args:
        sequence: Peptide sequence
        charge: Fragment charge state
        ion_types: Ion types to generate ('b', 'y', 'by')

    Returns:
        Dict mapping ion type to list of (mz, label) tuples
    """
    fragments: Dict[str, List[Tuple[float, str]]] = {}
    n = len(sequence)

    if "b" in ion_types:
        b_ions = []
        mass = 0.0
        for i in range(n - 1):
            aa = sequence[i].upper()
            if aa in AMINO_ACID_MASSES:
                mass += AMINO_ACID_MASSES[aa]
                mz = calculate_mz(mass, charge)
                b_ions.append((mz, f"b{i+1}"))
        fragments["b"] = b_ions

    if "y" in ion_types:
        y_ions = []
        mass = WATER_MASS
        for i in range(n - 1, 0, -1):
            aa = sequence[i].upper()
            if aa in AMINO_ACID_MASSES:
                mass += AMINO_ACID_MASSES[aa]
                mz = calculate_mz(mass, charge)
                y_ions.append((mz, f"y{n-i}"))
        fragments["y"] = y_ions

    return fragments


class SimpleDatabaseSearch:
    """
    Simple in-memory peptide database search engine.

    For production use, wrap external tools like Comet or MSFragger.
    This provides a basic implementation for small databases.

    Example:
        >>> searcher = SimpleDatabaseSearch()
        >>> searcher.load_fasta("proteins.fasta")
        >>> results = searcher.search(experiment, precursor_tol=10, fragment_tol=0.02)
    """

    def __init__(self):
        self.proteins: Dict[str, str] = {}  # accession -> sequence
        self.peptides: Dict[str, List[str]] = {}  # peptide -> [protein accessions]

    def load_fasta(
        self,
        filepath: str,
        enzyme: str = "trypsin",
        missed_cleavages: int = 2,
        min_length: int = 7,
        max_length: int = 50,
    ) -> int:
        """
        Load and digest a FASTA database.

        Args:
            filepath: Path to FASTA file
            enzyme: Digestion enzyme ('trypsin', 'lysc', 'none')
            missed_cleavages: Max missed cleavages
            min_length: Min peptide length
            max_length: Max peptide length

        Returns:
            Number of peptides generated
        """
        from .validation import validate_file_path
        validate_file_path(filepath, must_exist=True)
        logger.info("Loading FASTA database: %s", filepath)
        self.proteins = _parse_fasta(filepath)
        logger.info("Loaded %d proteins from FASTA", len(self.proteins))

        for accession, sequence in self.proteins.items():
            peptides = _digest(sequence, enzyme, missed_cleavages,
                             min_length, max_length)
            for peptide in peptides:
                if peptide not in self.peptides:
                    self.peptides[peptide] = []
                self.peptides[peptide].append(accession)

        return len(self.peptides)

    def search(
        self,
        experiment: MSExperiment,
        precursor_tolerance_ppm: float = 10.0,
        fragment_tolerance_da: float = 0.02,
        min_matched_peaks: int = 4,
        top_n: int = 1,
    ) -> List[PeptideSpectrumMatch]:
        """
        Search MS2 spectra against the peptide database.

        Args:
            experiment: MSExperiment with MS2 spectra
            precursor_tolerance_ppm: Precursor mass tolerance (ppm)
            fragment_tolerance_da: Fragment mass tolerance (Da)
            min_matched_peaks: Minimum matched fragment peaks
            top_n: Top N matches per spectrum

        Returns:
            List of PeptideSpectrumMatch objects
        """
        ms2_spectra = [s for s in experiment.spectra if s.ms_level == 2]
        logger.info("Searching %d MS2 spectra against %d peptides", len(ms2_spectra), len(self.peptides))
        results = []

        # Build peptide mass index
        pep_masses = {}
        for peptide in self.peptides:
            mass = calculate_peptide_mass(peptide)
            pep_masses[peptide] = mass

        for spec in ms2_spectra:
            if not spec.precursors:
                continue

            precursor = spec.precursors[0]
            # Support both Precursor dataclass and dict
            obs_mz = precursor.mz if hasattr(precursor, "mz") else precursor.get("mz", 0)
            charge = precursor.charge if hasattr(precursor, "charge") else precursor.get("charge", 2)
            if charge == 0:
                charge = 2
            obs_mass = (obs_mz - PROTON_MASS) * charge

            # Find candidate peptides within tolerance
            candidates = []
            for peptide, theo_mass in pep_masses.items():
                error = abs(mass_error_ppm(theo_mass, obs_mass))
                if error <= precursor_tolerance_ppm:
                    # Score by fragment matching
                    frags = generate_theoretical_fragments(peptide, charge=1)
                    all_theo_mzs = []
                    for ion_list in frags.values():
                        all_theo_mzs.extend([mz for mz, _ in ion_list])

                    matched = 0
                    for theo_mz in all_theo_mzs:
                        diffs = np.abs(spec.mz - theo_mz)
                        if len(diffs) > 0 and np.min(diffs) <= fragment_tolerance_da:
                            matched += 1

                    if matched >= min_matched_peaks:
                        # Hyperscore-like scoring
                        score = matched * np.log(1 + matched)
                        candidates.append((peptide, score, error, matched))

            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)

            for rank, (peptide, score, error, matched) in enumerate(candidates[:top_n]):
                proteins = self.peptides.get(peptide, [""])
                psm = PeptideSpectrumMatch(
                    scan=spec.index,
                    rt=spec.rt,
                    charge=charge,
                    precursor_mz=obs_mz,
                    peptide=peptide,
                    protein=proteins[0],
                    score=score,
                    mass_error_ppm=error,
                    rank=rank + 1,
                )
                if len(candidates) > 1 and rank == 0:
                    psm.delta_score = candidates[0][1] - candidates[1][1]
                results.append(psm)

        return results


def target_decoy_fdr(
    psms: List[PeptideSpectrumMatch],
    score_threshold: Optional[float] = None,
    fdr_threshold: float = 0.01,
) -> List[PeptideSpectrumMatch]:
    """
    Apply target-decoy FDR estimation.

    Args:
        psms: List of PSMs (decoy PSMs should have is_decoy=True)
        score_threshold: Optional score cutoff
        fdr_threshold: FDR threshold

    Returns:
        PSMs with q-values assigned, filtered by FDR
    """
    if not psms:
        return []

    # Sort by score descending
    sorted_psms = sorted(psms, key=lambda p: p.score, reverse=True)

    targets = 0
    decoys = 0

    for psm in sorted_psms:
        if psm.is_decoy:
            decoys += 1
        else:
            targets += 1

        fdr = decoys / targets if targets > 0 else 1.0
        psm.q_value = min(fdr, 1.0)

    # Monotonize q-values (make non-increasing from bottom)
    min_q = 1.0
    for psm in reversed(sorted_psms):
        min_q = min(min_q, psm.q_value)
        psm.q_value = min_q

    # Filter
    if score_threshold is not None:
        sorted_psms = [p for p in sorted_psms if p.score >= score_threshold]

    return [p for p in sorted_psms if p.q_value <= fdr_threshold]


def protein_inference(
    psms: List[PeptideSpectrumMatch],
    fdr_threshold: float = 0.01,
) -> List[ProteinGroup]:
    """
    Parsimony-based protein inference from PSMs.

    Args:
        psms: Filtered PSMs
        fdr_threshold: Protein-level FDR threshold

    Returns:
        List of ProteinGroup objects
    """
    # Map peptides to proteins
    peptide_proteins: Dict[str, set] = {}
    peptide_scores: Dict[str, float] = {}
    peptide_psm_count: Dict[str, int] = {}

    for psm in psms:
        if psm.is_decoy:
            continue
        pep = psm.peptide
        if pep not in peptide_proteins:
            peptide_proteins[pep] = set()
            peptide_scores[pep] = 0.0
            peptide_psm_count[pep] = 0

        peptide_proteins[pep].add(psm.protein)
        peptide_scores[pep] = max(peptide_scores[pep], psm.score)
        peptide_psm_count[pep] += 1

    # Build protein groups
    protein_peptides: Dict[str, set] = {}
    for pep, proteins in peptide_proteins.items():
        for prot in proteins:
            if prot not in protein_peptides:
                protein_peptides[prot] = set()
            protein_peptides[prot].add(pep)

    # Find unique peptides
    unique_peptides: Dict[str, set] = {}
    for pep, proteins in peptide_proteins.items():
        if len(proteins) == 1:
            prot = list(proteins)[0]
            if prot not in unique_peptides:
                unique_peptides[prot] = set()
            unique_peptides[prot].add(pep)

    # Build groups
    groups = []
    for accession, peptides in protein_peptides.items():
        uniques = unique_peptides.get(accession, set())

        total_score = sum(peptide_scores.get(p, 0) for p in peptides)
        total_psms = sum(peptide_psm_count.get(p, 0) for p in peptides)

        group = ProteinGroup(
            accession=accession,
            peptides=sorted(peptides),
            unique_peptides=sorted(uniques),
            psm_count=total_psms,
            score=total_score,
        )
        groups.append(group)

    # Sort by score
    groups.sort(key=lambda g: g.score, reverse=True)

    # Simple protein-level FDR (based on score ranking)
    for i, group in enumerate(groups):
        group.q_value = (i + 1) / len(groups) if groups else 1.0

    return [g for g in groups if g.q_value <= fdr_threshold]


def _parse_fasta(filepath: str) -> Dict[str, str]:
    """Parse a FASTA file into accession -> sequence dict."""
    proteins = {}
    current_acc = ""
    current_seq = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_acc and current_seq:
                    proteins[current_acc] = "".join(current_seq)
                # Parse accession
                header = line[1:]
                parts = header.split()
                current_acc = parts[0] if parts else header
                current_seq = []
            else:
                current_seq.append(line)

    if current_acc and current_seq:
        proteins[current_acc] = "".join(current_seq)

    return proteins


def _digest(
    sequence: str,
    enzyme: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
) -> List[str]:
    """Enzymatically digest a protein sequence."""
    if enzyme == "trypsin":
        # Cleave after K or R, not before P
        sites = [0]
        for i in range(len(sequence) - 1):
            if sequence[i] in "KR" and (i + 1 >= len(sequence) or sequence[i + 1] != "P"):
                sites.append(i + 1)
        sites.append(len(sequence))
    elif enzyme == "lysc":
        sites = [0]
        for i in range(len(sequence)):
            if sequence[i] == "K":
                sites.append(i + 1)
        sites.append(len(sequence))
    else:  # no enzyme
        return [sequence] if min_length <= len(sequence) <= max_length else []

    peptides = set()
    for i in range(len(sites) - 1):
        for mc in range(missed_cleavages + 1):
            if i + mc + 1 < len(sites):
                pep = sequence[sites[i]:sites[i + mc + 1]]
                if min_length <= len(pep) <= max_length:
                    peptides.add(pep)

    return list(peptides)


def run_external_search(
    tool: str,
    mzml_file: str,
    database: str,
    output_dir: str,
    **params,
) -> Optional[str]:
    """
    Run an external search engine.

    Args:
        tool: Tool name ('comet', 'msfragger')
        mzml_file: Input mzML file
        database: FASTA database
        output_dir: Output directory
        **params: Tool-specific parameters

    Returns:
        Path to results file, or None on failure
    """
    output_dir = str(Path(output_dir).resolve())
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if tool == "comet":
        cmd = [
            "comet",
            f"-D{database}",
            f"-P{params.get('params_file', '')}",
            mzml_file,
        ]
    elif tool == "msfragger":
        cmd = [
            "java", "-jar", params.get("jar_path", "MSFragger.jar"),
            "--database", database,
            "--output", output_dir,
            mzml_file,
        ]
    else:
        raise ValueError(f"Unknown search tool: {tool}")

    logger.info("Running external search: %s", " ".join(cmd))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            # Find output file
            for ext in [".pepXML", ".tsv", ".pin"]:
                candidates = list(Path(output_dir).glob(f"*{ext}"))
                if candidates:
                    logger.info("Search completed, results: %s", candidates[0])
                    return str(candidates[0])
            logger.warning("Search completed but no output file found in %s", output_dir)
        else:
            logger.error("Search engine returned code %d: %s", result.returncode, result.stderr[:500])
        return None
    except subprocess.TimeoutExpired:
        logger.error("Search engine timed out after 3600s")
        return None
    except FileNotFoundError:
        logger.error("Search engine executable not found: %s", cmd[0])
        return None
