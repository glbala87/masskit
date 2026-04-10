"""
Programmatic generation of valid mzML/mzXML test fixtures.

These fixtures provide small but realistic LC-MS files for integration testing
without requiring binary blobs in the repository.
"""

import base64
import struct
import zlib
from pathlib import Path
from typing import List, Tuple

import numpy as np


def encode_array(values: List[float], compress: bool = False) -> str:
    """Encode a float64 array as base64 (optionally zlib-compressed)."""
    binary = struct.pack(f"<{len(values)}d", *values)
    if compress:
        binary = zlib.compress(binary)
    return base64.b64encode(binary).decode("ascii")


def make_synthetic_spectrum(
    n_peaks: int = 30,
    mz_min: float = 100.0,
    mz_max: float = 800.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic spectrum with Gaussian peaks."""
    rng = np.random.default_rng(seed)
    mz = np.sort(rng.uniform(mz_min, mz_max, n_peaks))
    intensity = rng.lognormal(mean=8, sigma=1.5, size=n_peaks)
    return mz, intensity


def write_minimal_mzml(
    filepath: str,
    n_spectra: int = 5,
    n_peaks_per_spec: int = 30,
    include_ms2: bool = True,
    compress: bool = False,
) -> str:
    """
    Write a minimal but valid mzML file with synthetic spectra.

    Args:
        filepath: Output path
        n_spectra: Number of spectra to generate
        n_peaks_per_spec: Peaks per spectrum
        include_ms2: Whether to alternate MS1 and MS2 spectra
        compress: zlib-compress binary data

    Returns:
        Path to the written file
    """
    spectra_xml = []
    for i in range(n_spectra):
        ms_level = 2 if include_ms2 and i % 2 == 1 else 1
        rt = 60.0 + i * 5.0
        mz, intensity = make_synthetic_spectrum(n_peaks_per_spec, seed=i)

        mz_b64 = encode_array(mz.tolist(), compress=compress)
        int_b64 = encode_array(intensity.tolist(), compress=compress)
        compress_cv = (
            '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>'
            if compress
            else '<cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>'
        )

        precursor_xml = ""
        if ms_level == 2:
            precursor_xml = f'''
        <precursorList count="1">
          <precursor>
            <selectedIonList count="1">
              <selectedIon>
                <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="500.25" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
                <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
              </selectedIon>
            </selectedIonList>
            <activation>
              <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>
              <cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="35"/>
            </activation>
          </precursor>
        </precursorList>'''

        spectrum_xml = f'''    <spectrum index="{i}" id="controllerType=0 controllerNumber=1 scan={i + 1}" defaultArrayLength="{n_peaks_per_spec}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{ms_level}"/>
      <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>
      <scanList count="1">
        <scan>
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt}" unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>
        </scan>
      </scanList>{precursor_xml}
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          {compress_cv}
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="{len(int_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          {compress_cv}
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>'''
        spectra_xml.append(spectrum_xml)

    mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">
  <run id="test_run" defaultInstrumentConfigurationRef="IC1">
    <spectrumList count="{n_spectra}">
{chr(10).join(spectra_xml)}
    </spectrumList>
  </run>
</mzML>
'''
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(mzml)
    return filepath


def write_thermo_style_mzml(filepath: str, n_spectra: int = 5) -> str:
    """
    Write an mzML file mimicking Thermo-converted output (msconvert from Xcalibur RAW).

    Quirks emulated:
    - id format: "controllerType=0 controllerNumber=1 scan=N"
    - 32-bit intensity, 64-bit m/z (common Thermo default)
    - Scan time in minutes
    - Instrument config referenced in run element
    - profile + centroid spectra mixed
    """
    import numpy as _np
    spectra_xml = []
    for i in range(n_spectra):
        is_ms2 = i % 3 == 1
        ms_level = 2 if is_ms2 else 1
        is_centroid = is_ms2  # Thermo: MS2 typically centroided
        rt_min = 1.0 + i * 0.05  # in minutes
        mz, intensity = make_synthetic_spectrum(40, seed=i + 500)

        # 64-bit m/z
        mz_b64 = encode_array(mz.tolist())
        # 32-bit intensity
        int_binary = struct.pack(f"<{len(intensity)}f", *intensity.astype(_np.float32))
        int_b64 = base64.b64encode(int_binary).decode("ascii")

        spec_type_cv = (
            'centroid spectrum" value=""/><cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum'
            if is_centroid
            else 'profile spectrum" value=""/><cvParam cvRef="MS" accession="MS:1000128" name="profile spectrum'
        )

        precursor_xml = ""
        if is_ms2:
            precursor_xml = f'''
      <precursorList count="1">
        <precursor spectrumRef="controllerType=0 controllerNumber=1 scan={i}">
          <isolationWindow>
            <cvParam cvRef="MS" accession="MS:1000827" name="isolation window target m/z" value="500.25"/>
            <cvParam cvRef="MS" accession="MS:1000828" name="isolation window lower offset" value="0.5"/>
            <cvParam cvRef="MS" accession="MS:1000829" name="isolation window upper offset" value="0.5"/>
          </isolationWindow>
          <selectedIonList count="1">
            <selectedIon>
              <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="500.25"/>
              <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
              <cvParam cvRef="MS" accession="MS:1000042" name="peak intensity" value="1.5e6"/>
            </selectedIon>
          </selectedIonList>
          <activation>
            <cvParam cvRef="MS" accession="MS:1000422" name="beam-type collision-induced dissociation" value=""/>
            <cvParam cvRef="MS" accession="MS:1000045" name="collision energy" value="27.0"/>
          </activation>
        </precursor>
      </precursorList>'''

        spectrum_xml = f'''    <spectrum index="{i}" id="controllerType=0 controllerNumber=1 scan={i + 1}" defaultArrayLength="{len(mz)}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{ms_level}"/>
      <cvParam cvRef="MS" accession="MS:100012{8 if not is_centroid else 7}" name="{spec_type_cv}" value=""/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>
      <cvParam cvRef="MS" accession="MS:1000285" name="total ion current" value="{float(_np.sum(intensity)):.1f}"/>
      <scanList count="1">
        <scan instrumentConfigurationRef="IC1">
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt_min}" unitCvRef="UO" unitAccession="UO:0000031" unitName="minute"/>
          <cvParam cvRef="MS" accession="MS:1000927" name="ion injection time" value="50.0" unitAccession="UO:0000028" unitName="millisecond"/>
        </scan>
      </scanList>{precursor_xml}
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value="" unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="{len(int_b64)}">
          <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value="" unitCvRef="MS" unitAccession="MS:1000131" unitName="number of detector counts"/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>'''
        spectra_xml.append(spectrum_xml)

    mzml = f'''<?xml version="1.0" encoding="ISO-8859-1"?>
<indexedmzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.2_idx.xsd">
  <mzML xmlns="http://psi.hupo.org/ms/mzml" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://psi.hupo.org/ms/mzml http://psidev.info/files/ms/mzML/xsd/mzML1.1.0.xsd" id="thermo_test" version="1.1.0">
    <cvList count="2">
      <cv id="MS" fullName="Mass Spectrometry Ontology" version="4.1.0" URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>
      <cv id="UO" fullName="Unit Ontology" version="09:04:2014" URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/>
    </cvList>
    <fileDescription>
      <fileContent>
        <cvParam cvRef="MS" accession="MS:1000579" name="MS1 spectrum" value=""/>
        <cvParam cvRef="MS" accession="MS:1000580" name="MSn spectrum" value=""/>
      </fileContent>
    </fileDescription>
    <instrumentConfigurationList count="1">
      <instrumentConfiguration id="IC1">
        <cvParam cvRef="MS" accession="MS:1000557" name="LTQ Orbitrap Velos" value=""/>
      </instrumentConfiguration>
    </instrumentConfigurationList>
    <run id="thermo_run" defaultInstrumentConfigurationRef="IC1">
      <spectrumList count="{n_spectra}" defaultDataProcessingRef="dp1">
{chr(10).join(spectra_xml)}
      </spectrumList>
    </run>
  </mzML>
</indexedmzML>
'''
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(mzml)
    return filepath


def write_bruker_style_mzml(filepath: str, n_spectra: int = 5) -> str:
    """
    Write an mzML file mimicking Bruker-converted output (CompassXport / msconvert from .d).

    Quirks emulated:
    - id format: "spectrum=N"
    - All 64-bit
    - zlib compression
    - Scan time in seconds
    """
    import numpy as _np
    spectra_xml = []
    for i in range(n_spectra):
        ms_level = 2 if i % 2 == 1 else 1
        rt_sec = 50.0 + i * 5.0
        mz, intensity = make_synthetic_spectrum(35, seed=i + 700)

        mz_b64 = encode_array(mz.tolist(), compress=True)
        int_b64 = encode_array(intensity.tolist(), compress=True)

        precursor_xml = ""
        if ms_level == 2:
            precursor_xml = '''
      <precursorList count="1">
        <precursor>
          <selectedIonList count="1">
            <selectedIon>
              <cvParam cvRef="MS" accession="MS:1000744" name="selected ion m/z" value="450.5"/>
              <cvParam cvRef="MS" accession="MS:1000041" name="charge state" value="2"/>
            </selectedIon>
          </selectedIonList>
          <activation>
            <cvParam cvRef="MS" accession="MS:1000133" name="collision-induced dissociation" value=""/>
          </activation>
        </precursor>
      </precursorList>'''

        spectrum_xml = f'''    <spectrum index="{i}" id="spectrum={i + 1}" defaultArrayLength="{len(mz)}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="{ms_level}"/>
      <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>
      <cvParam cvRef="MS" accession="MS:1000130" name="positive scan" value=""/>
      <scanList count="1">
        <scan>
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt_sec}" unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>
        </scan>
      </scanList>{precursor_xml}
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="{len(int_b64)}">
          <cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>'''
        spectra_xml.append(spectrum_xml)

    mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="bruker_test">
  <run id="bruker_run">
    <spectrumList count="{n_spectra}">
{chr(10).join(spectra_xml)}
    </spectrumList>
  </run>
</mzML>
'''
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(mzml)
    return filepath


def write_waters_style_mzml(filepath: str, n_spectra: int = 5) -> str:
    """
    Write an mzML file mimicking Waters-converted output.

    Quirks emulated:
    - id format: "function=N process=N scan=N"
    - Negative polarity sample
    - 32-bit floats throughout
    """
    import numpy as _np
    spectra_xml = []
    for i in range(n_spectra):
        rt_sec = 30.0 + i * 6.0
        mz, intensity = make_synthetic_spectrum(25, seed=i + 900)

        mz_binary = struct.pack(f"<{len(mz)}f", *mz.astype(_np.float32))
        mz_b64 = base64.b64encode(mz_binary).decode("ascii")
        int_binary = struct.pack(f"<{len(intensity)}f", *intensity.astype(_np.float32))
        int_b64 = base64.b64encode(int_binary).decode("ascii")

        spectrum_xml = f'''    <spectrum index="{i}" id="function=1 process=0 scan={i + 1}" defaultArrayLength="{len(mz)}">
      <cvParam cvRef="MS" accession="MS:1000511" name="ms level" value="1"/>
      <cvParam cvRef="MS" accession="MS:1000127" name="centroid spectrum" value=""/>
      <cvParam cvRef="MS" accession="MS:1000129" name="negative scan" value=""/>
      <scanList count="1">
        <scan>
          <cvParam cvRef="MS" accession="MS:1000016" name="scan start time" value="{rt_sec}" unitCvRef="UO" unitAccession="UO:0000010" unitName="second"/>
        </scan>
      </scanList>
      <binaryDataArrayList count="2">
        <binaryDataArray encodedLength="{len(mz_b64)}">
          <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000514" name="m/z array" value=""/>
          <binary>{mz_b64}</binary>
        </binaryDataArray>
        <binaryDataArray encodedLength="{len(int_b64)}">
          <cvParam cvRef="MS" accession="MS:1000521" name="32-bit float" value=""/>
          <cvParam cvRef="MS" accession="MS:1000576" name="no compression" value=""/>
          <cvParam cvRef="MS" accession="MS:1000515" name="intensity array" value=""/>
          <binary>{int_b64}</binary>
        </binaryDataArray>
      </binaryDataArrayList>
    </spectrum>'''
        spectra_xml.append(spectrum_xml)

    mzml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0" id="waters_test">
  <run id="waters_run">
    <spectrumList count="{n_spectra}">
{chr(10).join(spectra_xml)}
    </spectrumList>
  </run>
</mzML>
'''
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(mzml)
    return filepath


def write_minimal_mzxml(
    filepath: str,
    n_spectra: int = 5,
    n_peaks_per_spec: int = 30,
) -> str:
    """Write a minimal but valid mzXML file with synthetic spectra."""
    scans_xml = []
    for i in range(n_spectra):
        rt = 60.0 + i * 5.0
        mz, intensity = make_synthetic_spectrum(n_peaks_per_spec, seed=i + 100)

        # mzXML interleaves m/z and intensity, big-endian by default
        interleaved: List[float] = []
        for m, it in zip(mz, intensity):
            interleaved.append(float(m))
            interleaved.append(float(it))

        binary = struct.pack(f">{len(interleaved)}d", *interleaved)
        peaks_b64 = base64.b64encode(binary).decode("ascii")

        scan_xml = f'''    <scan num="{i + 1}" msLevel="1" peaksCount="{n_peaks_per_spec}" polarity="+" retentionTime="PT{rt}S" centroided="1">
      <peaks precision="64" byteOrder="network" pairOrder="m/z-int">{peaks_b64}</peaks>
    </scan>'''
        scans_xml.append(scan_xml)

    mzxml = f'''<?xml version="1.0" encoding="utf-8"?>
<mzXML xmlns="http://sashimi.sourceforge.net/schema_revision/mzXML_3.2">
  <msRun scanCount="{n_spectra}">
{chr(10).join(scans_xml)}
  </msRun>
</mzXML>
'''
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text(mzxml)
    return filepath


def write_minimal_fasta(filepath: str, n_proteins: int = 3) -> str:
    """Write a small synthetic FASTA file."""
    proteins = [
        (">sp|P00001|TEST1_HUMAN Test protein 1",
         "MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGRFWDYLR"
         "WVQTLSEQVQEELLSSQVTQELRALMDETMKELKAYKSELEEQLTPVAEETRARLSK"),
        (">sp|P00002|TEST2_HUMAN Test protein 2",
         "MAKQLEDKVEELLSKNYHLENEVARLKKLVGERAGGKDQEELLNKLLENERLAEK"
         "GLAQTRTQAERMLLEAKLDLKHQRPRRPKDFAESLRR"),
        (">sp|DECOY_P00001|DECOY Test decoy 1",
         "RKSLRARRPVTEEAEAVPTLRDETPLKHESKEYAQLMKETMDELRLLEAKDLTVRELQRGYS"),
    ]
    lines = []
    for header, seq in proteins[:n_proteins]:
        lines.append(header)
        for j in range(0, len(seq), 60):
            lines.append(seq[j:j + 60])
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text("\n".join(lines) + "\n")
    return filepath


def write_minimal_mgf(filepath: str, n_spectra: int = 3) -> str:
    """Write a small synthetic MGF file (for spectral library tests)."""
    blocks = []
    for i in range(n_spectra):
        mz, intensity = make_synthetic_spectrum(20, seed=i + 200)
        peak_lines = "\n".join(f"{m:.4f} {it:.2f}" for m, it in zip(mz, intensity))
        block = (
            "BEGIN IONS\n"
            f"TITLE=test_compound_{i + 1}\n"
            f"PEPMASS=500.{200 + i}\n"
            f"CHARGE=2+\n"
            f"RTINSECONDS={60.0 + i * 5}\n"
            f"{peak_lines}\n"
            "END IONS\n"
        )
        blocks.append(block)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Path(filepath).write_text("\n".join(blocks))
    return filepath
