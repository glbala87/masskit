package org.lcms;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.lcms.core.MSExperiment;
import org.lcms.core.Spectrum;
import org.lcms.core.Chromatogram;

class MSExperimentTest {

    private Spectrum makeSpectrum(int msLevel, double rt, int nPeaks) {
        double[] mz = new double[nPeaks];
        double[] ints = new double[nPeaks];
        for (int i = 0; i < nPeaks; i++) {
            mz[i] = 100.0 + i * 10.0;
            ints[i] = 1000.0 + i * 100.0;
        }
        Spectrum s = new Spectrum(mz, ints);
        s.setMsLevel(msLevel);
        s.setRetentionTime(rt);
        return s;
    }

    @Test
    void testEmpty() {
        MSExperiment exp = new MSExperiment();
        assertEquals(0, exp.getSpectrumCount());
        assertFalse(exp.hasSpectra());
        assertFalse(exp.hasChromatograms());
    }

    @Test
    void testAddSpectrum() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 5));
        exp.addSpectrum(makeSpectrum(2, 65.0, 3));
        assertEquals(2, exp.getSpectrumCount());
        assertTrue(exp.hasSpectra());
    }

    @Test
    void testGetSpectraByLevel() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 5));
        exp.addSpectrum(makeSpectrum(2, 65.0, 3));
        exp.addSpectrum(makeSpectrum(1, 70.0, 5));

        var ms1 = exp.getSpectraByLevel(1);
        assertEquals(2, ms1.size());

        var ms2 = exp.getSpectraByLevel(2);
        assertEquals(1, ms2.size());
    }

    @Test
    void testCountSpectraByLevel() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 5));
        exp.addSpectrum(makeSpectrum(1, 70.0, 5));
        exp.addSpectrum(makeSpectrum(2, 65.0, 3));

        assertEquals(2, exp.countSpectraByLevel(1));
        assertEquals(1, exp.countSpectraByLevel(2));
        assertEquals(0, exp.countSpectraByLevel(3));
    }

    @Test
    void testGetSpectraInRtRange() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 3));
        exp.addSpectrum(makeSpectrum(1, 70.0, 3));
        exp.addSpectrum(makeSpectrum(1, 80.0, 3));

        var inRange = exp.getSpectraInRtRange(65.0, 75.0);
        assertEquals(1, inRange.size());
        assertEquals(70.0, inRange.get(0).getRetentionTime(), 0.01);
    }

    @Test
    void testGenerateTIC() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 3));
        exp.addSpectrum(makeSpectrum(1, 70.0, 3));

        Chromatogram tic = exp.generateTIC(1);
        assertEquals(2, tic.size());
    }

    @Test
    void testGenerateBPC() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 3));
        exp.addSpectrum(makeSpectrum(1, 70.0, 3));

        Chromatogram bpc = exp.generateBPC(1);
        assertEquals(2, bpc.size());
    }

    @Test
    void testClear() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 60.0, 3));
        exp.clear();
        assertEquals(0, exp.getSpectrumCount());
        assertFalse(exp.hasSpectra());
    }

    @Test
    void testMetadata() {
        MSExperiment exp = new MSExperiment();
        exp.setSourceFile("test.mzML");
        exp.setInstrumentModel("Orbitrap");
        assertEquals("test.mzML", exp.getSourceFile());
        assertEquals("Orbitrap", exp.getInstrumentModel());
    }

    @Test
    void testAddChromatogram() {
        MSExperiment exp = new MSExperiment();
        double[] rt = {10.0, 20.0, 30.0};
        double[] ints = {100.0, 200.0, 300.0};
        Chromatogram chrom = new Chromatogram(rt, ints);
        exp.addChromatogram(chrom);
        assertEquals(1, exp.getChromatogramCount());
        assertTrue(exp.hasChromatograms());
    }

    @Test
    void testSortSpectraByRt() {
        MSExperiment exp = new MSExperiment();
        exp.addSpectrum(makeSpectrum(1, 90.0, 2));
        exp.addSpectrum(makeSpectrum(1, 30.0, 2));
        exp.addSpectrum(makeSpectrum(1, 60.0, 2));
        exp.sortSpectraByRt();
        assertEquals(30.0, exp.getSpectrum(0).getRetentionTime(), 0.01);
        assertEquals(60.0, exp.getSpectrum(1).getRetentionTime(), 0.01);
    }
}
