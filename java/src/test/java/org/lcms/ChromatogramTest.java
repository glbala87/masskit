package org.lcms;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.lcms.core.Chromatogram;
import org.lcms.core.ChromatogramType;

class ChromatogramTest {

    @Test
    void testEmpty() {
        Chromatogram c = new Chromatogram();
        assertEquals(0, c.size());
        assertTrue(c.isEmpty());
    }

    @Test
    void testCreateWithData() {
        double[] rt = {10.0, 20.0, 30.0, 40.0};
        double[] ints = {100.0, 500.0, 300.0, 50.0};
        Chromatogram c = new Chromatogram(rt, ints);
        assertEquals(4, c.size());
        assertFalse(c.isEmpty());
    }

    @Test
    void testMaxIntensity() {
        double[] rt = {10.0, 20.0, 30.0};
        double[] ints = {100.0, 800.0, 300.0};
        Chromatogram c = new Chromatogram(rt, ints);
        assertEquals(800.0, c.getMaxIntensity(), 1e-10);
    }

    @Test
    void testApexRt() {
        double[] rt = {10.0, 20.0, 30.0};
        double[] ints = {100.0, 800.0, 300.0};
        Chromatogram c = new Chromatogram(rt, ints);
        assertEquals(20.0, c.getApexRt(), 1e-10);
    }

    @Test
    void testComputeArea() {
        // Triangle: base=2, height=1 -> area=1
        double[] rt = {0.0, 1.0, 2.0};
        double[] ints = {0.0, 1.0, 0.0};
        Chromatogram c = new Chromatogram(rt, ints);
        assertEquals(1.0, c.computeArea(), 0.01);
    }

    @Test
    void testExtractRange() {
        double[] rt = {10.0, 20.0, 30.0, 40.0, 50.0};
        double[] ints = {1.0, 2.0, 3.0, 4.0, 5.0};
        Chromatogram c = new Chromatogram(rt, ints);
        Chromatogram sub = c.extractRange(15.0, 35.0);
        // Should include points at 20 and 30
        assertTrue(sub.size() >= 2);
    }

    @Test
    void testType() {
        Chromatogram c = new Chromatogram();
        c.setType(ChromatogramType.TIC);
        assertEquals(ChromatogramType.TIC, c.getType());
    }

    @Test
    void testSRMFields() {
        Chromatogram c = new Chromatogram();
        c.setType(ChromatogramType.SRM);
        c.setPrecursorMz(500.0);
        c.setProductMz(300.0);
        assertEquals(500.0, c.getPrecursorMz(), 1e-10);
        assertEquals(300.0, c.getProductMz(), 1e-10);
    }
}
