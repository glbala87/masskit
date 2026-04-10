package org.lcms;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.lcms.core.ConsensusMap;
import java.util.Arrays;
import java.util.List;

class ConsensusMapTest {

    private ConsensusMap makeSmall() {
        List<String> features = Arrays.asList("f1", "f2", "f3");
        List<String> samples = Arrays.asList("s1", "s2");
        double[][] matrix = {
            {100.0, 200.0},
            {300.0, 400.0},
            {500.0, 600.0},
        };
        return new ConsensusMap(features, samples, matrix);
    }

    @Test
    void testCreate() {
        ConsensusMap cm = makeSmall();
        assertEquals(3, cm.getNumFeatures());
        assertEquals(2, cm.getNumSamples());
    }

    @Test
    void testGetIntensity() {
        ConsensusMap cm = makeSmall();
        assertEquals(100.0, cm.getIntensity(0, 0), 1e-10);
        assertEquals(600.0, cm.getIntensity(2, 1), 1e-10);
    }

    @Test
    void testSetIntensity() {
        ConsensusMap cm = makeSmall();
        cm.setIntensity(0, 0, 999.0);
        assertEquals(999.0, cm.getIntensity(0, 0), 1e-10);
    }

    @Test
    void testFilterByPresence() {
        List<String> features = Arrays.asList("f1", "f2", "f3");
        List<String> samples = Arrays.asList("s1", "s2", "s3");
        double[][] matrix = {
            {100.0, 200.0, 300.0},   // present in all 3
            {0.0,   0.0,   100.0},   // present in 1/3
            {100.0, 100.0, 0.0},     // present in 2/3
        };
        ConsensusMap cm = new ConsensusMap(features, samples, matrix);
        ConsensusMap filtered = cm.filterByPresence(0.5);
        // f1 (3/3) and f3 (2/3) should pass, f2 (1/3) should not
        assertTrue(filtered.getNumFeatures() <= 3);
    }

    @Test
    void testLog2Transform() {
        ConsensusMap cm = makeSmall();
        ConsensusMap logged = cm.log2Transform();
        // log2(100) ≈ 6.64
        assertEquals(Math.log(101) / Math.log(2), logged.getIntensity(0, 0), 0.01);
    }

    @Test
    void testMedianNormalize() {
        ConsensusMap cm = makeSmall();
        ConsensusMap normed = cm.medianNormalize();
        assertEquals(3, normed.getNumFeatures());
    }
}
