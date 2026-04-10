package org.lcms;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import org.lcms.core.IsotopePattern;

class IsotopePatternTest {

    @Test
    void testAveragineDistribution() {
        double[] dist = IsotopePattern.averagineDistribution(1000.0, 5);
        assertEquals(5, dist.length);
        assertTrue(dist[0] > 0);
        // First peak should be most intense for small masses
        assertTrue(dist[0] >= dist[4]);
    }

    @Test
    void testAveragineDistributionLargeMass() {
        double[] dist = IsotopePattern.averagineDistribution(5000.0, 8);
        assertEquals(8, dist.length);
        double sum = 0;
        for (double v : dist) sum += v;
        assertTrue(sum > 0.5);
    }

    @Test
    void testCosineSimilarity() {
        double[] a = {1.0, 2.0, 3.0};
        double[] b = {1.0, 2.0, 3.0};
        double sim = IsotopePattern.cosineSimilarity(a, b);
        assertEquals(1.0, sim, 0.01);
    }

    @Test
    void testCosineSimilarityOrthogonal() {
        double[] a = {1.0, 0.0, 0.0};
        double[] b = {0.0, 0.0, 1.0};
        double sim = IsotopePattern.cosineSimilarity(a, b);
        assertEquals(0.0, sim, 0.01);
    }

    @Test
    void testGetters() {
        IsotopePattern ip = new IsotopePattern();
        ip.setMonoisotopicMz(500.0);
        ip.setCharge(2);
        ip.setScore(0.95);
        ip.setRetentionTime(60.0);

        assertEquals(500.0, ip.getMonoisotopicMz(), 1e-10);
        assertEquals(2, ip.getCharge());
        assertEquals(0.95, ip.getScore(), 1e-10);
        assertEquals(60.0, ip.getRetentionTime(), 1e-10);
    }

    @Test
    void testNeutralMass() {
        IsotopePattern ip = new IsotopePattern();
        ip.setMonoisotopicMz(500.0);
        ip.setCharge(2);
        double nm = ip.getNeutralMass();
        // neutral_mass = (mz - proton) * charge ≈ (500 - 1.007) * 2 ≈ 997.99
        assertTrue(nm > 990);
        assertTrue(nm < 1000);
    }
}
