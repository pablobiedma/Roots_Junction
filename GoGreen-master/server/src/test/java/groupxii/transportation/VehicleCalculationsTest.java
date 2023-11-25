package groupxii.transportation;


import groupxii.database.Database;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

import java.io.IOException;

public class VehicleCalculationsTest {
    @Before
    public void startDb() {
        try {
            Database.instance.startDb();
        } catch (IOException e) {
            assertTrue(false);
        }
    }
    @Test
    public void testCalculate() {
        assertEquals(0, VehicleCalculations.calculateCO2(
                "car", 5));
    }

    @Test
    public void testInstantiated() {
        VehicleCalculations cal = new VehicleCalculations();
        assertEquals(0, cal.calculateCO2(
                "car", 5));
    }
}
