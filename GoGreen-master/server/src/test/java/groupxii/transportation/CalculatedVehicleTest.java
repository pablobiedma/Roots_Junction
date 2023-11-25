package groupxii.transportation;


import groupxii.database.Database;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class CalculatedVehicleTest {

    @Before
    public void startDB() {
        try{
            Database.instance.startDb();
        } catch (IOException e) {
            assertTrue(false);
        }
    }
    @Test
    public void testCalculatedMeal() {
        CalculatedVehicle cv = new CalculatedVehicle(
                "Metro", "car",
                4, 9);
        assertEquals(0, cv.getReducedCO2());
    }

}
