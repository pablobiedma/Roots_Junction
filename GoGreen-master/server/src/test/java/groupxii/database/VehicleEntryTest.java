package groupxii.database;

import com.mongodb.DBObject;
import groupxii.transportation.CalculatedVehicle;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;




public class VehicleEntryTest {

    VehicleEntry vehicleEntry;
    @Before
    public void createEntry() {
        try {
            Database.instance.startDb();
        }
        catch (IOException e) {
            assertTrue(false);
        }
        vehicleEntry = new VehicleEntry("bike","car",4,8);

    }

    @Test
    public void getGoodVehicleType() {
        // VehicleEntry ve = new VehicleEntry("bike","car",4,8);
        assertEquals("bike", vehicleEntry.getGoodVehicleType());
    }
    @Test
    public void getBadVehicleType() {
        //VehicleEntry ve = new VehicleEntry("bike","car",4,8);
        assertEquals("car", vehicleEntry.getBadvehicleType());
    }

    @Test
    public void getco2() {
        //VehicleEntry ve = new VehicleEntry("bike","car",4,8);
        CalculatedVehicle calculatedVehicle = new CalculatedVehicle("bike", "car", 4, 8);
        assertEquals(calculatedVehicle.getReducedCO2(), vehicleEntry.getReducedco2());
    }
    @Test
    public void getGoodAvgConsumption() {
        //VehicleEntry ve = new VehicleEntry("bike","car",4,8);
        assertEquals(4, vehicleEntry.getGoodAvgconsumption());
    }
    @Test
    public void getBadAvgConsumption() {
        //VehicleEntry ve = new VehicleEntry("bike","car",4,8);
        assertEquals(8, vehicleEntry.getBadavgconsumption());
    }

    @Test
    public void testToDBObject() {
        DBObject obj = vehicleEntry.toDbObject();
        assertTrue(obj.containsField("goodvehicleType"));
        assertTrue(obj.containsField("badvehicleType"));
        assertTrue(obj.containsField("reducedco2"));
        assertTrue(obj.containsField("goodavgconsumption"));
        assertTrue(obj.containsField("badavgconsumption"));
    }

}
