package groupxii.database;

import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class VehicleListEntryTest {

    VehicleListEntry vle;
    @Before
    public void createEntry() {
        vle = new VehicleListEntry("car", 82.332, "Diesel", 6);
    }

    @Test
    public void getFuel() {
        assertEquals("Diesel", vle.getFuel());
    }

    @Test
    public void getCo2PerVehicle() {
        assertEquals(vle.getCo2PerVehicle(), 82.332, 0);
    }

    @Test
    public void getAvgConsumption() {
        assertEquals(6, vle.getAvgConsumption());
    }

    @Test
    public void getVehicletype() {
        assertEquals("car", vle.getVehicletype());
    }

    @Test
    public void testFromNullObj() {
        DBObject obj = null;
        vle = new VehicleListEntry(obj);
        assertNull(vle.getVehicletype());
        assertEquals(vle.getCo2PerVehicle(), 0.0, 0);
        assertNull(vle.getFuel());
        assertEquals(vle.getAvgConsumption(), 0);
    }

    @Test
    public void testFromDBObj() {
        DBObject obj = new BasicDBObject().append("vehicletype","car")
                .append("co2PerVehicle", 84.3)
                .append("fuel", "Diesel")
                .append("AvgConsumption", 5);
        vle = new VehicleListEntry(obj);
        assertEquals(vle.getVehicletype(), "car");
        assertEquals(vle.getCo2PerVehicle(), 84.3, 0);
        assertEquals(vle.getFuel(), "Diesel");
        assertEquals(vle.getAvgConsumption(), 5);
    }


    @Test
    public void testFromBadObj() {
        DBObject obj = new BasicDBObject().append("vhcltype","car")
                .append("co2", 84.3)
                .append("gorivo", "Diesel")
                .append("Consum", 5);
        vle = new VehicleListEntry(obj);
        assertNull(vle.getVehicletype());
        assertEquals(vle.getCo2PerVehicle(), 0.0, 0);
        assertNull(vle.getFuel());
        assertEquals(vle.getAvgConsumption(), 0);
    }

    @Test
    public void testToDBObject() {
        DBObject object = vle.toDbObject();
        assertTrue(object.containsField("vehicletype"));
        assertTrue(object.containsField("co2PerVehicle"));
        assertTrue(object.containsField("fuel"));
        assertTrue(object.containsField("AvgConsumption"));
    }


}
