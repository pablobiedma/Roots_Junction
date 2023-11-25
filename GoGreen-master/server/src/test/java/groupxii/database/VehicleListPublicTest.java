package groupxii.database;

import org.junit.Test;

import static org.junit.Assert.*;

public class VehicleListPublicTest {

    @Test
    public void defConstructorTest() {
        VehicleListPublic  vehicleListPublic = new VehicleListPublic();
        assertTrue(vehicleListPublic.getVehicleList().isEmpty());
    }

    @Test
    public void getVehicleList() {
        VehicleListPublic  vehicleListPublic = new VehicleListPublic();
        VehicleListPublic  vehicleListPublic2 = new VehicleListPublic();
        assertEquals(vehicleListPublic.getVehicleList(),vehicleListPublic2.getVehicleList());

    }
    @Test
    public void addVehicleName() {
        VehicleListPublic vehicleListPublic = new VehicleListPublic();
        vehicleListPublic.addVehicleName("car");
        assertEquals(vehicleListPublic.getVehicleList().get(0),"car");
    }
}
