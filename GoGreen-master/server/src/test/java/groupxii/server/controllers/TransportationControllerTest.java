package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.UserEntry;
import groupxii.database.VehicleEntry;
import groupxii.transportation.CalculatedVehicle;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.security.Principal;
import java.util.List;

import static org.junit.Assert.*;


public class TransportationControllerTest {

    TransportationController transportationController;
    UserEntry usr;
    @Before
    public void createTC() {
        try {
            Database.instance.startDb();
        } catch (IOException e) {
            assertTrue(false);
        }
        transportationController = new TransportationController();
        usr = new UserEntry(1,"Ivan","pass");
        Database.instance.saveNonBlocking(usr);
    }
    @Before

    @Test
    public void calculatevehicleTest() {
        CalculatedVehicle localCalculateVehicle = new CalculatedVehicle("bike", "car", 4, 8);
        CalculatedVehicle remoteCalculateVehicle = transportationController.calculateVehicle("bike", 4, "car", 8);
        assertEquals(localCalculateVehicle.getReducedCO2(), remoteCalculateVehicle.getReducedCO2());
    }

    @Test
    public void getNameList() {
        List<String> nameList = transportationController.getNameList();
        assertFalse(nameList.isEmpty());
    }

    @Test
    public void testSaveVehicleData() {
        VehicleEntry vehicleEntry = new VehicleEntry("bike","car",4,8);
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        Database.instance.addUsedVehicle(username, vehicleEntry);
        transportationController.saveVehicleData("bike",4,"car",8,principal);
        assertEquals(transportationController.getNameList(),Database.instance.getVehicleListVehicleNames());
    }

    @Test
    public void testGetUsedVehicles() {
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        UserEntry userEntry = Database.instance.findUserByName(username);
        List<VehicleEntry> vehicleEntries = transportationController.getUsedVehicleList(principal);
        assertEquals(userEntry.getUsedVehicles(),vehicleEntries);
    }
}
