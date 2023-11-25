package groupxii.database;

import java.util.ArrayList;
import java.util.List;

/**
 * A list of all VehicleList vehicle types for convenient send to the client.
 */
public class VehicleListPublic {
    private List<String> vehicleList;

    public VehicleListPublic() {
        vehicleList = new ArrayList<>();
    }

    public List<String> getVehicleList() {
        return this.vehicleList;
    }

    public void addVehicleName(String vehicle) {
        vehicleList.add(vehicle);
    }
}
