package groupxii.transportation;

import groupxii.database.Database;
import groupxii.database.VehicleListEntry;

public class VehicleCalculations {

    /**
     * Calculates the co2 consumed by a given vehicle.
     */
    public static int calculateCO2(String chosenVehicle, int chosenAvgConsumption) {

        VehicleListEntry vehicleListEntry =
                Database.instance.findVehicleListEntry(chosenVehicle);
        int result = -1;
        double vehicleCO2 = vehicleListEntry.getCo2PerVehicle();
        double avgConsumption = vehicleListEntry.getAvgConsumption();

        double co2PerGram = vehicleCO2 / avgConsumption;
        result = (int) (co2PerGram * chosenAvgConsumption);
        return result;

    }
}
