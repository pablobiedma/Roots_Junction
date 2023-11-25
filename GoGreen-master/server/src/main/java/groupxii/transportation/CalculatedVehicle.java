package groupxii.transportation;


public class CalculatedVehicle {
    private int reducedCO2;

    /**
     * calculates the reduced co2 between 2 vehicles.
     */
    public CalculatedVehicle(String goodvehicletype, String badvehicletype,
                             int goodavgconsumption, int badavgconsumption) {
        this.reducedCO2 = 1;

        int result = VehicleCalculations.calculateCO2(badvehicletype, badavgconsumption);
        result -= VehicleCalculations.calculateCO2(goodvehicletype, goodavgconsumption);
        this.reducedCO2 = result;

    }

    public int getReducedCO2() {
        return this.reducedCO2;
    }

}
