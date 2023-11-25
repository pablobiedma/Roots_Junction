package groupxii.database;

import com.mongodb.DBObject;
import groupxii.transportation.CalculatedVehicle;

/**
 * Represents entries in the database which tracks
 * which vehicle each request indicates.
 */
public class VehicleEntry extends Entry {
    private String goodvehicleType;
    private String badvehicleType;
    private int reducedco2;
    private int goodavgconsumption;
    private int badavgconsumption;

    /**
     * Creates VehicleEntry with a userid, vehicle type, co2 emission and type of fuel.
     */
    public VehicleEntry(String goodvehicleType, String badvehicletype,
                        int goodavgconsumption,
                        int badavgconsumption) {
        super();
        this.goodvehicleType = goodvehicleType;
        this.badvehicleType = badvehicletype;
        this.reducedco2 = new CalculatedVehicle(goodvehicleType, badvehicletype ,
                goodavgconsumption, badavgconsumption).getReducedCO2();
        this.goodavgconsumption = goodavgconsumption;
        this.badavgconsumption = badavgconsumption;
    }

    public String getGoodVehicleType() {
        return goodvehicleType;
    }

    public String getBadvehicleType() {
        return badvehicleType;
    }

    public int getReducedco2() {
        return reducedco2;
    }

    public int getGoodAvgconsumption() {
        return goodavgconsumption;
    }

    public int getBadavgconsumption() {
        return badavgconsumption;
    }

    /**
     * Translates into a MongoDB JSON object.
     */
    public final DBObject toDbObject() {
        return super.toBasicDbObject()
                .append("goodvehicleType", this.goodvehicleType)
                .append("badvehicleType", this.badvehicleType)
                .append("reducedco2",this.reducedco2)
                .append("goodavgconsumption",this.goodavgconsumption)
                .append("badavgconsumption",this.badavgconsumption);
    }
}
