package groupxii.database;

import com.mongodb.DBObject;
import org.bson.BSONObject;

public class VehicleListEntry extends Entry {
    private String vehicletype;
    private double co2PerVehicle;
    private String fuel;
    private int avgConsumption;

    /**
     * creates a VehicleEntry representation for the database.
     */
    public VehicleListEntry(String vehicletype, double co2PerVehicle, String fuel,
                            int avgConsumption) {
        super();
        this.vehicletype = vehicletype;
        this.co2PerVehicle = co2PerVehicle;
        this.fuel = fuel;
        this.avgConsumption = avgConsumption;
    }


    /**
     * Constructs a VehicleListEntry from a BSONObject, usually a result
     * from query.
     */
    public VehicleListEntry(BSONObject object) {
        if (object == null) {
            return;
        }
        if (object.containsField("vehicletype")) {
            this.vehicletype = (String) object.get("vehicletype");
        }
        if (object.containsField("co2PerVehicle")) {
            this.co2PerVehicle = (double) object.get("co2PerVehicle");
        }
        if (object.containsField("fuel")) {
            this.fuel = (String) object.get("fuel");
        }
        if (object.containsField("AvgConsumption")) {
            this.avgConsumption = (int) object.get("AvgConsumption");
        }
    }

    public String getFuel() {
        return fuel;
    }

    public double getCo2PerVehicle() {
        return co2PerVehicle;
    }

    public int getAvgConsumption() {
        return avgConsumption;
    }

    public String getVehicletype() {
        return vehicletype;
    }

    /**
     * Translate into a MongoDB JSON object.
     */
    public DBObject toDbObject() {
        return super.toBasicDbObject()
                .append("vehicletype", this.vehicletype)
                .append("co2PerVehicle", this.co2PerVehicle)
                .append("fuel",this.fuel)
                .append("AvgConsumption", this.avgConsumption);

    }
}
