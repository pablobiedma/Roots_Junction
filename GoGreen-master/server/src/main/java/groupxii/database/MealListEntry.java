package groupxii.database;

import com.mongodb.DBObject;
import org.bson.BSONObject;

/**
 * Representation of a single entry from the MealList file.
 */
public class MealListEntry extends Entry {
    private String foodName;
    private double co2PerServing;
    private double calPerServing;
    private double servingSize;

    /**
     * Creates a MealList entry representation for the database.
     */
    public MealListEntry(String foodName, double co2PerServing,
                             double calPerServing, double servingSize) {
        super();
        this.foodName = foodName;
        this.co2PerServing = co2PerServing;
        this.calPerServing = calPerServing;
        this.servingSize = servingSize;
    }

    /**
     * Constructs a MealListEntry from a BSONObject, usually a result from query.
     */
    public MealListEntry(BSONObject obj) {
        if (obj == null) {
            return;
        }
        if (obj.containsField("foodName")) {
            this.foodName = (String)obj.get("foodName");
        }
        if (obj.containsField("co2PerServing")) {
            this.co2PerServing = (double)obj.get("co2PerServing");
        }
        if (obj.containsField("calPerServing")) {
            this.calPerServing = (double)obj.get("calPerServing");
        }
        if (obj.containsField("servingSize")) {
            this.servingSize = (double)obj.get("servingSize");
        }
    }

    public String getFoodName() {
        return this.foodName;
    }
    
    public double getCO2PerServing() {
        return this.co2PerServing;
    }

    public double getCalPerServing() {
        return this.calPerServing;
    }

    public double getServingSize() {
        return this.servingSize;
    }

    /**
     * Translate into a MongoDB JSON object.
     */
    public DBObject toDbObject() {
        return super.toBasicDbObject()
            .append("foodName", this.foodName)
            .append("co2PerServing", this.co2PerServing)
            .append("calPerServing", this.calPerServing)
            .append("servingSize", this.servingSize);
    }
}
