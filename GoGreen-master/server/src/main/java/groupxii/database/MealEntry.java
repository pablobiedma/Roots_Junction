package groupxii.database;

import com.mongodb.DBObject;
import groupxii.vegetarianmeal.CalculatedMeal;

public class MealEntry extends Entry {

    /**
     * Represents entries in the database for every meal
     * the client decides to post.
     */
    private String goodFoodName;
    private int goodServingSize;
    private String badFoodName;
    private int badServingSize;
    private int reducedCo2;

    /**
     * Constructor of MealEntry.
     */
    public MealEntry(String goodFoodName, int goodServingSize,
                     String badFoodName, int badServingSize) {
        this.goodFoodName = goodFoodName;
        this.goodServingSize = goodServingSize;
        this.badFoodName = badFoodName;
        this.badServingSize = badServingSize;
        this.reducedCo2 = new CalculatedMeal(goodFoodName, goodServingSize,
                                             badFoodName, badServingSize).getReducedCO2();
    }

    public String getGoodFoodName() {
        return goodFoodName;
    }

    public String getBadFoodName() {
        return badFoodName;
    }

    public int getGoodServingSize() {
        return goodServingSize;
    }

    public int getBadServingSize() {
        return badServingSize;
    }

    public int getReducedCo2() {
        return reducedCo2;
    }

    /**
     * Translates into a MongoDB JSON object.
     */
    public final DBObject toDbObject() {
        return super.toBasicDbObject()
                .append("goodFoodName", this.goodFoodName)
                .append("goodServingSize", this.goodServingSize)
                .append("badFoodName", this.badFoodName)
                .append("badServingSize", this.badServingSize)
                .append("reducedCo2", this.reducedCo2);
    }
}
