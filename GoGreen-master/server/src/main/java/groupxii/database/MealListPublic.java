package groupxii.database;

import java.util.ArrayList;
import java.util.List;

/**
 * A list of all MealList food names for conveniet send to the client.
 */
public class MealListPublic {
    private List<String> mealList;

    public MealListPublic() {
        mealList = new ArrayList<>();
    }

    public List<String> getMealList() {
        return this.mealList;
    }

    public void addFoodName(String foodName) {
        mealList.add(foodName);
    }
}

