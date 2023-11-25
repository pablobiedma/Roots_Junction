package groupxii.client.connector;

public class VegetarianMealConnector {
    /**
     * Shortcut for /mealNameList.
     */
    public static String retrieveMealList() {
        String resource = "/mealNameList";
        return Connector.getRequest(resource);
    }

    /**
     * calculate the reduced co2 of a meal.
     * @param goodFoodName name of the good food
     * @param goodServingSize size of the good food
     * @param badFoodName name of the bad food
     * @param badServingSize size of the bad food
     * @return String with the amount of reduced CO2
     */
    public static String calculateCO2Reduction(String goodFoodName, int goodServingSize,
                                               String badFoodName, int badServingSize) {
        String resource = "/calculateMeal"
                + "?goodFoodName=" + goodFoodName + ""
                + "&goodServingSize=" + goodServingSize
                + "&badFoodName=" + badFoodName + ""
                + "&badServingSize=" + badServingSize;

        return Connector.getRequest(resource);
    }

    /**
     * Shortcut for /saveMealData.
     */
    public static String commitMeal(String goodFoodName, int goodServingSize,
                                    String badFoodName, int badServingSize) {
        String resource = "/saveMealData"
                + "?goodFoodName=" + goodFoodName + ""
                + "&goodServingSize=" + goodServingSize
                + "&badFoodName=" + badFoodName + ""
                + "&badServingSize=" + badServingSize;

        return Connector.postRequest(resource);
    }

    /**
     * Shortcut for /getEatenMealList.
     */
    public static String retrieveEatenMealList() {
        String resource = "/getEatenMealList";
        return Connector.getRequest(resource);
    }
}
