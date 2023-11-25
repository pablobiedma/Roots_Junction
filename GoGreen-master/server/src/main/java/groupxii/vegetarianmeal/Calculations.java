package groupxii.vegetarianmeal;

import groupxii.database.Database;
import groupxii.database.MealListEntry;

public class Calculations {
    /**The Chosen food is looked up in the Database and
     * than the private values are updated and the CO2 emmision is calculated.
     */
    public static int calculateCO2(String chosenFood, int chosenServingSize) {
        MealListEntry mealListEntry = Database.instance.findMealListEntry(chosenFood);
        // if the food cannot be found find a way to return an error (throw custom exception?)
        int result = -1;
        double foodCO2 = mealListEntry.getCO2PerServing();
        double servingSize = mealListEntry.getServingSize();

        double co2PerGram = foodCO2 / servingSize;
        result = (int)(co2PerGram * chosenServingSize);
        return result;
    }

    /**this is the second calculation that is done, people also choose
     * a bad food what they normally eat, so the reduced CO2 can be calculated.
     */

    /*
    public int reducedCO2(
            String badFood, int chosenServingSize, int goodFood) throws IOException {
        this.reducedCO2 = (int) (calculateCO2(badFood, chosenServingSize) - goodFood);
        //System.out.println(reducedCO2);
        return this.reducedCO2;
    }
    */

    /* *shrug*
    //this is the last calculation that is done, the points for the user are calculated.
    public int calculatePoints() {
        return (int) this.reducedCO2 * 5;
    }
    */
}
