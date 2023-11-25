package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.MealEntry;
import groupxii.database.UserEntry;
import groupxii.vegetarianmeal.CalculatedMeal;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.List;

@RestController
public class VegetarianMealController {
    /**
    This method will transform the data from the mealList into one string, which then can be used
    by the client, so the choiceboxes/listviews in the GUI are able to show all the meal names.
     */
    /**
     * Return a list of all available food entries that the vegetarian meal feature can process.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/mealNameList")
    public List<String> getNameList() {
        return Database.instance.getMealListFoodNames();
    }

    /**
     * Calculate the saved CO2 and send the response to the server.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/calculateMeal")
    public CalculatedMeal calculateMeal(@RequestParam(value = "goodFoodName",
                                                      defaultValue = "Unknown")
                                        String goodFoodName,
                                        @RequestParam(value = "goodServingSize",
                                                      defaultValue = "-1")
                                        Integer goodServingSize,
                                        @RequestParam(value = "badFoodName",
                                                      defaultValue = "Unknown")
                                        String badFoodName,
                                        @RequestParam(value = "badServingSize",
                                                      defaultValue = "-1")
                                        Integer badServingSize) {

        return new CalculatedMeal(goodFoodName, goodServingSize, badFoodName, badServingSize);
    }

    /**
    the client can send data to the server with the right values as parameter,
    then this method will store the data in the database.
     */
    @RequestMapping(method = RequestMethod.POST, value = "/saveMealData")
    public void saveMealData(@RequestParam(value = "goodFoodName",
                                                         defaultValue = "Unknown")
                                           String goodFoodName,
                                           @RequestParam(value = "goodServingSize",
                                                         defaultValue = "-1")
                                           Integer goodServingSize,
                                           @RequestParam(value = "badFoodName",
                                                         defaultValue = "Unknown")
                                           String badFoodName,
                                           @RequestParam(value = "badServingSize",
                                                         defaultValue = "-1")
                                           Integer badServingSize,
                                           Principal principal) {
        String username = principal.getName();
        MealEntry saveMeal = new MealEntry(goodFoodName, goodServingSize,
                                           badFoodName, badServingSize);
        Database.instance.addEatenMeal(username, saveMeal);
        Database.instance.incrementReducedCo2(username, saveMeal.getReducedCo2());
    }

    /**
     * Given the principal, return a EatenMealList.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/getEatenMealList")
    public List<MealEntry> getEatenMealList(Principal principal) {
        String username = principal.getName();
        UserEntry user =  Database.instance.findUserByName(username);
        List<MealEntry> list = user.getEatenMeals(); 
        return list;
    }
}
