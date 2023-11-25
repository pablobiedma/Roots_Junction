package groupxii.server.controllers;

import groupxii.database.MealEntry;
import groupxii.database.Database;
import groupxii.database.MealListPublic;
import groupxii.database.UserEntry;
import groupxii.vegetarianmeal.CalculatedMeal;
import org.junit.Test;
import org.junit.Before;

import java.io.IOException;
import java.security.Principal;
import java.util.List;

import static org.junit.Assert.*;

public class VegetarianMealControllerTest {

    VegetarianMealController vegetarianMealController;
    UserEntry usr;
    @Before
    public void createVMC() {
	    try {
		    Database.instance.startDb();
	    } catch (IOException e) {
		    assertTrue(false);
	    }
	    vegetarianMealController = new VegetarianMealController();
        usr = new UserEntry(1,"Ivan","pass");
        Database.instance.saveNonBlocking(usr);
    }

    @Test
    public void calculateMealTest() {
	CalculatedMeal localCalculateMeal = new CalculatedMeal("GRAPEFRUIT", 100, "EGGPLANT", 120);
	CalculatedMeal remoteCalculateMeal = vegetarianMealController.calculateMeal("GRAPEFRUIT", 100, "EGGPLANT", 120);
        assertEquals(localCalculateMeal.getReducedCO2(), remoteCalculateMeal.getReducedCO2());
    }

    @Test
    public void getNameList() {
        List<String> nameList = vegetarianMealController.getNameList();
        assertFalse(nameList.isEmpty());
    }

    @Test
    public void testSaveMealData() {
        MealEntry mealEntry  = new MealEntry("GRAPEFRUIT", 100, "BUTTER", 120);
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        Database.instance.addEatenMeal(username, mealEntry);
        vegetarianMealController.saveMealData("GRAPEFRUIT", 100, "BUTTER", 120,principal);
        assertEquals(vegetarianMealController.getNameList(),Database.instance.getMealListFoodNames());
    }

    @Test
    public void testGetEatenMealList() {
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        UserEntry userEntry = Database.instance.findUserByName(username);
        List<MealEntry> mealEntries = vegetarianMealController.getEatenMealList(principal);
        assertEquals(userEntry.getEatenMeals(),mealEntries);
    }

}
