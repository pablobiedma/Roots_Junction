package groupxii.client.vegetarianmeal;

import groupxii.client.connector.VegetarianMealConnector;
import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;
import java.util.List;

/**
 * Class that retrieves and parses the eaten meal list.
 */
//TODO
public class EatenMealList {
    /**
     * Asks the connector to retrieve the eaten meal list and parses it.
     */
    public static List<String> getEatenMealList() {
        List<String> eatenMealList = new ArrayList<>();

        JSONArray jsonArray = new JSONArray(VegetarianMealConnector.retrieveEatenMealList());

        for (int i = 0; i < jsonArray.length(); i++ ) {
            JSONObject jsonEntry = (JSONObject)jsonArray.get(i);
            String entry = "Ate a ";

            entry += jsonEntry.get("goodFoodName").toString();
            entry += " and saved: ";
            entry += jsonEntry.get("reducedCo2").toString();
            entry += " of Co2!";

            eatenMealList.add(entry);
        }

        return eatenMealList;
    }
}
