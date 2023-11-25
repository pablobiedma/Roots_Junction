package groupxii.client.vegetarianmeal;

import com.fasterxml.jackson.databind.ObjectMapper;
import groupxii.client.connector.VegetarianMealConnector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Class that stores the meal list.
 */
public class MealList {
    private List<String> mealList;

    /**
     * Asks the connector to retrieve the meal list and parses it.
     */
    public MealList() {
        String json = VegetarianMealConnector.retrieveMealList();
        this.parseJson(json);
    }

    private void parseJson(String json) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            //Does this has to be like this?
            this.mealList = mapper.readValue(json,
                            mapper.getTypeFactory()
                                    .constructCollectionType(List.class,
                                            String.class));
        } catch (IOException e) {
            this.mealList = new ArrayList<>();
            System.err.println(e.getMessage());
        }
    }

    public List<String> getMealList() {
        return this.mealList;
    }
}
