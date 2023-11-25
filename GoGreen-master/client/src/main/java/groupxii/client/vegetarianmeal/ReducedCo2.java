package groupxii.client.vegetarianmeal;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import groupxii.client.connector.VegetarianMealConnector;

import java.io.IOException;

public class ReducedCo2 {

    /**
     * get the reduced CO2 of a meal.
     * @param goodFoodName name of the good food
     * @param goodServingSize size of the good food
     * @param badFoodName name of the bad food
     * @param badServingSize size of the bad food
     * @return String with the reduced CO2 amount
     */
    public static String getReducedCo2(String goodFoodName,
                                       int goodServingSize,
                                       String badFoodName,
                                       int badServingSize) {
        String response = VegetarianMealConnector
                .calculateCO2Reduction(goodFoodName,
                        goodServingSize,
                        badFoodName,
                        badServingSize);
        ObjectMapper objectMapper = new ObjectMapper();

        JsonNode node = null;
        try {
            node = objectMapper.readTree(response);
        } catch (IOException e) {
            e.printStackTrace();
        }

        String result = node.get("reducedCO2").asText();
        return result;
    }
}
