package groupxii.localproducts;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class ReadLocalProductJson {

    private List<LocalShop> localShopList = new ArrayList<>();
    private String location = "";

    public String getLocation() {
        return this.location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public void setLocalShopList(List<LocalShop> localShopList) {
        this.localShopList = localShopList;
    }

    public List<LocalShop> getLocalShopList() {
        return localShopList;
    }

    /**
     * reads all the local shops from the JSON.
     * @throws IOException if it is not recognized as a JSON
     */
    public void readLocalProductJson() throws IOException {
        GetLocalProductsJson getLocalProductsJson = new GetLocalProductsJson();
        ObjectMapper obj = new ObjectMapper();
        JsonNode rootNode = obj.readTree(getLocalProductsJson.getLocalShopJson(location));

        //iterates over all the elements in the JsonNode and stores the data we need
        JsonNode results = rootNode.path("results");
        Iterator<JsonNode> elements = results.elements();
        while (elements.hasNext()) {
            JsonNode node = elements.next();
            String name = node.get("name").asText();
            String location = node.get("vicinity").asText();
            location = location.replaceAll(", ", " - ");
            double rating = node.get("rating").asDouble();
            localShopList.add(new LocalShop(name, rating, location));
            elements.next();
        }
    }

    /**
     * transforms the String into a user friendly readable string.
     * @return a sentence for the GUI
     */
    public String localShopToString() {
        String result = "";
        for (int i = 0; i < localShopList.size(); i++) {
            result = result + " SHOP:  " +  localShopList.get(i).getName() + "  - LOCATED AT:  "
                    + localShopList.get(i).getLocation() + "  - RATING:  "
                    + localShopList.get(i).getRating() + ", ";
        }
        System.out.println(result);
        return result;
    }

    public void clearList() {
        localShopList.clear();
    }
}
