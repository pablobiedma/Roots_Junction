package groupxii.server.controllers;

import groupxii.localproducts.LocalShop;
import groupxii.localproducts.ReadLocalProductJson;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RestController
public class LocalProductsController {

    private List<LocalShop> localShopList = new ArrayList<>();
    private ReadLocalProductJson readLocalProductJson = new ReadLocalProductJson();
    private String responseString = "";

    public void setLocalShopList(List<LocalShop> localShopList) {
        this.localShopList = localShopList;
    }

    /**
     * get the local shop data.
     * @param location long- and latitude of the user
     * @return String with the local shops data
     * @throws IOException if it is not able to connect to the api
     */
    @RequestMapping(method = RequestMethod.GET, value = "/localshops")
    public String localShopData(@RequestParam(value = "location",
            defaultValue = "52.011578,4.357068") String location) throws IOException {
        readLocalProductJson.setLocation(location);
        readLocalProductJson.readLocalProductJson();
        responseString = readLocalProductJson.localShopToString();
        readLocalProductJson.clearList();
        return responseString;
    }
}
