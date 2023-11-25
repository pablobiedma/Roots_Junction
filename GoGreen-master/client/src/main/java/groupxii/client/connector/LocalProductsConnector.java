package groupxii.client.connector;

import groupxii.client.localproducts.GetUserLocation;

public class LocalProductsConnector {
    /**
     * Asks for the local shops from the server.
     * @return list with the shops nearby
     */
    public static String retrieveLocalShops() {
        String resource = Connector.instance.getRequest("/localshops?location="
                + GetUserLocation.getUserLocation());
        return resource;
    }
}
