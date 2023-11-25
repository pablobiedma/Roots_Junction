package groupxii.client.connector;

public class TransportConnector {
    /**
     * Shortcut for /vehicleNameList.
     */
    public static String retrieveTransportList() {
        String resource = "/vehicleNameList";
        return Connector.getRequest(resource);
    }

    /**
     * Shortcut for /calculateTransprot.
     */
    public static String calculateCO2Reduction(String goodTransportName,
                                               String badTransportName,
                                               int goodConsumption,
                                               int badConsumption) {
        String resource = "/calculateTransport"
                + "?goodVehicleType=" + goodTransportName
                + "&badVehicleType" + badTransportName
                + "&goodAvgConsumption" + goodConsumption
                + "&badAvgConsumption" + badConsumption;

        return Connector.getRequest(resource);
    }

    /**
     * Shortcut for /usedTransportList.
     */
    public static String retrieveUsedTransportList() {
        String resource = "/usedTransportList";
        return Connector.getRequest(resource);
    }
}
