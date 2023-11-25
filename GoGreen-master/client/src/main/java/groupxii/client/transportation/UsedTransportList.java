package groupxii.client.transportation;

import groupxii.client.connector.TransportConnector;

import java.util.Arrays;
import java.util.List;

/**
 * Class that stores the used transport.
 */
public class UsedTransportList {
    private List<String> transportList;

    public UsedTransportList() {
        transportList = Arrays.asList(TransportConnector.retrieveUsedTransportList().split(" - "));
    }

    /**
     * Asks the connector to retrieve the eaten meal list and parses it.
     */
    public void setUsedTransportList() {
        String usedTransportStr = TransportConnector.retrieveUsedTransportList();
        transportList = Arrays.asList(usedTransportStr.split(" - "));
    }

    public List<String> getEatenMealList() {
        return this.transportList;
    }
}
