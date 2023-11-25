package groupxii.client.connector;

import static groupxii.client.connector.Connector.postRequest;

public class UserConnector {

    /**
     * communicates with the target server, and updates the
     * total co2 reduction of the user in the database.
     * @param amount of CO2 the user has reduced
     * @return JSON String
     */
    public static String updateReducedCo2(int amount) {
        return postRequest("/increaseReducedCo2OfUser?amount="
                + amount);
    }
}
