package groupxii.client.connector;

public class LoginConnector {
    /**
     * Shortcut for requesting /login.
     */
    public static String postCredentials(String reqBody) {
        String resource = "/login";

        return Connector.postRequest(resource, reqBody);
    }

    /**
     * Shortcut for requesting /register with two variables.
     */
    public static String register(String username, String password) {
        String resource = "/register"
            + "?username=" + username
            + "&password=" + password;

        return Connector.postRequest(resource);
    }
}
