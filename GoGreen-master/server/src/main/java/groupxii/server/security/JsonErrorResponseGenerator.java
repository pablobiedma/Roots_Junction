package groupxii.server.security;

import java.util.Date;

public class JsonErrorResponseGenerator {
    public JsonErrorResponseGenerator() {
    }

    /**
     * Creates a message and returns it as a String.
     */
    public static String createMessage(int statusCode, String message, String path) {
        long date = new Date().getTime();
        String errType = "Unknown error.";

        if (statusCode == 400) {
            errType = "Bad Request";
        } else if (statusCode == 401) {
            errType = "Unauthorized";
        }

        return "{\"timestamp\": " + date + ", "
            + "\"status\": " + statusCode + ", "
            + "\"error\": \"" + errType + "\", "
            + "\"message\": \"" + message + "\", "
            + "\"path\": \"" + path + "\"}";
    }
}
