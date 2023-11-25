package groupxii.client.connector;

import java.io.IOException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;

/**
 * Dumb, but those two classes don't seem mokable.
 */
public class OpenConnection {
    /**
     * Creates HttpURLConnection.
     */
    public static HttpURLConnection openConnection(String host)
            throws MalformedURLException, IOException {
        URL url = new URL(host);
        HttpURLConnection con = (HttpURLConnection)url.openConnection();
        return con;
    }
}
