package groupxii.client.connector;

/**
 * Holds the location of the target server.
 */
public class Target {
    //private String protocol = "http";
    // private String address = "rilia.xyz";
    //private int port = 8080;
    // TODO read those from file

    public static String getHost() {
        String host = "http://rilia.xyz:8080";
        return host;
    }
}


