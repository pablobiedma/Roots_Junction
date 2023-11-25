package groupxii.client;

import groupxii.client.connector.LoginConnector;

/**
 * Holds the verification token and periodically updates it from the server.
 */
public class TokenManager {
    public static TokenManager instance;

    /**
     * The daemon timeout should roughly match the validty of the token.
     * TODO read the validity off the token.
     */
    public static final int timeout = 86390000;

    /**
     * Holds the JSON request to be posted on login.
     */
    private String loginReqBody;

    /**
     * Holds the JWT from the server.
     */
    private String jwt;

    /**
     * Constructs a token manager for the provided username and password.
     */
    public TokenManager(String username, String password) {
        jwt = null;

        loginReqBody = "{"
            + "\"username\":"
            + "\"" + username + "\", "
            + "\"password\":"
            + "\"" + password + "\"}";
        //TODO hash+salt the password

        Thread daemon = new Thread(new Daemon());
        daemon.setDaemon(true);
        daemon.start();
    }

    /**
     * Returns the currently held JWT.
     */
    public String getToken() {
        return this.jwt;
    }

    class Daemon implements Runnable {
        /**
         * Periodically requests a new token from the server.
         * Should be a daemon thread.
         */
        public void run() {
            while (true) {
                refreshToken();

                try {
                    Thread.sleep(timeout);
                } catch (InterruptedException e) {
                    //Die
                    return;
                }
            }
        }
    }

    public void refreshToken() {
        jwt = LoginConnector.postCredentials(loginReqBody);
        System.err.println("DEBUG: Recieved Token: " + jwt);
    }
}
