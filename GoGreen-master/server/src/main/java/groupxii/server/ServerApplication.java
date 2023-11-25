package groupxii.server;

import groupxii.database.Database;
import groupxii.server.security.SecurityKey;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class ServerApplication {
    /**
     * The main method of our server.
     */
    public static void main(String[] args) throws IOException {
        Database.instance.startDb();
//        SecurityKey.instance.readKey();
        SpringApplication.run(ServerApplication.class, args);
    }
}
