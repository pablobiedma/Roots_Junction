package groupxii.client.localproducts;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.maxmind.geoip2.exception.GeoIp2Exception;

import java.io.IOException;
import java.net.URL;
import java.util.Scanner;


public class GetUserLocation {

    /**
     * gets the location of the user by using his IP adress in combination with the eth0.com API.
     * @return String with the latitude and longitude of the user.
     * @throws IOException if it is not able to connect to the api.
     * @throws GeoIp2Exception if ip cannot be turned into a location.
     */
    public static String getUserLocation() {
        ObjectMapper objectMapper = new ObjectMapper();
        JsonNode rootNode = null;
        try {
            String ipAddress = new Scanner(new URL("http://eth0.me/").openStream()).nextLine();
            rootNode = objectMapper.readTree(new URL("http://api.ipstack.com/" + ipAddress + "?access_key=f8449c29422a48b1dd367afadaa10714"));
        } catch (IOException e) {
            e.printStackTrace();
            return "52.011578,4.357068";
        }
        String longitude = rootNode.get("longitude").asText();
        String latitude = rootNode.get("latitude").asText();

        if (!longitude.equals("") && !latitude.equals("")) {
            return latitude + "," + longitude;
        } else {
            return "52.011578,4.357068";
        }

    }
}
