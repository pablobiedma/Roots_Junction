package groupxii.client.localproducts;

import org.junit.Test;

import static groupxii.client.localproducts.GetUserLocation.getUserLocation;
import static org.junit.Assert.*;

public class GetUserLocationTest {

    @Test
    public void getUserLocationTest() {
        assertTrue(getUserLocation().contains(","));
    }
}