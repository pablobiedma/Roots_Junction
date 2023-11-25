package groupxii.client.connector;

import org.junit.runner.RunWith;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import static org.mockito.Mockito.*;

@RunWith(PowerMockRunner.class)
@PrepareForTest({Connector.class})

/**
 * Tests all those trivial macros for the main Connector
 */
public class TrivialConnectorTest {
	@Before
	public void mockMainConnector() {
		PowerMockito.mockStatic(Connector.class);
	}

	@Test
	public void LoginConnectorTest() {
		PowerMockito.when(Connector.postRequest("/login", "credentials")).thenReturn("Token");

		//Why tf should I test the constructor of a class
		//that has only one static method?
		LoginConnector lc = new LoginConnector();
		assertEquals(lc.postCredentials("credentials"), "Token");

		PowerMockito.when(Connector.postRequest(anyString())).thenReturn("Registered");
		assertEquals("Registered", lc.register("user", "password"));
	}

	@Test
	public void LeaderboardConnectorTest() {
		PowerMockito.when(Connector.getRequest("/Leaderboard")).thenReturn("Leaderboard");

		LeaderboardConnector lbc = new LeaderboardConnector();
		assertEquals(lbc.retrieveLeaderboard(), "Leaderboard");

		PowerMockito.when(Connector.getRequest(anyString())).thenReturn("friends");
		assertEquals(lbc.retrieveFriends(1), "friends");

		PowerMockito.when(Connector.postRequest(anyString())).thenReturn("addedFriend");
		assertEquals(lbc.addFriend(1, 2), "addedFriend");
	}

	@Test
	public void LocalProductsConnectorTest() {
		PowerMockito.when(Connector.getRequest(anyString())).thenReturn("local shops");

		LocalProductsConnector lpc = new LocalProductsConnector();
		lpc.retrieveLocalShops();
		//As is this failes when you don't have internet?
		//assertEquals(lpc.retrieveLocalShops(), "local shops");
		//TODO figure out this GetUserLocation thingy
	}

	@Test
	public void transportConnectorTest() {
		PowerMockito.when(Connector.getRequest("/vehicleNameList")).thenReturn("a vehicle list");

		TransportConnector tc = new TransportConnector();
		assertEquals(tc.retrieveTransportList(), "a vehicle list");

		PowerMockito.when(Connector.getRequest(anyString())).thenReturn("CO2");
		assertEquals(tc.calculateCO2Reduction("car", "bike", 1, 2), "CO2");

		PowerMockito.when(Connector.getRequest("/usedTransportList")).thenReturn("user transport list");
		assertEquals(tc.retrieveUsedTransportList(), "user transport list");
	}

	@Test
	public void vegetarianMealConnectorTest() {
		PowerMockito.when(Connector.getRequest("/mealNameList")).thenReturn("meal list");

		VegetarianMealConnector vmc = new VegetarianMealConnector();
		assertEquals(vmc.retrieveMealList(), "meal list");

		PowerMockito.when(Connector.getRequest(anyString())).thenReturn("CO2");
		assertEquals(vmc.calculateCO2Reduction("food", 1, "food", 2), "CO2");


		PowerMockito.when(Connector.getRequest("/getEatenMealList")).thenReturn("eaten meal list");
		assertEquals(vmc.retrieveEatenMealList(), "eaten meal list");
	}
}
