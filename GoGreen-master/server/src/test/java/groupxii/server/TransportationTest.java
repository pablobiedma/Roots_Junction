package groupxii.server;

import org.junit.Test;
import static org.junit.Assert.assertEquals;

public class TransportationTest {

	// testing vehicle suggestion
	@Test
	public void testSuggestVehicleCar() {
		Transportation transportation = new Transportation(1, "car");
		assertEquals("At least use public transport!", transportation.getVehicleSuggestion());
	}
	@Test
	public void testSuggestVehiclePublic_transport() {
		Transportation transportation = new Transportation(1, "public_transport");
		assertEquals("Use a bike instead", transportation.getVehicleSuggestion());
	}
	@Test
	public void testSuggestVehicleBike() {
		Transportation transportation = new Transportation(1, "bike");
		assertEquals("Good job!", transportation.getVehicleSuggestion());
	}
	@Test
	public void testSuggestVehicleUnknown() {
		Transportation transportation = new Transportation(1, "Unknown");
		assertEquals("Just use a bike", transportation.getVehicleSuggestion());
	}
	@Test
	public void testSuggestVehicleEmptyStr() {
		Transportation transportation = new Transportation(1, "");
		assertEquals("Just use a bike", transportation.getVehicleSuggestion());
	}

	@Test
	public void testConstructorID() {
		Transportation transportation = new Transportation(1, "car");
		assertEquals(1, transportation.getId());
	}
}
