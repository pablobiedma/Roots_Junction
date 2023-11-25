package groupxii.vegetarianmeal;

import groupxii.database.Database;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

import java.io.IOException;

public class CalculationsTest {
	@Before
	public void startDb() {
		try {
			Database.instance.startDb();
		} catch (IOException e) {
			assertTrue(false);
		}
	}
	@Test
	public void testCalculate() {
		assertEquals(19, Calculations.calculateCO2(
					"GRAPEFRUIT", 100));
	}

	@Test
	public void testInstantiated() {
		Calculations cal = new Calculations();
		assertEquals(19, cal.calculateCO2(
					"GRAPEFRUIT", 100));
	}
}
