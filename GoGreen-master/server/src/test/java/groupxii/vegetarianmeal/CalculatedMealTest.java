package groupxii.vegetarianmeal;

import groupxii.database.Database;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import org.junit.Test;
import org.junit.Before;

import java.io.IOException;

public class CalculatedMealTest {
	@Before
	public void startDB() {
		try{
			Database.instance.startDb();
		} catch (IOException e) {
			assertTrue(false);
		}
	}
	@Test
	public void testCalculatedMeal() {
		CalculatedMeal cm = new CalculatedMeal(
				"GRAPEFRUIT", 100,
				"GRAPEFRUIT", 100);
		assertEquals(0, cm.getReducedCO2());
	}
}
