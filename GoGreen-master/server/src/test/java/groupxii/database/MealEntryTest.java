package groupxii.database;

import groupxii.vegetarianmeal.CalculatedMeal;
import com.mongodb.DBObject;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;

public class MealEntryTest {
	MealEntry me;
	@Before
	public void createEntry() {
		//needed to fetch  info for calculating CO2
		try {
			Database.instance.startDb();
		} catch (IOException e) {
			assertTrue(false);
		}
		me = new MealEntry("GRAPEFRUIT", 100, "BUTTER", 120);
	}

	@Test
	public void testGoodFodName() {
		assertEquals(me.getGoodFoodName(), "GRAPEFRUIT");
	}

	@Test
	public void testBadFoodName() {
		assertEquals(me.getBadFoodName(), "BUTTER");
	}

	@Test
	public void testGoodServingSize() {
		assertEquals(me.getGoodServingSize(), 100);
	}

	@Test
	public void testBadServingSize() {
		assertEquals(me.getBadServingSize(), 120);
	}

	@Test
	public void testReaducedCO2() {
		CalculatedMeal cm = new CalculatedMeal("GRAPEFRUIT", 100, "BUTTER", 120);
		assertEquals(me.getReducedCo2(), cm.getReducedCO2());
	}

	@Test
	public void testToDBObject() {
		DBObject obj = me.toDbObject();
		assertTrue(obj.containsField("goodFoodName"));
		assertTrue(obj.containsField("goodServingSize"));
		assertTrue(obj.containsField("badFoodName"));
		assertTrue(obj.containsField("badServingSize"));
		assertTrue(obj.containsField("reducedCo2"));
	}
}
