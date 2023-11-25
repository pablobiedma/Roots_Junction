package groupxii.database;

import groupxii.vegetarianmeal.CalculatedMeal;
import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;

import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertNull;

public class MealListEntryTest {
	MealListEntry mle;
	@Before
	public void createEntry() {
		mle = new MealListEntry("GRAPEFRUIT", 23.0, 30.0, 60.0);
	}
	@Test
	public void testGetFoodName() {
		assertEquals(mle.getFoodName(), "GRAPEFRUIT");
	}

	@Test
	public void testGetCO2PerServ() {
		assertEquals(mle.getCO2PerServing(), 23.0, 0);
	}

	@Test
	public void testGetCalPerServ() {
		assertEquals(mle.getCalPerServing(), 30.0, 0);
	}

	@Test
	public void testGetServingSize() {
		assertEquals(mle.getServingSize(), 60.0, 0);
	}

	@Test
	public void testToDBObject() {
		DBObject obj = mle.toDbObject();
		assertTrue(obj.containsField("foodName"));
		assertTrue(obj.containsField("co2PerServing"));
		assertTrue(obj.containsField("calPerServing"));
		assertTrue(obj.containsField("servingSize"));
	}

	@Test
	public void testFromDBObject() {
		DBObject obj = new BasicDBObject()
					.append("foodName", "GRAPEFRUIT")
					.append("co2PerServing", 23.0)
					.append("calPerServing", 30.0)
					.append("servingSize", 60.0);
		mle = new MealListEntry(obj);
		testToDBObject();
	}

	@Test
	public void testFromNullObj() {
		DBObject obj = null;
		mle = new MealListEntry(obj);
		assertNull(mle.getFoodName());
		assertEquals(mle.getCO2PerServing(), 0.0, 0);
		assertEquals(mle.getCalPerServing(), 0.0, 0);
		assertEquals(mle.getServingSize(), 0.0, 0);
	}

	@Test
	public void testFromBadDBObject() {
		DBObject obj = new BasicDBObject()
					.append("fodName", "CAT")
					.append("co2PerServng", 23)
					.append("calerServing", "boat")
					.append("servingMass", 60.0);
		mle = new MealListEntry(obj);
		assertNull(mle.getFoodName());
		assertEquals(mle.getCO2PerServing(), 0.0, 0);
		assertEquals(mle.getCalPerServing(), 0.0, 0);
		assertEquals(mle.getServingSize(), 0.0, 0);
	}
}
