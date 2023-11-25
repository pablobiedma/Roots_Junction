package groupxii.database;

import static org.junit.Assert.*;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class MealListPublicTest {
	@Test
	public void defConstructorTest() {
		MealListPublic mlp = new MealListPublic();
		assertTrue(mlp.getMealList().isEmpty());
	}

	@Test
	public void testAddFood() {
		MealListPublic mlp = new MealListPublic();
		mlp.addFoodName("beer");
		assertEquals(mlp.getMealList().get(0), "beer");
	}

	//If all of those fails then probably getMealList is broken
}
