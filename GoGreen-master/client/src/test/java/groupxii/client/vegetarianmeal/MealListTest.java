package groupxii.client.vegetarianmeal;

import groupxii.client.connector.VegetarianMealConnector;

import org.junit.runner.RunWith;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import static org.mockito.Mockito.*;

import java.util.ArrayList;
import java.util.List;

@RunWith(PowerMockRunner.class)
@PrepareForTest({VegetarianMealConnector.class})

public class MealListTest {

    @Test
    public void getReducedCo2Test() {
		PowerMockito.mockStatic(VegetarianMealConnector.class);
		PowerMockito.when(
						VegetarianMealConnector.retrieveMealList())
					.thenReturn("[\"APPLE\", \"GRAPEFRUIT\"]");

		MealList ml = new MealList();

		List<String> expected = new ArrayList<String>();
		expected.add("APPLE");
		expected.add("GRAPEFRUIT");

		assertEquals(expected, ml.getMealList());

    }
}
