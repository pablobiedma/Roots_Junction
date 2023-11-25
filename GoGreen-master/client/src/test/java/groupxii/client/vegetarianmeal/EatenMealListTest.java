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

public class EatenMealListTest {

    @Test
    public void getReducedCo2Test() {
		PowerMockito.mockStatic(VegetarianMealConnector.class);
		PowerMockito.when(
						VegetarianMealConnector.retrieveEatenMealList())
					.thenReturn("[{\"goodFoodName\":\"APPLE\", \"reducedCo2\":\"120\"}, {\"goodFoodName\":\"GRAPEFRUIT\", \"reducedCo2\":\"120\"}]");

		EatenMealList eml = new EatenMealList();

		List<String> expected = new ArrayList<String>();
		expected.add("Ate a APPLE and saved: 120 of Co2!");
		expected.add("Ate a GRAPEFRUIT and saved: 120 of Co2!");

		assertEquals(expected, eml.getEatenMealList());

    }
}
