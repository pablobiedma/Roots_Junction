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

@RunWith(PowerMockRunner.class)
@PrepareForTest({VegetarianMealConnector.class})

public class ReducedCo2Test {

    @Test
    public void getReducedCo2Test() {
		PowerMockito.mockStatic(VegetarianMealConnector.class);
		PowerMockito.when(
						VegetarianMealConnector.calculateCO2Reduction(anyString(), anyInt(),
						                                             anyString(), anyInt()))
					.thenReturn("{\"reducedCO2\":\"120\"}");

		ReducedCo2 rco2 = new ReducedCo2();
		assertEquals("120", rco2.getReducedCo2("food", 3, "food2", 2));

    }
}
