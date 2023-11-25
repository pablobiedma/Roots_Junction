package groupxii.solarpanels;


import groupxii.database.Database;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;


public class PanelCalculationsTest {
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
        assertEquals(1571,PanelCalculations.calculateCO2("Monocrystalline", 45));
    }

    @Test
    public void testInstantiateId() {
        PanelCalculations panelCalculations = new PanelCalculations();
        assertEquals(1571,PanelCalculations.calculateCO2("Monocrystalline", 45));

    }
}
