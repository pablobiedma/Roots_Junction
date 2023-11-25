package groupxii.solarpanels;


import groupxii.database.Database;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;

public class CalculatedPanelTest {

    @Before
    public void startDB() {
        try{
            Database.instance.startDb();
        } catch (IOException e) {
            assertTrue(false);
        }
    }

    @Test
    public void testCalculatePanel() {
        CalculatedPanel calculatedPanel = new CalculatedPanel("Thin-Film", 34);
        assertEquals(955,calculatedPanel.getReducedCO2());
    }


}
