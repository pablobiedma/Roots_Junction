package groupxii.database;



import com.mongodb.DBObject;
import groupxii.solarpanels.CalculatedPanel;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.*;


public class PanelEntryTest {

    PanelEntry panelEntry;
    @Before
    public void createEntry() {
        try {
            Database.instance.startDb();
        }
        catch (IOException e) {
            assertTrue(false);
        }
        panelEntry = new PanelEntry("Monocrystalline", 45,23);

    }

    @Test
    public void testPaneltype() {
        assertEquals("Monocrystalline", panelEntry.getPaneltype());
    }

    @Test
    public void testEfficiencyrate() {
        assertEquals(45, panelEntry.getEfficiencyrate());
    }

    @Test
    public void testAmount() {
        assertEquals(23, panelEntry.getAmount());
    }

    @Test
    public void testReducedCO2() {
        CalculatedPanel calculatedPanel = new CalculatedPanel("Monocrystalline", 23);
        assertEquals(calculatedPanel.getReducedCO2(),panelEntry.getReducedco2());
    }

    @Test
    public void testToDBObject() {
        DBObject obj = panelEntry.toDbObject();
        assertTrue(obj.containsField("paneltype"));
        assertTrue(obj.containsField("reducedco2"));
        assertTrue(obj.containsField("efficiencyrate"));
        assertTrue(obj.containsField("amount"));
    }


}
