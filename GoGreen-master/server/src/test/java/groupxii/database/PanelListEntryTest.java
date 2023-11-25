package groupxii.database;


import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class PanelListEntryTest {
    PanelListEntry panelListEntry;
    @Before
    public void createEntry() {
        panelListEntry = new PanelListEntry("Monocrystalline", 45.0, 32, 44);
    }

    @Test
    public void testPanelType() {
        assertEquals("Monocrystalline", panelListEntry.getPaneltype());
    }

    @Test
    public void testCo2PerPanel() {
        assertEquals(panelListEntry.getCo2PerPanel(), 45.0,0);
    }

    @Test
    public void testEfficiencyrate() {
        assertEquals(32,panelListEntry.getEfficiencyrate());
    }

    @Test
    public void testAmount() {
        assertEquals(44, panelListEntry.getAmount());
    }

    @Test
    public void testToDBObject() {
        DBObject obj = panelListEntry.toDbObject();
        assertTrue(obj.containsField("paneltype"));
        assertTrue(obj.containsField("co2PerPanel"));
        assertTrue(obj.containsField("efficiencyrate"));
        assertTrue(obj.containsField("amount"));
    }

    @Test
    public void testFromDBObject() {
        DBObject dbObject = new BasicDBObject()
                .append("paneltype","Monocrystalline")
                .append("co2PerPanel",45.0)
                .append("efficiencyrate",32)
                .append("amount",44);
        panelListEntry = new PanelListEntry(dbObject);
        testToDBObject();
    }

    @Test
    public void testFromNullObj() {
        DBObject obj = null;
        panelListEntry = new PanelListEntry(obj);
        assertNull(panelListEntry.getPaneltype());
        assertEquals(panelListEntry.getCo2PerPanel(), 0.0, 0);
        assertEquals(panelListEntry.getEfficiencyrate(), 0);
        assertEquals(panelListEntry.getAmount(),  0);
    }

    @Test
    public void testFromBadDBObject() {
        DBObject obj = new BasicDBObject()
                .append("type", "Monocrystalline")
                .append("co2", 45.0)
                .append("eff", 32)
                .append("howMuch", 44);
        panelListEntry = new PanelListEntry(obj);
        assertNull(panelListEntry.getPaneltype());
        assertEquals(panelListEntry.getCo2PerPanel(), 0.0, 0);
        assertEquals(panelListEntry.getEfficiencyrate(), 0);
        assertEquals(panelListEntry.getAmount(),  0);
    }




}
