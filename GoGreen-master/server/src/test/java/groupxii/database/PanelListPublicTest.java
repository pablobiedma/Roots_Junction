package groupxii.database;

import org.junit.Test;

import static org.junit.Assert.*;

public class PanelListPublicTest {

    @Test
    public void testConstructor() {
        PanelListPublic panelListPublic = new PanelListPublic();
        assertTrue(panelListPublic.getPanelList().isEmpty());
    }

    @Test
    public void getPanelList() {
        PanelListPublic panelListPublic = new PanelListPublic();
        PanelListPublic panelListPublic2 = new PanelListPublic();
        panelListPublic.addPaneltype("Monocrystalline");
        panelListPublic2.addPaneltype("Monocrystalline");
        assertEquals(panelListPublic.getPanelList(),panelListPublic2.getPanelList());
    }

    @Test
    public void  testAddPanelType() {
        PanelListPublic panelListPublic = new PanelListPublic();
        panelListPublic.addPaneltype("Monocrystalline");
        assertEquals(panelListPublic.getPanelList().get(0),"Monocrystalline");
    }
}
