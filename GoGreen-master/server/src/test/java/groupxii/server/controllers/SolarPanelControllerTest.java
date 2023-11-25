package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.PanelEntry;
import groupxii.database.UserEntry;
import groupxii.solarpanels.CalculatedPanel;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.security.Principal;
import java.util.List;

import static org.junit.Assert.*;

public class SolarPanelControllerTest {

    SolarPanelController solarPanelController;
    UserEntry usr;
    @Before
    public void createSPC() {
        try {
            Database.instance.startDb();
        } catch (IOException e) {
            assertTrue(false);
        }
        solarPanelController = new SolarPanelController();
        usr = new UserEntry(1,"Ivan","pass");
        Database.instance.saveNonBlocking(usr);
    }

    @Test
    public void calculatePanelTest() {
        CalculatedPanel calculatedPanel = new CalculatedPanel("Monocrystalline", 44);
        CalculatedPanel calculatedPanel2 = solarPanelController.calculatePanel("Monocrystalline", 44);
        assertEquals(calculatedPanel.getReducedCO2(),calculatedPanel2.getReducedCO2());
    }

    @Test
    public void getNameList() {
        List<String> namelist = solarPanelController.getNameList();
        assertFalse(namelist.isEmpty());
    }

    @Test
    public void savePanelData() {
        PanelEntry panelEntry  = new PanelEntry("Monocrystalline", 34,45);
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        Database.instance.addUsedPanel(username, panelEntry);
        solarPanelController.savePanelData("Monocrystalline", 34,45,principal);
        assertEquals(solarPanelController.getNameList(),Database.instance.getPanelListPanelNames());
    }

    @Test
    public void getUsedPanelList() {
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        UserEntry userEntry = Database.instance.findUserByName(username);
        List<PanelEntry> panelEntries = solarPanelController.getUsedPanelList(principal);
        assertEquals(userEntry.getUsedPanels(),panelEntries);
    }
}
