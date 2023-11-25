package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.PanelEntry;
import groupxii.database.UserEntry;
import groupxii.solarpanels.CalculatedPanel;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.List;

@RestController
public class SolarPanelController {

    /**
     * Return a list of all available panel entries that the solar panel feature can process.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/getPanelList")
    public List<String> getNameList() {
        return Database.instance.getPanelListPanelNames();
    }

    /**
     * Calculate the saved CO2 and send the response to the server.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/calculatePanel")
    public CalculatedPanel calculatePanel(@RequestParam(value = "paneltype",
            defaultValue = "Unknown") String paneltype,
                                          @RequestParam(value = "paneltype",
                                                  defaultValue = "0") Integer amount) {
        return new CalculatedPanel(paneltype, amount);
    }

    /**
     * the client can send data to the server with the right values as parameter,
     * then this method will store the data in the database.
     */
    @RequestMapping(method = RequestMethod.POST, value = "/savePanelData")
    public void savePanelData(@RequestParam(value = "paneltype", defaultValue = "Unknown")
                                          String paneltype,
                              @RequestParam(value = "efficiencyrate", defaultValue = "-1")
                                      Integer efficiencyrate,
                              @RequestParam(value = "amount", defaultValue = "-1")
                                      Integer amount, Principal principal) {
        String username = principal.getName();
        PanelEntry panelEntry = new PanelEntry(paneltype, efficiencyrate, amount);
        Database.instance.addUsedPanel(username, panelEntry);
    }

    /**
     * Given the principal, return the used panel list.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/getUsedPanelList")
    public List<PanelEntry> getUsedPanelList(Principal principal) {
        String username = principal.getName();
        UserEntry user = Database.instance.findUserByName(username);
        List<PanelEntry> usedPanels = user.getUsedPanels();
        return usedPanels;
    }

}
