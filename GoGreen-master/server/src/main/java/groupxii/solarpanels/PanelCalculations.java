package groupxii.solarpanels;

import groupxii.database.Database;
import groupxii.database.PanelListEntry;


public class PanelCalculations {

    /**
     * This is the first calculation that is done. the Chosen panel is looked up in
     * the panelList and then the private values are updated and the CO2 emission is calculated.
     */
    public static int calculateCO2(String chosenPanel, int chosenamount) {
        PanelListEntry panelListEntry = Database.instance.findPanelListEntry(chosenPanel);
        int result = -1;
        double co2 = panelListEntry.getCo2PerPanel();
        int amount = panelListEntry.getAmount();
        result = (int) (co2 * amount);
        return result;
    }
}
