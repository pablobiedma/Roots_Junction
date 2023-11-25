package groupxii.database;

import com.mongodb.DBObject;
import groupxii.solarpanels.CalculatedPanel;

public class PanelEntry extends Entry {

    /**
     * Represents entries in the database which tracks
     * which solarpanel each request indicates.
     */
    private String paneltype;
    private int efficiencyrate;
    private int reducedco2;
    private int amount;

    /**
     * Constructor of PanelEntry.
     */
    public PanelEntry(String paneltype, int efficiencyrate, int amount) {
        this.paneltype = paneltype;
        this.efficiencyrate = efficiencyrate;
        this.reducedco2 = new CalculatedPanel(paneltype, amount).getReducedCO2();
        this.amount = amount;
    }

    public String getPaneltype() {

        return paneltype;
    }

    public int getEfficiencyrate() {

        return  efficiencyrate;
    }

    public int getReducedco2() {

        return reducedco2;
    }

    public int getAmount() {

        return amount;
    }


    /**
     * Translates into a MongoDB JSON object.
     */
    public final DBObject toDbObject() {

        return super.toBasicDbObject()
                .append("paneltype",this.paneltype)
                .append("reducedco2", this.reducedco2)
                .append("efficiencyrate",this.efficiencyrate)
                .append("amount",this.amount);

    }
}
