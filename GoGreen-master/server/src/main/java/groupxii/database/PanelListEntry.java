package groupxii.database;

import com.mongodb.DBObject;
import org.bson.BSONObject;

public class PanelListEntry extends  Entry {

    private String paneltype;
    private double co2PerPanel;
    private int efficiencyrate;
    private int amount;


    /**
     * Representation of a single entry from the PanelList file.
     */
    public PanelListEntry(String paneltype, double co2PerPanel,int efficiencyrate , int amount) {
        super();
        this.paneltype = paneltype;
        this.co2PerPanel = co2PerPanel;
        this.efficiencyrate = efficiencyrate;
        this.amount = amount;
    }

    /**
     * Constructs a PanelListEntry from a BSONObject, usually a result from query.
     */
    public PanelListEntry(BSONObject object) {
        if (object == null) {
            return;
        }
        if (object.containsField("paneltype")) {
            this.paneltype = (String) object.get("paneltype");
        }
        if (object.containsField("co2PerPanel")) {
            this.co2PerPanel = (double) object.get("co2PerPanel");
        }
        if (object.containsField("efficiencyrate")) {
            this.efficiencyrate = (int) object.get("efficiencyrate");
        }
        if (object.containsField("amount")) {
            this.amount = (int) object.get("amount");
        }
    }

    public double getCo2PerPanel() {
        return this.co2PerPanel;
    }

    public int getAmount() {
        return this.amount;
    }

    public String getPaneltype() {
        return this.paneltype;
    }

    public int getEfficiencyrate() {
        return efficiencyrate;
    }

    /**
     * Translate into a MongoDb JSON object.
     */
    public DBObject toDbObject() {
        return super.toBasicDbObject()
                .append("paneltype",this.paneltype)
                .append("co2PerPanel",this.co2PerPanel)
                .append("efficiencyrate",this.efficiencyrate)
                .append("amount",this.amount);

    }
}
