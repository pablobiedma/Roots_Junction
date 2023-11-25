package groupxii.database;

import java.util.ArrayList;
import java.util.List;

/**
 * A list of all PanelList panel names for convenient send to the client.
 */
public class PanelListPublic {
    private List<String> panelList;

    public PanelListPublic() {
        panelList = new ArrayList<>();
    }

    public List<String> getPanelList() {
        return this.panelList;
    }

    public void addPaneltype(String paneltype) {
        panelList.add(paneltype);
    }
}
