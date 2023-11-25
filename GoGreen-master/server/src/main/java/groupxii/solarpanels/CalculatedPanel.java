package groupxii.solarpanels;


public class CalculatedPanel {
    int reducedCO2;

    /**
     * Constructor of CalculatedPanel.
     */
    public CalculatedPanel(String paneltype, int amount) {
        this.reducedCO2 = 1;
        int result =  PanelCalculations.calculateCO2(paneltype, amount);
        this.reducedCO2 = result;
    }

    public int getReducedCO2() {
        return this.reducedCO2;
    }
}
