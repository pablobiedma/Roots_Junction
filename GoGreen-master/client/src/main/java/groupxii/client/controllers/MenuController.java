package groupxii.client.controllers;

import groupxii.client.Main;
import groupxii.client.connector.Connector;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.ProgressBar;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;

import java.net.URL;
import java.util.ResourceBundle;

public class MenuController implements Initializable {

    private Main main = new Main();

    @FXML
    private Text reducedCo2Text = new Text();

    @FXML
    private Text trees = new Text();

    @FXML
    private ProgressBar levelProgress = new ProgressBar();

    @FXML
    private Text level = new Text();

    /**
     * switches to the vegetarian meal feature.
     * @param event mouse click
     * @throws Exception throws IO Exception when something goes wrong
     */

    @FXML
    public void btnVegetarianMeal(MouseEvent event) throws Exception {
        main.changeScene("VegetarianMeal.fxml", event);
    }

    @FXML
    public void btnSolarPanels(MouseEvent event) throws Exception {
        main.changeScene("SolarPanels.fxml", event);
    }

    @FXML
    public void btnTransport(MouseEvent event) throws Exception {
        main.changeScene("Transportation.fxml", event);
    }

    @FXML
    public void btnComingSoon(MouseEvent event) throws Exception {
        main.changeScene("VegetarianMeal.fxml", event);
    }

    @FXML
    public void btnLeaderBoard(MouseEvent event) throws Exception {
        main.changeScene("Leaderboard.fxml", event);
    }

    @FXML
    public void btnLocalProducts(MouseEvent event) throws Exception {
        main.changeScene("LocalProducts.fxml", event);
    }

    @FXML
    public void btnTemperature(MouseEvent event) throws Exception {
        main.changeScene("Temperature.fxml", event);
    }

    @FXML
    public void logout(MouseEvent event) throws Exception {
        main.changeScene("Login.fxml", event);
    }

    @Override
    public void initialize(URL url, ResourceBundle resourceBundle) {
        String reducedCo2 = Connector.getRequest("/getReducedCo2OfUser");
        //What are all this things?
        reducedCo2Text.setText(reducedCo2);
        double reducedCo2Number = Double.parseDouble(reducedCo2);
        double savedTrees = reducedCo2Number / 24500;
        trees.setText(Double.toString(savedTrees).substring(0,3));
        int levelNumber = (int) reducedCo2Number / 10000;
        level.setText(Integer.toString(levelNumber));
        double levelProgressNumber = (reducedCo2Number % 10000) / 10000;
        levelProgress.setProgress(levelProgressNumber);
    }

}
