package groupxii.client.controllers;

import groupxii.client.Main;
import groupxii.client.connector.TransportConnector;
import groupxii.client.transportation.UsedTransportList;
import groupxii.client.transportation.VehicleNameList;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ListView;
import javafx.scene.control.Slider;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;
//import javafx.scene.text.Text;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;


public class TransportationController implements Initializable {

    /*
    @FXML
    private Button btnCalculateTransport = new Button();

    @FXML
    private Button btnSafeTransport = new Button();
    */

    @FXML
    private ChoiceBox<String> cb = new ChoiceBox();

    @FXML
    private ChoiceBox<String> cb1 = new ChoiceBox();

    @FXML
    private Text reducedCo2Text = new Text();

    @FXML
    private Slider slider = new Slider();

    @FXML
    private Slider slider1 = new Slider();

    @FXML
    private Text savedCo2 = new Text();

    @FXML
    private ListView<String> usedTransportListView = new ListView();

    //    @FXML
    //    private ListView<String> usedTransportListView = new ListView();
    UsedTransportList usedTransportList = new UsedTransportList();

    @Override
    public void initialize(URL location, ResourceBundle resources) {

        VehicleNameList vehicleNameList = new VehicleNameList();
        ObservableList<String> listObservable;
        listObservable = FXCollections.observableArrayList(vehicleNameList.getVehicleNameList());
        cb1.getItems().addAll(listObservable);
        cb.getItems().addAll(listObservable);
        updateListView();
    }

    /**
     * updates the listview with all the items from the database.
     */
    @FXML
    public void updateListView() {
        usedTransportList.setUsedTransportList();
        ObservableList<String> usedTransportObservableList = FXCollections
                .observableArrayList(usedTransportList.getEatenMealList());
        usedTransportListView.setItems(usedTransportObservableList);


        //        try {
        //        usedTransportListStr =
        //        new Scanner(new URL(host + "usedTransportList").openStream(),
        //                    "UTF-8").nextLine();
        //        } catch (IOException e) {
        //            e.printStackTrace();
        //        }
        //        usedTransportListViewItems = Arrays.asList(usedTransportListStr.split(" - "));
        //        ObservableList<String> usedTransportObservable =
        //                FXCollections.observableArrayList(usedTransportListViewItems);
        //        usedTransportListView.setItems(usedTransportObservable);
    }

    //TODO
    @FXML
    public void calculateTransport(MouseEvent event) {

    }

    /**
     * Calculates the co2 reduced emmision when the user clicks the calculate button.
     * it also calculates the amount of points the user earned.
     *
     * @param event mouse click
     * @throws Exception throws exception when something went wrong
     */
    @FXML
    public void safeTransport(MouseEvent event) throws Exception {
        //        badTransportName = cb1.getValue();
        //        goodTransportName = cb.getValue();
        //        goodfuel = cb3.getValue();
        //        badfuel = cb4.getValue();
        //        int goodConsumption = (int) slider.getValue();
        //        int badConsumption = (int) slider1.getValue();
        //        safeTransport.safeVehicle(goodTransportName, badTransportName,goodfuel,
        //                badfuel,goodConsumption , badConsumption);
        String goodTransportName = cb.getValue();
        int goodConsumption = (int) slider.getValue();
        String badTransportName = cb1.getValue();
        int badConsumption = (int) slider1.getValue();


        String result;
        result = TransportConnector.calculateCO2Reduction(goodTransportName,
                 badTransportName, goodConsumption, badConsumption);
        //TODO update contoller to display result
        updateListView();
    }

    @FXML
    public void btnBack(MouseEvent event) throws IOException {
        Main main = new Main();
        main.changeScene("Menu.fxml", event);
    }
}
