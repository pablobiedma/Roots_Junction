package groupxii.client.controllers;

import com.maxmind.geoip2.exception.GeoIp2Exception;
import groupxii.client.Main;
import groupxii.client.connector.LocalProductsConnector;
import groupxii.client.connector.UserConnector;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.ListView;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;

import java.awt.Desktop;
import java.io.IOException;
import java.net.URI;
import java.util.Arrays;
import java.util.List;

public class LocalProductsController {

    @FXML
    private Text textfield = new Text();

    @FXML
    private ListView<String> localShops = new ListView<String>();

    @FXML
    private ImageView mapsImage = new ImageView();

    private int userId = 1;

    /**
     * Initializes the local products page.
     */
    public void initialize() {
        String listItemsStr = LocalProductsConnector.retrieveLocalShops();
        List<String> listViewItems = Arrays.asList(listItemsStr.split(", "));
        ObservableList<String> listViewObservable =
                FXCollections.observableArrayList(listViewItems);
        localShops.setItems(listViewObservable);
    }

    /**
     * gets an image of the location of the shop.
     * @param event mouse click.
     * @throws GeoIp2Exception if the location can not be initialized.
     * @throws IOException if the connection cannot be made.
     */
    @FXML
    public void getLocation(MouseEvent event) throws GeoIp2Exception, IOException {
        String url = getShopLocation();
        if (url == null) {
            textfield.setText("Please choose a shop first");
        }  else {
            mapsImage.setImage(new Image("https://maps.googleapis.com/maps/api/staticmap?center=" + url + "&zoom=12&size=900x150&maptype=roadmap&markers=color:red%7Clabel:Shop%7C" + url + "&key=AIzaSyBvn_zZpLGUjLJBxIUoGHgJjzo2VlZm3jg"));
        }
    }

    /**
     * opens the browser and navigates the user to the shop.
     * @param event mouse click.
     * @throws Exception if the connection to the url cannot be made.
     */
    @FXML
    public void navigate(MouseEvent event) throws Exception {
        String url = getShopLocation();
        if (localShops.getSelectionModel().getSelectedItem() == null) {
            textfield.setText("Please choose a shop first");
        } else {
            Desktop.getDesktop().browse(URI.create("https://www.google.com/maps/search/?api=1&query=" + url));
        }
    }

    /**
     * gets the adress of the shop from the listview.
     * @return String with the adress of the shop.
     */
    public String getShopLocation() {

        if (localShops.getSelectionModel().getSelectedItem() == null) {
            return "server is not running";
        } else {
            String locationStr = localShops.getSelectionModel().getSelectedItem();
            locationStr = locationStr.substring(locationStr.indexOf("LOCATED AT: "),
                    locationStr.indexOf(" - RATING:"));
            locationStr = locationStr.replace(' ', '+').substring(0, locationStr.length() - 1);
            return locationStr;
        }
    }

    /**
     * When a shop is chosen it saves the saved Co2 to the user's score.
     */
    public void safeLocalProduct() {
        if (localShops.getSelectionModel().getSelectedItem() == null) {
            textfield.setText("Please choose a shop first");
        } else {
            UserConnector.updateReducedCo2(300);
            textfield.setText("You saved 300 grams of CO2 emission!");
        }
    }

    @FXML
    public void btnBack(MouseEvent event) throws IOException {
        Main main = new Main();
        main.changeScene("Menu.fxml", event);
    }
}
