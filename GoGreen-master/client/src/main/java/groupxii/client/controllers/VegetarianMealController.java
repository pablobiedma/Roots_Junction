package groupxii.client.controllers;

import groupxii.client.Main;
import groupxii.client.connector.VegetarianMealConnector;
import groupxii.client.vegetarianmeal.EatenMealList;
import groupxii.client.vegetarianmeal.MealList;
import groupxii.client.vegetarianmeal.ReducedCo2;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.ListView;
import javafx.scene.control.Slider;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;

import java.io.IOException;
import java.net.URL;
import java.util.ResourceBundle;

public class VegetarianMealController implements Initializable {

    @FXML
    private ChoiceBox<String> choiceBoxGoodFood = new ChoiceBox();

    @FXML
    private ChoiceBox<String> choiceBoxBadFood = new ChoiceBox();

    @FXML
    private Slider sliderGoodFood = new Slider();

    @FXML
    private Slider sliderBadFood = new Slider();

    @FXML
    private Text reducedCo2Text = new Text();

    @FXML
    private ListView<String> eatenMealsListView = new ListView();

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        MealList mealList = new MealList();
        ObservableList<String> listObservable
                = FXCollections.observableArrayList(mealList.getMealList());
        choiceBoxGoodFood.setItems(listObservable);
        choiceBoxBadFood.setItems(listObservable);
        updateListView();
    }

    /**
     * updates the list view with all the items from the database.
     */

    @FXML
    public void updateListView() {
        ObservableList<String> eatenMealObservableList
                = FXCollections.observableArrayList(EatenMealList.getEatenMealList());
        eatenMealsListView.setItems(eatenMealObservableList);
    }

    /**
     * Sends request to calculate a meal and displays the result.
     */
    @FXML
    public void calculateMeal(MouseEvent event) {
        String goodFoodName = choiceBoxGoodFood.getValue();
        int goodServingSize = (int)sliderGoodFood.getValue();
        String badFoodName = choiceBoxBadFood.getValue();
        int badServingSize = (int)sliderBadFood.getValue();

        String result = ReducedCo2.getReducedCo2(goodFoodName,
                                                 goodServingSize,
                                                 badFoodName,
                                                 badServingSize);

        reducedCo2Text.setText("This will reduce " + result + " grams of CO2");
    }

    /**
     * Calculates the co2 reduced emission when the user clicks the calculate button.
     * @param event mouse click
     * @throws Exception throws exception when something went wrong
     */
    @FXML
    public void safeMeal(MouseEvent event) throws Exception {
        String goodFoodName = choiceBoxGoodFood.getValue();
        int goodServingSize = (int)sliderGoodFood.getValue();
        String badFoodName = choiceBoxBadFood.getValue();
        int badServingSize = (int)sliderBadFood.getValue();

        VegetarianMealConnector.commitMeal(goodFoodName,
                                           goodServingSize,
                                           badFoodName,
                                           badServingSize);

        reducedCo2Text.setText("Enjoy your meal :-)");
        updateListView();
    }

    @FXML
    public void btnBack(MouseEvent event) throws IOException {
        Main main = new Main();
        main.changeScene("Menu.fxml", event);
    }
}
