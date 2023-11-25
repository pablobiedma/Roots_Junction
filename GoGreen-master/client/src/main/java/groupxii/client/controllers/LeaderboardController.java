package groupxii.client.controllers;

import groupxii.client.Main;
import groupxii.client.connector.LeaderboardConnector;
import groupxii.client.leaderboard.JsonConverter;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.text.Text;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LeaderboardController {

    private static int userId = 1;

    @FXML
    private Text addedFriend = new Text();

    private String overallListStr = "";
    private String friendListStr = "";
    //Should be the userId that is assigned to the user when he/she registers

    @FXML
    private ListView<HBoxCell> overallLeaderboard = new ListView();

    @FXML
    private ListView friendsLeaderboard = new ListView();

    public class HBoxCell extends HBox {
        Label label = new Label();
        Button button = new Button();

        HBoxCell(String labelText, String buttonText, int friendId) {
            super();

            label.setText(labelText);
            label.setMaxWidth(Double.MAX_VALUE);
            HBox.setHgrow(label, Priority.ALWAYS);
            button.setOnAction(new EventHandler<ActionEvent>() {
                @Override
                public void handle(ActionEvent event) {
                    addFriend(friendId);
                }
            });
            button.setText(buttonText);

            this.getChildren().addAll(label, button);
        }
    }

    /**
    Initializes the leaderboard page.
     */
    public void initialize() {

        /*
        try {
            List<DBObject> overallListJsonStr = (DBObject) JSON.
                    parse(new Scanner(new URL(host + "Leaderboard").openStream(),
                    "UTF-8").nextLine());
            String friendListJsonStr = new Scanner(new URL(host
                    + "getFriends?Id="
                    + userId).openStream(),
                    "UTF-8").nextLine();
            JsonConverter jsonConverter = new JsonConverter();
            overallListStr = jsonConverter.
                    leaderboardToString(overallListJsonStr)
        } catch (IOException e) {
            e.printStackTrace();
        }
        */

        overallListStr = JsonConverter
                .leaderboardToString(LeaderboardConnector.retrieveLeaderboard());
        List<String> overallLeaderboardList = Arrays.asList(overallListStr.split(","));
        List<HBoxCell> overallList = new ArrayList<>();
        for (int i = 0; i < overallLeaderboardList.size(); i++) {
            overallList.add(new HBoxCell(overallLeaderboardList.get(i),
                    "ADD FRIEND",
                    Integer.parseInt(overallLeaderboardList.get(i + 1))));
            i++;
        }
        ObservableList<HBoxCell> overallLeaderboardObservableList;
        overallLeaderboardObservableList = FXCollections.observableArrayList(overallList);
        overallLeaderboard.setItems(overallLeaderboardObservableList);

        friendListStr = JsonConverter
                .leaderboardToString(LeaderboardConnector.retrieveFriends(userId));
        List<String> friendsLeaderboardList = Arrays.asList(friendListStr.split(","));

        //TODO Take a look at how to unfriend someone, (will fix this if we have time enough)
        List<HBoxCell> friendList = new ArrayList<>();
        for (int i = 0; i < friendsLeaderboardList.size(); i++) {
            friendList.add(new HBoxCell(friendsLeaderboardList.get(i),
                    "UNFOLLOW",
                    Integer.parseInt(overallLeaderboardList.get(i + 1 ))));
            i++;
        }
        ObservableList<HBoxCell> friendsLeaderboardObservableList;
        friendsLeaderboardObservableList = FXCollections.observableArrayList(friendList);
        friendsLeaderboard.setItems(friendsLeaderboardObservableList);
        //System.out.println("Overall List:   " + overallLeaderboardList);
        //System.out.println("Friend List:   " + friendsLeaderboardList);
        //ObservableList<String> friendsLeaderboardObservableList =
        // FXCollections.observableArrayList(friendsLeaderboardList);
        //friendsLeaderboard.setItems(friendsLeaderboardObservableList);
    }

    /**
     * Receives an id of an user and adds him/her to the friend list.
     */
    public String addFriend(int friendId) {
        LeaderboardConnector.addFriend(userId, friendId);
        addedFriend.setText("Added new friend!");
        initialize();
        return "Added new friend!";
    }

    @FXML
    public void btnBack(MouseEvent event) throws IOException {
        Main main = new Main();
        main.changeScene("Menu.fxml", event);
    }
}
