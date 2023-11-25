package groupxii.client.controllers;

import groupxii.client.Main;
import groupxii.client.TokenManager;
import groupxii.client.connector.LoginConnector;
import javafx.fxml.FXML;
import javafx.scene.control.PasswordField;
import javafx.scene.control.TextField;
import javafx.scene.input.MouseEvent;
import javafx.scene.text.Text;

public class LoginController {
    @FXML
    private TextField txtUsername = new TextField();

    @FXML
    private PasswordField txtPassword = new PasswordField();

    @FXML
    private Text errorMessage = new Text();

    /**
     * Login button functionality.
     */
    @FXML
    public void loginButton(MouseEvent event) throws Exception {
        if (txtUsername.getText().isEmpty() || txtPassword.getText().isEmpty()) {
            errorMessage.setText("Please enter a username and/or password.");
        } else {
            String username = txtUsername.getText();
            String password = txtPassword.getText();
            // Initializa TokenManager
            TokenManager.instance = new TokenManager(username, password);
            TokenManager.instance.refreshToken();

            //TODO better error checks
            if (TokenManager.instance.getToken() != null) {
                Main main = new Main();
                main.changeScene("Menu.fxml", event);
            } else {
                errorMessage.setText("Could not authenticate, "
                        + "please check your username/password or network connection");
            }
        }
    }

    /**
     * Sign up(register) button functionality.
     */
    @FXML
    public void signUpButton(MouseEvent event) {
        if (txtUsername.getText().isEmpty() || txtPassword.getText().isEmpty()) {
            errorMessage.setText("Please enter a username and/or password.");
        } else {
            String username = txtUsername.getText();
            String password = txtPassword.getText();

            //This should not stay like that
            LoginConnector.register(username, password);
        }
    }
}

