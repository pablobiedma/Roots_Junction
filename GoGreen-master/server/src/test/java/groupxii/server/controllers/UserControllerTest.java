package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.UserEntry;
import groupxii.server.controllers.UserController;

import org.mockito.Mockito;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.security.Principal;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

public class UserControllerTest {

    UserController controller;
    Principal mocked;

    @Before
    public void init() throws IOException {
        Database.instance.setDbName("testController");
        Database.instance.startDb();

        controller = new UserController();
	mocked = Mockito.mock(Principal.class);
        Mockito.when(mocked.getName()).thenReturn("user");
    }

    @Test
    public void saveUserTest() throws IOException {
	controller.saveUserData("user", "pass");
	//nothing happens => pass
    }

    @Test
    public void saveUserNoPassTest() throws IOException {
	controller.saveUserData("user", "");
	//nothing happens => pass
    }

    @Test
    public void saveUserNoNameTest() throws IOException {
	controller.saveUserData("", "pass");
	//nothing happens => pass
    }

    @Test
    public void addFriendTest() {
	    controller.addFriend(mocked, 0);
    }

    @Test
    public void getFriendListTest() {
	    controller.getFriends(mocked);
    }
	    

//    @Test
//    public void getFriendsTest() throws IOException {
//        UserController usr = new UserController();
//        List<Integer> list = new ArrayList<>();
//        List<String> list2 = new ArrayList<>();
//        UserEntry entry = usr.saveUserData("Ivan",100,2,12,list);
//        assertEquals(list2,usr.getFriends(1));
//    }
}
