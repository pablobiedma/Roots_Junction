package groupxii.server.controllers;

import groupxii.database.Database;
import groupxii.database.UserEntry;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.ArrayList;
import java.util.List;

@RestController
public class UserController {


    /**
     * Receives data and creates a user entry in the user collection.
     */
    @RequestMapping(method = RequestMethod.POST, value = "/register")
    public void saveUserData(@RequestParam(value = "username",
        defaultValue = "") String username,
                              @RequestParam(value = "password",
                                      defaultValue = "")
                                      String password)  {
        //TODO error handling, return responses
        if (username.isEmpty() || password.isEmpty()) {
            return;
        }

        //TODO verify name is unique
        /*
        if (Database.instance.findUserByName(username)) {
            return;
        }
        */

        int userId = Database.instance.getUserCount() + 1;
        UserEntry userEntry = new UserEntry(userId, username, password);
        Database.instance.save(userEntry);
    }

    /**
     * Receives user's id and returns the list
     * of his/her friends.
     */
    //Isn't this unnecessary slow?
    @RequestMapping(method = RequestMethod.GET, value = "/getFriends")
    public List<UserEntry> getFriends(Principal principal) {
        String username = principal.getName();
        UserEntry user =  Database.instance.findUserByName(username);
        List<Integer> friendsIdList = user.getFriendsId();

        List<UserEntry> friends = new ArrayList<>();
        for (int i = 0; i < friendsIdList.size(); i++) {
            UserEntry friend =  Database.instance.findUserById(friendsIdList.get(i));
            friends.add(friend);
        }
        return friends;
    }

    /**
     * returns all users sorted by points.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/Leaderboard")
    public List<UserEntry> leaderboard() {
        List<UserEntry> users = Database.instance.sortUsersByReducedCo2();
        return users;
    }

    /**
     * receives two id's and adds the second one as a friend to the first one.
     */
    //TODO why does it take the userID of the friend but not of the user itself?
    // on the client everything is based on the userID since
    // that was the first intention and is much better.
    @RequestMapping(method = RequestMethod.POST, value = "/addFriend")
    public void addFriend(Principal principal,
                          @RequestParam(value = "newFriend",
                                  defaultValue = "Unknown") int friendsId) {
        String user = principal.getName();
        Database.instance.addFriend(user, friendsId);
    }

    /**
     * Returns the reduced Co2(points) of a particular user.
     */
    @RequestMapping(method = RequestMethod.GET, value = "/getReducedCo2OfUser")
    public int getReducedCo2OfUser(Principal principal) {
        String username = principal.getName();
        UserEntry user = Database.instance.findUserByName(username);
        int reducedCo2 = user.getReducedCo2();
        return reducedCo2;
    }

    /**
     * Increases the reduced Co2 of an user by a given amount.
     */
    @RequestMapping(method = RequestMethod.POST, value = "/increaseReducedCo2OfUser")
    public int increaserReducedCo2OfUser(Principal principal,
                                         @RequestParam(value = "amount",
                                                 defaultValue = "0") int amount) {
        String username = principal.getName();
        //UserEntry user = Database.instance.findUserByName(username);
        Database.instance.incrementReducedCo2(username, amount);
        return amount;
    }


}


