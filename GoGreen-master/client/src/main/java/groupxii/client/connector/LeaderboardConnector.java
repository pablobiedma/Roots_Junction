package groupxii.client.connector;

public class LeaderboardConnector {

    public static String retrieveLeaderboard() {
        String resource = "/Leaderboard";
        return Connector.getRequest(resource);
    }

    public static String retrieveFriends(int userId) {
        String resource = "/getFriends?Id=" + userId;
        return Connector.getRequest(resource);
    }

    public static String addFriend(int userId, int friendId) {
        String resource = "/addFriend?newFriend=" + friendId + "&userId=" + userId;
        return Connector.postRequest(resource);
    }
}
