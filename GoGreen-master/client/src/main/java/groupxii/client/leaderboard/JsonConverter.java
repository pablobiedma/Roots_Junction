package groupxii.client.leaderboard;

import org.json.JSONArray;
import org.json.JSONObject;

public class JsonConverter {

    /**
     * converts the JSON from the server into a client readable String.
     * @param jsonList Leaderboard List in JSON format from the server
     * @return Leaderboard String
     */
    public static String leaderboardToString(String jsonList) {
        JSONArray jsonArray = new JSONArray(jsonList);
        String result = "";
        for (int j = 0; j < jsonArray.length(); j++) {
            JSONObject jsonObject = (JSONObject) jsonArray.get(j);
            result  = result + jsonObject.get("username").toString()
                    + " - " + jsonObject.get("reducedCo2")
                    + " CO2 Reduction ,"
                    + jsonObject.get("userId") + ",";
        }
        return result;
    }
}
