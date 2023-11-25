package groupxii.client.leaderboard;

import org.junit.Test;

import static groupxii.client.leaderboard.JsonConverter.leaderboardToString;
import static org.junit.Assert.*;

public class JsonConverterTest {

    @Test
    public void leaderboardToStringTest() {
        assertEquals(leaderboardToString("[{\"userId\":1,\"username\":\"test\",\"points\":2649,\"badge\":3,\"reducedCo2\":7151,\"friendsId\":[2,4,1,3]}]"), "test - 7151 CO2 Reduction ,1,");
    }
}