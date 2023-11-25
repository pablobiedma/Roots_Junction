package groupxii.database;

import com.mongodb.DBObject;
import org.bson.BSONObject;

import java.util.ArrayList;
import java.util.List;

public class UserEntry extends Entry {

    private int userId;
    private String username;
    private String password;
    private int points;
    private int badge;
    private int reducedCo2;
    private List<Integer> friendsId ;
    private List<MealEntry> eatenMeals ;
    private List<VehicleEntry> usedVehicles;
    private List<PanelEntry> usedPanels;

    /** 
     * Constructor for the UserEntry class.
     */
    public UserEntry(int userId, 
                     String username,
                     String password) {
        this.userId = userId;
        this.username = username;
        this.password = password;
        this.points = 0;
        this.badge = 0;
        this.reducedCo2 = 0;
        this.friendsId = new ArrayList<>();
        this.eatenMeals = new ArrayList<>();
        this.usedVehicles = new ArrayList<>();
        this.usedPanels = new ArrayList<>();
    }

    /**
     * Construct UserEntry from a BSONObject, 
     * mostly used to translate find queries form Database to easier to work with format.
     */
    public UserEntry(BSONObject obj) {
        if (obj == null) {
            return;
        }
        this.userId = (int)obj.get("userId");
        this.username = (String)obj.get("username");
        this.password = (String)obj.get("password");
        this.points = (int)obj.get("points");
        this.badge = (int)obj.get("badge");
        this.reducedCo2 = (int)obj.get("reducedCo2");
        this.friendsId = (ArrayList<Integer>)obj.get("friendsId");
        this.eatenMeals = (ArrayList<MealEntry>)obj.get("eatenMeals");
        this.usedVehicles = (ArrayList<VehicleEntry>)obj.get("usedVehicles");
        this.usedPanels = (ArrayList<PanelEntry>)obj.get("usedPanels");
    }

    public int getUserId() {
        return userId;
    }

    public String getUsername() {
        return username;
    }

    public String getPassword() {
        return password;
    }

    public int getPoints() {
        return points;
    }

    public int getBadge() {
        return badge;
    }

    public int getReducedCo2() {
        return reducedCo2;
    }

    public List<Integer> getFriendsId() {
        return this.friendsId;
    }
    
    public List<MealEntry> getEatenMeals() {
        return this.eatenMeals;
    }

    public List<VehicleEntry> getUsedVehicles() {
        return this.usedVehicles;
    }

    public List<PanelEntry> getUsedPanels() {
        return this.usedPanels;
    }

    /**
     * Translates into a MongoDB JSON object.
     */
    public final DBObject toDbObject() {
        return super.toBasicDbObject()
                .append("userId", this.userId)
                .append("username", this.username)
                .append("password",this.password)
                .append("points", this.points)
                .append("badge", this.badge)
                .append("reducedCo2", this.reducedCo2)
                .append("friendsId",this.friendsId)
                .append("eatenMeals",this.eatenMeals)
                .append("usedVehicles", this.usedVehicles)
                .append("usedPanels", this.usedPanels);
    }
}

