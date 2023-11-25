package groupxii.database;

import com.mongodb.BasicDBObject;
import com.mongodb.DBObject;
import org.bson.types.ObjectId;

/**
 * Abstract class to be used in creating for creation of MongoDB entries.
 */
public abstract class Entry {
    public ObjectId id; // MongoDB's _id

    Entry() {
        this.id = new ObjectId();
    }

    public abstract DBObject toDbObject();

    public BasicDBObject toBasicDbObject() {
        return new BasicDBObject("_id", this.id);
    }
}
