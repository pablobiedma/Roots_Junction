package groupxii.database;

import com.mongodb.DBObject;
import org.junit.Test;
import static org.junit.Assert.*;

public class EntryTest {
	private class SimpleEntry extends Entry {
		public SimpleEntry() {
			super();
		}

		public DBObject toDbObject() {
			return super.toBasicDbObject();
		}
	}


}
