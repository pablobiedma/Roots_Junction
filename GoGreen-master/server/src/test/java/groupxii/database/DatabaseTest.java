package groupxii.database;

import com.mongodb.DB;
import com.mongodb.MongoClient;

import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.Rule;
import static org.junit.Assert.*;
import org.junit.rules.ExpectedException;

import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import org.mockito.Mockito.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@RunWith(PowerMockRunner.class)
@PrepareForTest({Database.class})

public class DatabaseTest {
	@Rule
	ExpectedException thrown = ExpectedException.none();

	static List<Integer> friendsId;
	static List<MealEntry> eatenMeals;
	static UserEntry usr;
	static String addr;
	static String port;

	@BeforeClass
	public static void setUpDb() throws IOException {
		addr = "localhost";
		Database.instance.setDbName("test");
		Database.instance.startDb();

		friendsId = new ArrayList<>();
		eatenMeals = new ArrayList<>();

		usr = new UserEntry(1,"Ivan","pass");
		Database.instance.save(usr);
	}


	@Test
	public void testEnvVar() {
		port = "27017";
		PowerMockito.mockStatic(System.class);
		PowerMockito.when(System.getenv("DB_ADDRESS")).thenReturn(addr);
		PowerMockito.when(System.getenv("DB_PORT")).thenReturn(port);
		Database test = new Database();

		assertEquals(addr, test.getDbAddr());
		assertEquals(Integer.parseInt(port), test.getDbPort());
	}

	@Test
	public void testNoEnvPort() {
		PowerMockito.mockStatic(System.class);
		PowerMockito.when(System.getenv("DB_ADDRESS")).thenReturn(addr);
		PowerMockito.when(System.getenv("DB_PORT")).thenReturn(null);

		Database test = new Database();

		assertEquals(addr, test.getDbAddr());
		assertEquals(27017, test.getDbPort());
	}

	@Test
	public void testNPEPort() {
		PowerMockito.mockStatic(System.class);
		PowerMockito.when(System.getenv("DB_ADDRESS")).thenReturn(addr);
		try {
		PowerMockito.doThrow(new NullPointerException()).when(System.class, "getenv", "DB_PORT");

		} catch (Exception e) {
			//dunno
			assertTrue(false);
		}
		Database test = new Database();

		assertEquals(addr, test.getDbAddr());
		assertEquals(27017, test.getDbPort());
	}

	@Test
	public void testNoEnvAddr() {
		port = "27017";
		PowerMockito.mockStatic(System.class);
		PowerMockito.when(System.getenv("DB_ADDRESS")).thenReturn(null);
		PowerMockito.when(System.getenv("DB_PORT")).thenReturn(port);

		Database test = new Database();

		assertEquals("localhost", test.getDbAddr());
		assertEquals(Integer.parseInt(port), test.getDbPort());
	}

	@Test
	public void testBadEnvPort() {
		port = "PORT_IS_27017";
		PowerMockito.mockStatic(System.class);
		PowerMockito.when(System.getenv("DB_ADDRESS")).thenReturn(addr);
		PowerMockito.when(System.getenv("DB_PORT")).thenReturn(port);

		Database test = new Database();

		assertEquals(addr, test.getDbAddr());
		assertEquals(27017, test.getDbPort());
	}


	/*
	 * deprecated?
	@Test
	public void testSave() {
		VehicleEntry entry = new VehicleEntry(1337, "car");
		Database.instance.save(entry);
		assertEquals(entry.toDbObject(), Database.instance.findVehicleEntry(entry));
	}
	*/

	@Test
	public void testSaveNull() {
		Database.instance.save(null);
		//nothing happens => test passes
	}

	/*
	 * deprecated?
	@Test
	public void testSaveNonBlocking() throws InterruptedException {
		VehicleEntry entry = new VehicleEntry(1345, "car");
		Database.instance.saveNonBlocking(entry);

		assertEquals(entry.toDbObject(), Database.instance.findVehicleEntry(entry));
	}
	*/

	// User operation realted test block:
	// Only positive tests :/
	@Test
	public void testUserById() {
		UserEntry foundUser = Database.instance.findUserById(1);
		assertEquals(foundUser.getUserId(), usr.getUserId());
		assertEquals(foundUser.getUsername(), usr.getUsername());
	}

	@Test
	public void testFindByUserName() {
		UserEntry foundUser = Database.instance.findUserByName("Ivan");
		assertEquals(foundUser.getUserId(), usr.getUserId());
		assertEquals(foundUser.getUsername(), usr.getUsername());
	}


	@Test
	public void testSortUsersByPoints() {
		UserEntry usr2 = new UserEntry(2, "CO2Reducer", "pass");
		Database.instance.save(usr2);
		Database.instance.incrementReducedCo2("CO2Reducer", 2);
		List<UserEntry> sorted = Database.instance.sortUsersByReducedCo2();
		assertEquals(sorted.get(0).getUsername(), "CO2Reducer");
		assertEquals(sorted.get(1).getUsername(), "Ivan");
	}

	@Test
	public void testAddFriend() {
		Database.instance.addFriend("Ivan",2);
		friendsId.add(2);
		UserEntry updatedUser = Database.instance.findUserById(1);
		assertEquals(updatedUser.getFriendsId(), friendsId);
	}

	@Test
	public void testIncrementReducedCO2() {
		Database.instance.incrementReducedCo2("Ivan", 2);
		UserEntry updatedUser = Database.instance.findUserById(1);
		assertEquals(2, updatedUser.getReducedCo2());
	}

	@Test
	public void testGetUserCount() {
		//Users should be CO2Reducer and Ivan
		assertEquals(2, Database.instance.getUserCount());
	}

	@Test
	public void testAddMealEntry() {
		MealEntry me = new MealEntry("GRAPEFRUIT", 100,
				             "GRAPEFRUIT", 100);
		Database.instance.addEatenMeal("Ivan", me);
		UserEntry updatedUser = Database.instance.findUserById(1);
		assertNotNull(updatedUser.getEatenMeals().get(0)); //:(
	}


	//End of User related operation block

	@AfterClass
	public static void dropTestDb() {
		MongoClient mc;
		DB db;
		mc = new MongoClient(Database.instance.getDbAddr(),
				     Database.instance.getDbPort());
		db = mc.getDB("test");
		db.dropDatabase();
	}
}
