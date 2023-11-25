package groupxii.server.security;

import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.authentication.AnonymousAuthenticationToken;
import org.springframework.security.core.Authentication;

import org.junit.Before;
import org.junit.Test;
import org.junit.Rule;
import org.junit.runner.RunWith;
import org.junit.rules.ExpectedException;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;

import org.mockito.Mockito;

import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;
import org.powermock.reflect.Whitebox;

import groupxii.database.Database;
import groupxii.database.UserEntry;

@RunWith(PowerMockRunner.class)
@PrepareForTest({Database.class})

public class UsernameAuthenticationProviderTest {
	@Rule
	public ExpectedException thrown = ExpectedException.none();

	UsernameAuthenticationProvider uap;
	@Before
	public void createuap() {
		uap = new UsernameAuthenticationProvider();
	}

	@Test
	public void missingUsernameTest() {
		thrown.expect(BadCredentialsException.class);
		thrown.expectMessage("Missing username");
		Authentication missingUsername = new UsernamePasswordAuthenticationToken(null, null);

		uap.authenticate(missingUsername);
	}

	@Test
	public void testSupportTrue() {
		assertTrue(uap.supports(UsernamePasswordAuthenticationToken.class));
	}

	@Test
	public void testSupportFalse() {
		assertFalse(uap.supports(AnonymousAuthenticationToken.class));
	}

	@Test
	public void testUsername0day() {
		Database mockedDB = Mockito.mock(Database.class);
		Mockito.when(mockedDB.findUserByName("0day")).thenReturn(new UserEntry(0, "0day", "pass"));

		Whitebox.setInternalState(Database.class, "instance", mockedDB);

		Authentication zeroDay = new UsernamePasswordAuthenticationToken("0day", null);
		uap.authenticate(zeroDay);
		// No exception thrown -> test pass
	}

	/*
	@Test
	public void testUsernameWrongToken() {
//		thrown.expect(BadCredentialsException.class);
//		thrown.expectMessage("Username verification failed");

		Database mockedDB = Mockito.mock(Database.class);
		Mockito.when(mockedDB.findUserByName("0day")).thenReturn(new UserEntry(0, "0day", "pass"));

		Whitebox.setInternalState(Database.class, "instance", mockedDB);
		Authentication firstDay = new UsernamePasswordAuthenticationToken("1day", "pass");
//		uap.authenticate(firstDay);
//		TODO fix this
	}
	*/



}
