package groupxii.server.security;

import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.InsufficientAuthenticationException;
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

public class CredentialsDbAuthenticationProviderTest {
	@Rule
	public ExpectedException thrown = ExpectedException.none();

	CredentialsDbAuthenticationProvider cdap;
	@Before
	public void createap() {
		cdap = new CredentialsDbAuthenticationProvider();
	}

	@Test
	public void missingUsernameTest() {
		thrown.expect(InsufficientAuthenticationException.class);
		thrown.expectMessage("Username or password are null");
		Authentication missingUsername = new UsernamePasswordAuthenticationToken(null, "pass");

		cdap.authenticate(missingUsername);
	}

	@Test
	public void missingPasswordTest() {
		thrown.expect(InsufficientAuthenticationException.class);
		thrown.expectMessage("Username or password are null");
		Authentication missingPassword = new UsernamePasswordAuthenticationToken("0day", null);

		cdap.authenticate(missingPassword);
	}

	@Test
	public void testSupportTrue() {
		assertTrue(cdap.supports(UsernamePasswordAuthenticationToken.class));
	}

	@Test
	public void testSupportFalse() {
		assertFalse(cdap.supports(AnonymousAuthenticationToken.class));
	}

	@Test
	public void testUsername0day() {
		Database mockedDB = Mockito.mock(Database.class);
		Mockito.when(mockedDB.findUserByName("0day")).thenReturn(new UserEntry(0, "0day", "pass"));

		Whitebox.setInternalState(Database.class, "instance", mockedDB);

		Authentication zeroDay = new UsernamePasswordAuthenticationToken("0day", "pass");
		cdap.authenticate(zeroDay);
		// No exception thrown -> test pass
	}

	@Test
	public void testUsernameBadPass() {
		thrown.expect(BadCredentialsException.class);
		thrown.expectMessage("Authentication failed");

		Database mockedDB = Mockito.mock(Database.class);
		Mockito.when(mockedDB.findUserByName("0day")).thenReturn(new UserEntry(0, "0day", "pass"));

		Whitebox.setInternalState(Database.class, "instance", mockedDB);

		Authentication firstDay = new UsernamePasswordAuthenticationToken("0day", "hacker");
		cdap.authenticate(firstDay);
	}



}
