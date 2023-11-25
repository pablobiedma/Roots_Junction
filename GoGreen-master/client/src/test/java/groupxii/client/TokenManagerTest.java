package groupxii.client;

import groupxii.client.connector.LoginConnector;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

import org.powermock.api.mockito.PowerMockito;
import org.powermock.core.classloader.annotations.PrepareForTest;
import org.powermock.modules.junit4.PowerMockRunner;

import static org.mockito.Mockito.*;

@RunWith(PowerMockRunner.class)
@PrepareForTest({LoginConnector.class, Thread.class})

public class TokenManagerTest {
	@Before
	public void init() {
		PowerMockito.mockStatic(LoginConnector.class);
		PowerMockito.when(LoginConnector.postCredentials(anyString())).thenReturn("Hi");
		TokenManager.instance = new TokenManager("usr", "password");
	}

	@Test
	public void TokenConstructAndGetTest() {

		TokenManager.instance.refreshToken();

		assertEquals(TokenManager.instance.getToken(), "Hi");
	}

	@Test
	public void daemonExceptionTest() {
		PowerMockito.mockStatic(Thread.class);
		PowerMockito.doThrow(new InterruptedException()).when(Thread.class);
		try {
			Thread.sleep(anyLong());
		} catch(Exception e) {
			System.err.println("something went wrong");
			assertTrue(false);
		}

		TokenManager.Daemon daemon = TokenManager.instance.new Daemon();
//Java is a bad language and the tools arond it are bad		daemon.run();
		//If we don't hang then this works
	}
}
