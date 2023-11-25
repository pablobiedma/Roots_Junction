package groupxii.server.security;

import org.junit.Test;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertNotNull;

import java.io.IOException;

public class SecurityKeyTest {
	/*
	@Test
	public void testReadKeyNull() {
		SecurityKey.instance.readKey(null);
		assertNull(SecurityKey.instance.getKey());
	}
	*/

	@Test
	public void restReadKeyDefault() throws IOException {
		//Bad test but could be usefull to ensure that a key exists
		//before startup
		SecurityKey.instance.readKey();
		assertNotNull(SecurityKey.instance.getKey());
	}
}
