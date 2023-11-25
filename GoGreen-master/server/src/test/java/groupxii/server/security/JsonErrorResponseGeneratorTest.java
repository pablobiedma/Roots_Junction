package groupxii.server.security;

import org.junit.Test;
import static org.junit.Assert.*;

public class JsonErrorResponseGeneratorTest {
	@Test
	public void testCreateMessage() {
		JsonErrorResponseGenerator messageCreater = new JsonErrorResponseGenerator();
		assertNotNull(messageCreater.createMessage(200, "OK", "/login"));
	}
}
