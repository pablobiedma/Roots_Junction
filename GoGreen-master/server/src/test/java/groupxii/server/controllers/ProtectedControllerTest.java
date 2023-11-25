package groupxii.server.controllers;

import org.junit.Before;
import org.junit.Test;

import java.security.Principal;

import static org.junit.Assert.*;

public class ProtectedControllerTest {

    ProtectedController protectedController;
    @Before
    public void createSPC() {
        protectedController = new ProtectedController();
    }

    @Test
    public void protectedEndpoint() {
        Principal principal = new Principal() {
            @Override
            public String getName() {
                return "Ivan";
            }
        };
        String username = principal.getName();
        assertEquals("Hello, " + username + "!",protectedController.protectedEndpoint(principal));
    }
}