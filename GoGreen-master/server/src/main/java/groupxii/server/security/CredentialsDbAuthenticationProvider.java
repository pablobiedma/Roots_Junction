package groupxii.server.security;

import groupxii.database.Database;
import groupxii.database.UserEntry;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.InsufficientAuthenticationException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;

/**
 * Authentication provider that verifies the validity of the provided credentials
 * from the database. (WIP)
 */
public class CredentialsDbAuthenticationProvider implements AuthenticationProvider {
    @Override
    public Authentication authenticate(Authentication authentication)
                                           throws AuthenticationException { 
        Object principal = authentication.getPrincipal();
        Object credentials = authentication.getCredentials();
        if (principal == null || credentials == null) {
            throw new InsufficientAuthenticationException("Username or password are null");
        }

        String username = principal.toString();
        String password = credentials.toString();

        UserEntry found = Database.instance.findUserByName(username);
        //TODO figure out how to pull this V
        //boolean match = WebSecurityConfiguration.passwordEncoder()
        //.matches(password, found.getPassword());
        boolean match = password.equals(found.getPassword());
        if (match) {
            return new UsernamePasswordAuthenticationToken(username, password);
        }

        throw new BadCredentialsException("Authentication failed");
    }

    @Override
    public boolean supports(Class<?> authentication) {
        return authentication.equals(
               UsernamePasswordAuthenticationToken.class);
    }
}
