package groupxii.server.security;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;
import org.json.JSONException;
import org.json.JSONObject;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.InsufficientAuthenticationException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;

import java.io.IOException;
import java.util.Date;
import java.util.stream.Collectors;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Security filter that's responsible for generating JSON Web Tokens.
 */
public class JwtGeneratingFilter extends UsernamePasswordAuthenticationFilter {

    private final AuthenticationManager authenticationManager;
    private final String path;

    /**
     * Creates a filter with the provided AuthenticationManager on the /login location.
     */
    public JwtGeneratingFilter(AuthenticationManager authenticationManager) {
        this.authenticationManager = authenticationManager;

        path = "/login";
        setFilterProcessesUrl(path);
    }

    /**
     * Attempts to read a POST request for credentials in JSON format.
     */
    @Override
    public Authentication attemptAuthentication(HttpServletRequest request,
                                                HttpServletResponse response) 
                                                    throws AuthenticationException {
        String body = null;
    
        if (!("POST".equalsIgnoreCase(request.getMethod()))) {
            throw new InsufficientAuthenticationException("Request must be POST");
        }
    
        try {
            body = request.getReader().lines().collect(Collectors.joining(System.lineSeparator()));
        } catch (IOException e) {
            throw new InsufficientAuthenticationException("Could not read request body");
        }
    
        JSONObject json = null;
    
        String username = null;
        String password = null;
    
        try {
            json = new JSONObject(body);
            username = json.getString("username");
            password = json.getString("password");
        } catch (JSONException e) {
            throw new InsufficientAuthenticationException("Could not parse JSON");
        }
    
        UsernamePasswordAuthenticationToken authenticationToken = 
                                                new UsernamePasswordAuthenticationToken(
                                                    username, 
                                                    password);

        return authenticationManager.authenticate(authenticationToken);
    }

    /**
     * Generates and sends back a valid and signed JSON Web Token. 
     * Note that SecurityKey should have read a valid and secure key.
     */
    @Override
    protected void successfulAuthentication(HttpServletRequest request, 
                                            HttpServletResponse response,
                                            FilterChain filterChain, 
                                            Authentication authentication) {
        String username = authentication.getPrincipal().toString();
        //TODO figure out a good value for this
        int expirationTime = 86400000;
        String jws = Jwts.builder()
            .setSubject(username)
            .signWith(Keys.hmacShaKeyFor(SecurityKey.instance.getKey()))
            .setHeaderParam("typ", "JWT")
            .setIssuer("goGreen-server")
            .setIssuedAt(new Date(System.currentTimeMillis()))
            .setExpiration(new Date(System.currentTimeMillis() + expirationTime))
            .compact();

        response.addHeader("Authorization", "Bearer " + jws);
    }

    /**
     * Sends an appropriate response when the authentication fails.
     */
    @Override
    protected void unsuccessfulAuthentication(HttpServletRequest request, 
                                              HttpServletResponse response,
                                              AuthenticationException failed) 
                                                  throws IOException, ServletException {
        int sc = -1;
        if (failed instanceof InsufficientAuthenticationException) {
            sc = HttpServletResponse.SC_BAD_REQUEST;
        } else if (failed instanceof BadCredentialsException) {
            sc = HttpServletResponse.SC_UNAUTHORIZED;
        }
        response.setStatus(sc);
        response.setContentType("application/json");
        response.getWriter().append(
            JsonErrorResponseGenerator.createMessage(sc, failed.getMessage(), path));

    }
}
