package groupxii.server.security;

import io.jsonwebtoken.Claims;
import io.jsonwebtoken.Jws;
import io.jsonwebtoken.JwtException;
import io.jsonwebtoken.Jwts;
import org.springframework.http.HttpHeaders;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.web.authentication.www.BasicAuthenticationFilter;

import java.io.IOException;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * Security filter that ensures that the provided JSON Web Token is valid.
 */
public class JwtVerificationFilter extends BasicAuthenticationFilter {

    public JwtVerificationFilter(AuthenticationManager authenticationManager) {
        super(authenticationManager);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest servletRequest, 
                                    HttpServletResponse servletResponse,
                                    FilterChain filterChain) 
                                        throws IOException, ServletException {
        HttpServletRequest request = (HttpServletRequest)servletRequest;
        HttpServletResponse response = (HttpServletResponse)servletResponse;

        String token = request.getHeader(HttpHeaders.AUTHORIZATION);

        if (token == null || !token.startsWith("Bearer ")) {
            filterChain.doFilter(request, response);
            return;
        }

        
        UsernamePasswordAuthenticationToken authentication = getAuthenticaion(token);
        if (authentication == null) {
            filterChain.doFilter(request, response);
            return;
        }

        SecurityContextHolder.getContext().setAuthentication(authentication);
        filterChain.doFilter(request, response);
    }

    private UsernamePasswordAuthenticationToken getAuthenticaion(String jwsString) {
        Jws<Claims> jws;

        try {
            jws = Jwts.parser()
                .setSigningKey(SecurityKey.instance.getKey())
                .parseClaimsJws(jwsString.replace("Bearer ", ""));
            String username = new String(jws.getBody().getSubject());
            return new UsernamePasswordAuthenticationToken(username, null);
        } catch (JwtException e) {
            return null;
        }
    }
}
