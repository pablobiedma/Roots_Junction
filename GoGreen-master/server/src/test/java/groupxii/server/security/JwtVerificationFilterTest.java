package groupxii.server.security;

import groupxii.server.security.SecurityKey;

import io.jsonwebtoken.Jwts;
import io.jsonwebtoken.security.Keys;

import org.springframework.security.authentication.*;
import org.springframework.security.core.*;

import org.junit.Before;
import org.junit.Test;
import org.junit.Rule;
import org.junit.rules.ExpectedException;

import static org.mockito.Mockito.*;
import static org.junit.Assert.*;

import java.util.*;
import java.io.*;
import javax.servlet.http.*;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;

public class JwtVerificationFilterTest {
//	@Rule
//	ExpectedException thrown = ExpectedException.none();

	void passTest() {
		assertTrue(true);
	}
	void failTest() {
		assertTrue(false);
	}

	AuthenticationManager authman;
	private HttpServletRequest request;
	private HttpServletResponse response;
	private FilterChain chain;

	private class BreakCapsul extends JwtVerificationFilter {
		public BreakCapsul() {
			super(authman);
		}

		public void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain) throws IOException, ServletException {
			super.doFilterInternal(request, response, chain);
		}
	}

	BreakCapsul jwtver;

	@Before
	public void createjwtver() throws IOException {
		SecurityKey.instance.readKey();
		List<AuthenticationProvider> all = new ArrayList<>();
		all.add(new UsernameAuthenticationProvider());
		authman = new ProviderManager(all);

		jwtver = new BreakCapsul();

		request = mock(HttpServletRequest.class);
		when(request.getProtocol()).thenReturn("HTTP/1.1");

		String validJWS = Jwts.builder()
			.setSubject("0day")
			.signWith(Keys.hmacShaKeyFor(SecurityKey.instance.getKey()))
			.setHeaderParam("typ", "JWT")
			.setIssuedAt(new Date(System.currentTimeMillis()))
			.setExpiration(new Date(System.currentTimeMillis() + 86400000))
			.setIssuer("goGreen-server")
			.compact();

		when(request.getHeader("Authorization")).thenReturn("Bearer "+validJWS);
		response = mock(HttpServletResponse.class);

		chain = mock(FilterChain.class);
	}

	@Test
	public void tokenNullTest() throws IOException, ServletException {
		when(request.getHeader("Authorization")).thenReturn(null);

		jwtver.doFilterInternal(request, response, chain);
		verify(chain, times(1)).doFilter(request, response);
	}

	@Test
	public void tokenNotBearer() throws IOException, ServletException {
		when(request.getHeader("Authorization")).thenReturn("NotABearer");

		jwtver.doFilterInternal(request, response, chain);
		verify(chain, times(1)).doFilter(request, response);
	}

	@Test
	public void invalidTokenTest() throws IOException, ServletException {
		when(request.getHeader("Authorization")).thenReturn("badtoken");

		jwtver.doFilterInternal(request, response, chain);
		verify(chain, times(1)).doFilter(request, response);
	}

	@Test
	public void validTokenTest() throws IOException, ServletException {
		jwtver.doFilterInternal(request, response, chain);
	}



}
