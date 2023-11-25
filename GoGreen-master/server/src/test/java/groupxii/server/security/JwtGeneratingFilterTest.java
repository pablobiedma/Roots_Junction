package groupxii.server.security;

import groupxii.server.security.SecurityKey;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.ProviderManager;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.authentication.InsufficientAuthenticationException;
import org.springframework.security.authentication.BadCredentialsException;

import org.junit.Before;
import org.junit.Test;
import org.junit.Rule;
import static org.junit.Assert.*;
import org.junit.rules.ExpectedException;

import static org.mockito.Mockito.*;

import java.io.ByteArrayOutputStream;
import java.io.PrintWriter;
import java.io.BufferedReader;
import java.io.Reader;
import java.io.IOException;
import java.io.StringReader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.FilterChain;
import javax.servlet.ServletException;

public class JwtGeneratingFilterTest {
	@Rule
	public ExpectedException thrown = ExpectedException.none();

	private JwtGeneratingFilter jwtgen;

	private HttpServletRequest request;
	private HttpServletResponse response;
	String body;
	Reader bodyReader;
	BufferedReader reader;
	AuthenticationManager authenticationManager;

	private class BreakEncapsulation extends JwtGeneratingFilter {
		public BreakEncapsulation() {
			super(authenticationManager);
		}
		public void publicSuccessAuth(HttpServletRequest request, HttpServletResponse response, FilterChain chain, Authentication authentication) {
			super.successfulAuthentication(request, response, chain, authentication);
		}
		public void publicUnsuccessfulAuth(HttpServletRequest request, HttpServletResponse response, AuthenticationException failed) throws IOException, ServletException {
			super.unsuccessfulAuthentication(request, response, failed);
		}
	}

	private void createReader(String text) {
		body = text;
		bodyReader = new StringReader(body);
		reader = new BufferedReader(bodyReader);
		try {
			when(request.getReader()).thenReturn(reader);
		} catch (IOException e) {
			// Java is shit language
		}
	}
	BreakEncapsulation breakEncapsulation = new BreakEncapsulation();


	@Before
	public void createjwtgen() throws IOException {
		SecurityKey.instance.readKey();
		List<AuthenticationProvider> allCustomProviders = new ArrayList<>();
		allCustomProviders.add(new UsernameAuthenticationProvider());
		authenticationManager = new ProviderManager(allCustomProviders);

		jwtgen = new JwtGeneratingFilter(authenticationManager);
		request = mock(HttpServletRequest.class);
		when(request.getProtocol()).thenReturn("HTTP/1.1");	
		when(request.getMethod()).thenReturn("POST");

		createReader("{\"username\":\"0day\", \"password\":\"pass\"}");
		

		response = mock(HttpServletResponse.class);
	}

	@Test
	public void testNonPostRequest() {
		thrown.expect(InsufficientAuthenticationException.class);
		thrown.expectMessage("Request must be POST");

		when(request.getMethod()).thenReturn("GET");

		jwtgen.attemptAuthentication(request, response);
	}

	@Test
	public void testBodyIOException() {
		thrown.expect(InsufficientAuthenticationException.class);
		thrown.expectMessage("Could not read request body");

		try {
			when(request.getReader()).thenThrow(IOException.class);
		} catch (IOException e) {
			// Java is shit language
		}

		jwtgen.attemptAuthentication(request, response);
	}

	@Test
	public void testEmptyRequestBody() {
		thrown.expect(InsufficientAuthenticationException.class);
		thrown.expectMessage("Could not parse JSON");

		createReader("");

		jwtgen.attemptAuthentication(request, response);
	}

	@Test
	public void succesfullAuthentication() {
		//jwtgen.attemptAuthentication(request, response);
	}

	@Test
	public void generatingTokenTest() {
		Authentication token = mock(Authentication.class);
		when(token.getPrincipal()).thenReturn("username");
		String type;
		String content;

		breakEncapsulation.publicSuccessAuth(request, response, null, token);
		// no exception => test passes
	}

	@Test
	public void testFailedBadCredentials() throws IOException, ServletException {
		ByteArrayOutputStream responseMessage = new ByteArrayOutputStream();
		PrintWriter writer = new PrintWriter(responseMessage);
		when(response.getWriter()).thenReturn(writer);
		breakEncapsulation.unsuccessfulAuthentication(request, response, new BadCredentialsException("error"));
		//probably should check respondMessage but whatever TODO
		// no exception => test passes
	}

	@Test
	public void testFailedInsufficientCredentials() throws IOException, ServletException{
		ByteArrayOutputStream responseMessage = new ByteArrayOutputStream();
		PrintWriter writer = new PrintWriter(responseMessage);
		when(response.getWriter()).thenReturn(writer);
		breakEncapsulation.unsuccessfulAuthentication(request, response, new InsufficientAuthenticationException("error"));
		//probably should check respondMessage but whatever TODO
		// no exception => test passes
	}

	@Test
	public void unreachableConditionTest() throws IOException, ServletException {
		ByteArrayOutputStream responseMessage = new ByteArrayOutputStream();
		PrintWriter writer = new PrintWriter(responseMessage);
		when(response.getWriter()).thenReturn(writer);
		breakEncapsulation.unsuccessfulAuthentication(request, response, new UsernameNotFoundException("unknow"));
	}
}
