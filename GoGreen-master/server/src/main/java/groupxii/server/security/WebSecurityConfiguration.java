package groupxii.server.security;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.AuthenticationProvider;
import org.springframework.security.authentication.ProviderManager;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.annotation.web.configuration.WebSecurityConfigurerAdapter;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.authentication.AnonymousAuthenticationFilter;

import java.util.ArrayList;
import java.util.List;

@Configuration
@EnableWebSecurity
public class WebSecurityConfiguration extends WebSecurityConfigurerAdapter {
    public WebSecurityConfiguration() {
        super(false);
    }

    @Override
    protected AuthenticationManager authenticationManager() throws Exception {
        List<AuthenticationProvider> allCustomProviders = new ArrayList<>();
        allCustomProviders.add(new UsernameAuthenticationProvider());
        allCustomProviders.add(new CredentialsDbAuthenticationProvider());
        return new ProviderManager(allCustomProviders);
    }
 
    protected AuthenticationManager credentialsAuthentiactionManager() throws Exception {
        List<AuthenticationProvider> allCustomProviders = new ArrayList<>();
        allCustomProviders.add(new CredentialsDbAuthenticationProvider());
        return new ProviderManager(allCustomProviders);
    }
 
    protected AuthenticationManager usernameAuthentiactionManager() throws Exception {
        List<AuthenticationProvider> allCustomProviders = new ArrayList<>();
        allCustomProviders.add(new UsernameAuthenticationProvider());
        return new ProviderManager(allCustomProviders);
    }


    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
                .addFilterBefore(new JwtVerificationFilter(usernameAuthentiactionManager()),
                        AnonymousAuthenticationFilter.class)
                .csrf().disable() // Unecessary in REST
                .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                .and()
                .httpBasic().disable()
                .addFilter(new JwtGeneratingFilter(credentialsAuthentiactionManager()))
                .authorizeRequests()
            .antMatchers("/login").permitAll()
            .and()
            .authorizeRequests()
            .antMatchers("/register").permitAll()
            .and()
            .authorizeRequests()
            .anyRequest().authenticated();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
