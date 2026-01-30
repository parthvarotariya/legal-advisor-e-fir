package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.User;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.UserRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Optional;

@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Autowired
    private UserRepo userRepository;

    @Autowired
    private PoliceRepo policeRepository;

    @Override
    public UserDetails loadUserByUsername(String email)
            throws UsernameNotFoundException {

        // Check User table
        Optional<User> user = userRepository.findByEmail(email);
        if (user.isPresent()) {
            return new org.springframework.security.core.userdetails.User(
                    user.get().getEmail(),
                    user.get().getPassword(),
                    new ArrayList<>()
            );
        }

        // Check Police table
        Optional<Police> police = policeRepository.findByEmail(email);
        if (police.isPresent()) {
            return new org.springframework.security.core.userdetails.User(
                    police.get().getEmail(),
                    police.get().getPassword(),
                    new ArrayList<>()
            );
        }

        throw new UsernameNotFoundException("User not found");
    }
}
