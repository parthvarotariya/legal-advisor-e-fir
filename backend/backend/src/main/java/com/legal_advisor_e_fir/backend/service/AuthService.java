package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.*;
import com.legal_advisor_e_fir.backend.exceptions.InvalidCredentialsException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.User;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import com.legal_advisor_e_fir.backend.repository.UserRepo;
import io.jsonwebtoken.Jwt;
import jakarta.transaction.Transactional;
import org.hibernate.annotations.NaturalId;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

@Service
public class AuthService implements IAuthService{
    private final UserRepo userRepo;
    private final PoliceRepo policeRepo;
    private final PoliceStationRepo policeStationRepo;
    private final PasswordEncoder passwordEncoder;
    private final JwtUtil jwtUtil;

    public AuthService(@Autowired UserRepo userRepo, @Autowired PoliceRepo policeRepo, @Autowired PasswordEncoder passwordEncoder,
                       @Autowired PoliceStationRepo policeStationRepo, @Autowired JwtUtil jwtUtil)
    {
        this.userRepo = userRepo;
        this.policeRepo = policeRepo;
        this.passwordEncoder = passwordEncoder;
        this.policeStationRepo = policeStationRepo;
        this.jwtUtil = jwtUtil;
    }
    @Override
    public UserLoginResponseDto userLogin(LoginRequestDto request) {

        User user = userRepo.findByEmail(request.getEmail())
                .orElseThrow(() -> new ResourceNotFoundException(
                        "User not found with email: " + request.getEmail()));

        if (!passwordEncoder.matches(request.getPassword(), user.getPassword())) {
            throw new InvalidCredentialsException("Invalid email or password");
        }

        String token = jwtUtil.generateToken(user.getEmail());
        UserResponseDto response = mapToResponse(user);
        UserLoginResponseDto loginResponse = new UserLoginResponseDto();
        loginResponse.setToken(token);
        loginResponse.setUser(response);

        return loginResponse;
    }

    @Override
    @Transactional
    public UserLoginResponseDto userRegister(UserRequestDto request) {

        if (userRepo.existsByEmail(request.getEmail())) {
            throw new InvalidCredentialsException("Email already registered");
        }

        User user = mapToEntity(request);
        user.setPassword(passwordEncoder.encode(request.getPassword()));
        User savedUser = userRepo.save(user);

        String token = jwtUtil.generateToken(user.getEmail());

        UserResponseDto response = mapToResponse(user);
        UserLoginResponseDto loginResponse = new UserLoginResponseDto();
        loginResponse.setToken(token);
        loginResponse.setUser(response);

        return loginResponse;
    }


    @Override
    public PoliceLoginResponseDto policeLogin(LoginRequestDto request) {

        Police police = policeRepo.findByEmail(request.getEmail())
                .orElseThrow(() ->
                        new ResourceNotFoundException("Police not found with this email")
                );

        if (!passwordEncoder.matches(request.getPassword(), police.getPassword())) {
            throw new InvalidCredentialsException("Invalid email or password");
        }

        String token = jwtUtil.generateToken(request.getEmail());
        PoliceResponseDto policeResponse = mapToPoliceResponse(police);

        PoliceLoginResponseDto loginResponse = new PoliceLoginResponseDto();
        loginResponse.setToken(token);
        loginResponse.setPolice(policeResponse);

        return loginResponse;
    }




    @Override
    @Transactional
    public PoliceLoginResponseDto policeRegister(PoliceRequestDto request) {

        if (policeRepo.existsByEmail(request.getEmail())) {
            throw new InvalidCredentialsException(
                    "Email already registered: " + request.getEmail()
            );
        }
        if (policeRepo.existsByBadgeNumber(request.getBadgeNumber())) {
            throw new InvalidCredentialsException(
                    "Badge number already exists: " + request.getBadgeNumber()
            );
        }
        PoliceStation policeStation = policeStationRepo.findById(request.getStationId())
                .orElseThrow(() ->
                        new ResourceNotFoundException(
                                "Police station not found with id: " + request.getStationId()
                        )
                );


        Police police = mapToPoliceEntity(request, policeStation);
        police.setPassword(passwordEncoder.encode(request.getPassword()));

        Police savedPolice = policeRepo.save(police);

        String token = jwtUtil.generateToken(request.getEmail());

        PoliceResponseDto policeResponse = mapToPoliceResponse(police);
        PoliceLoginResponseDto loginResponse = new PoliceLoginResponseDto();
        loginResponse.setToken(token);
        loginResponse.setPolice(policeResponse);

        return loginResponse;
    }


    private UserResponseDto mapToResponse(User user) {

        UserResponseDto response = new UserResponseDto();
        response.setId(user.getId());
        response.setName(user.getName());
        response.setEmail(user.getEmail());
        response.setMobileNumber(user.getMobileNumber());
        response.setAddress(user.getAddress());

       return response;
    }

    private User mapToEntity(UserRequestDto request) {

        User user = new User();
        user.setName(request.getName());
        user.setEmail(request.getEmail());
        user.setPassword(request.getPassword()); // hashed later
        user.setMobileNumber(request.getMobileNumber());
        user.setAddress(request.getAddress());

        return user;
    }

    private PoliceResponseDto mapToPoliceResponse(Police police) {
        PoliceResponseDto response = new PoliceResponseDto();
        response.setPoliceId(police.getPoliceId());
        response.setName(police.getName());
        response.setBadgeNumber(police.getBadgeNumber());
        response.setRank(police.getRank());
        response.setEmail(police.getEmail());
        response.setMobileNumber(police.getMobileNumber());
        response.setRole(police.getRole());
        response.setStationId(police.getPoliceStation().getStationId());
        response.setStationName(police.getPoliceStation().getStationName());
        response.setStationCode(police.getPoliceStation().getStationCode());
        response.setCreatedAt(police.getCreatedAt());
        response.setUpdatedAt(police.getUpdatedAt());
        return response;
    }

    @Override
    public void resetPolicePassword(String email, String newPassword) {
        Police police = policeRepo.findByEmail(email)
                .orElseThrow(() ->
                        new ResourceNotFoundException("Police not found with this email")
                );
        
        police.setPassword(passwordEncoder.encode(newPassword));
        policeRepo.save(police);
    }

    private Police mapToPoliceEntity(PoliceRequestDto request, PoliceStation policeStation) {
        Police police = new Police();
        police.setName(request.getName());
        police.setBadgeNumber(request.getBadgeNumber());
        police.setRank(request.getRank());
        police.setEmail(request.getEmail());
        police.setMobileNumber(request.getMobileNumber());
        police.setPassword(request.getPassword()); // Should be hashed in production
        police.setRole(request.getRole());
        police.setPoliceStation(policeStation);
        return police;
    }
}
