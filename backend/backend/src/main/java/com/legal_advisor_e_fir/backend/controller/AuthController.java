package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.LoginRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceLoginResponseDto;
import com.legal_advisor_e_fir.backend.dto.UserLoginResponseDto;
import com.legal_advisor_e_fir.backend.service.IAuthService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final IAuthService authService;

    public AuthController(IAuthService authService) {
        this.authService = authService;
    }


    @PostMapping("/user/login")
    public UserLoginResponseDto userLogin(
            @Valid @RequestBody LoginRequestDto request) {

        return authService.userLogin(request);
    }

    @PostMapping("/user/register")
    public UserLoginResponseDto userRegister(
            @Valid @RequestBody UserRequestDto request) {

        return authService.userRegister(request);
    }


    @PostMapping("/police/login")
    public PoliceLoginResponseDto policeLogin(
            @Valid @RequestBody LoginRequestDto request) {

        return authService.policeLogin(request);
    }

    @PostMapping("/police/register")
    public PoliceLoginResponseDto policeRegister(
            @Valid @RequestBody PoliceRequestDto request) {

        return authService.policeRegister(request);
    }
}
