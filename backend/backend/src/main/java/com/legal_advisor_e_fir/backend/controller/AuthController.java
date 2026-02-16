package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.LoginRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceLoginResponseDto;
import com.legal_advisor_e_fir.backend.dto.UserLoginResponseDto;
import com.legal_advisor_e_fir.backend.service.IAuthService;
import com.legal_advisor_e_fir.backend.service.JwtUtil;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/auth")
public class AuthController {

    private final IAuthService authService;
    private final JwtUtil jwtUtil;

    public AuthController(IAuthService authService, JwtUtil jwtUtil) {
        this.authService = authService;
        this.jwtUtil = jwtUtil;
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

    @PostMapping("/police/reset-password")
    public String resetPolicePassword(
            @RequestParam String email,
            @RequestParam String newPassword) {
        authService.resetPolicePassword(email, newPassword);
        return "Password reset successful";
    }

    @PostMapping("/admin/login")
    public Map<String, Object> adminLogin(@RequestBody Map<String, String> request) {
        
        String username = request.get("username");
        String password = request.get("password");
        
        System.out.println("Admin login attempt - Username: " + username + ", Password: " + password);
        
        // Hardcoded admin credentials
        if ("admin".equals(username) && "admin123".equals(password)) {
            String token = jwtUtil.generateToken("admin@system");
            
            Map<String, Object> response = new HashMap<>();
            response.put("token", token);
            
            Map<String, String> admin = new HashMap<>();
            admin.put("username", "admin");
            admin.put("role", "ADMIN");
            response.put("admin", admin);
            
            return response;
        }
        
        throw new RuntimeException("Invalid admin credentials");
    }
}
