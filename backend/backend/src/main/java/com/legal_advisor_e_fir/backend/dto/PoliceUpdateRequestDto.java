package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.Role;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceUpdateRequestDto {

    @Size(min = 2, max = 100, message = "Name must be between 2 and 100 characters")
    private String name;

    @Pattern(
            regexp = "^[A-Z0-9]{6,15}$",
            message = "Badge number must be 6-15 characters long and contain only uppercase letters and numbers"
    )
    private String badgeNumber;

    @Size(min = 2, max = 50, message = "Rank must be between 2 and 50 characters")
    private String rank;

    @Email(message = "Invalid email format")
    private String email;

    @Pattern(
            regexp = "^[6-9][0-9]{9}$",
            message = "Mobile number must be a valid 10-digit Indian number"
    )
    private String mobileNumber;

    @Size(min = 8, max = 100, message = "Password must be between 8 and 100 characters")
    @Pattern(
            regexp = "^(?=.*[0-9])(?=.*[a-z])(?=.*[A-Z])(?=.*[@#$%^&+=!]).{8,}$",
            message = "Password must contain at least one digit, one lowercase letter, one uppercase letter, and one special character (@#$%^&+=!)"
    )
    private String password;

    private Role role;

    @Positive(message = "Station ID must be a positive number")
    private Long stationId;

    public PoliceUpdateRequestDto() {
    }
}
