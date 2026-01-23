package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.Role;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceRequestDto {

    @NotBlank(message = "Name is required")
    @Size(min = 2, max = 100, message = "Name must be between 2 and 100 characters")
    private String name;

    @NotBlank(message = "Badge number is required")
    @Pattern(
            regexp = "^[A-Z0-9]{6,15}$",
            message = "Badge number must be 6-15 characters long and contain only uppercase letters and numbers"
    )
    private String badgeNumber;

    @NotBlank(message = "Rank is required")
    @Size(min = 2, max = 50, message = "Rank must be between 2 and 50 characters")
    private String rank;

    @NotBlank(message = "Email is required")
    @Email(message = "Invalid email format")
    private String email;

    @NotBlank(message = "Mobile number is required")
    @Pattern(
            regexp = "^[6-9][0-9]{9}$",
            message = "Mobile number must be a valid 10-digit Indian number"
    )
    private String mobileNumber;

    @NotBlank(message = "Password is required")
    @Size(min = 8, max = 100, message = "Password must be at least 8 characters long")
    private String password;

    @NotNull(message = "Role is required")
    private Role role;

    @NotNull(message = "Station ID is required")
    @Positive(message = "Station ID must be a positive number")
    private Long stationId;

    public PoliceRequestDto() {
    }
}
