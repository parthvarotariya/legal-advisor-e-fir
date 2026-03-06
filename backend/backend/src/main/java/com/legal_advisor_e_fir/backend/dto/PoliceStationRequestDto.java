package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceStationRequestDto {
    @NotBlank(message = "Station code is required")
    @Pattern(
            regexp = "^[A-Z0-9-]{4,10}$",
            message = "Station code must be 4-10 characters long and contain only uppercase letters, numbers, and hyphens"
    )
    private String stationCode;
    
    @NotBlank(message = "Station name is required")
    @Size(min = 3, max = 100, message = "Station name must be between 3 and 100 characters")
    private String stationName;
    
    @NotBlank(message = "Address is required")
    @Size(min = 10, max = 200, message = "Address must be between 10 and 200 characters")
    private String address;
    
    @NotBlank(message = "District is required")
    @Size(min = 2, max = 50, message = "District name must be between 2 and 50 characters")
    private String district;
    
    @NotBlank(message = "State is required")
    @Size(min = 2, max = 50, message = "State name must be between 2 and 50 characters")
    private String state;

    private Long subdivisionId;
}
