package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceStationRequestDto {
    @NotBlank(message = "Station code is required")
    private String stationCode;
    
    @NotBlank(message = "Station name is required")
    private String stationName;
    
    private String address;
    
    @NotBlank(message = "District is required")
    private String district;
    
    @NotBlank(message = "State is required")
    private String state;
}
