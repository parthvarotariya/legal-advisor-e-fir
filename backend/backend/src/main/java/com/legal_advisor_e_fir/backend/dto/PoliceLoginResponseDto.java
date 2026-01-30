package com.legal_advisor_e_fir.backend.dto;


import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceLoginResponseDto {

    private String token;
    private PoliceResponseDto police;
}
