package com.legal_advisor_e_fir.backend.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class PoliceStationResponseDto {
    private Long stationId;
    private String stationCode;
    private String stationName;
    private String address;
    private String district;
    private String state;
}
