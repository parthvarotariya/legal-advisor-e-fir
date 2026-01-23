package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.Role;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class PoliceResponseDto {

    private Long policeId;
    private String name;
    private String badgeNumber;
    private String rank;
    private String email;
    private String mobileNumber;
    private Role role;
    private Long stationId;
    private String stationName;
    private String stationCode;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    public PoliceResponseDto() {
    }

    public PoliceResponseDto(Long policeId, String name, String badgeNumber, String rank, 
                            String email, String mobileNumber, Role role, 
                            Long stationId, String stationName, String stationCode,
                            LocalDateTime createdAt, LocalDateTime updatedAt) {
        this.policeId = policeId;
        this.name = name;
        this.badgeNumber = badgeNumber;
        this.rank = rank;
        this.email = email;
        this.mobileNumber = mobileNumber;
        this.role = role;
        this.stationId = stationId;
        this.stationName = stationName;
        this.stationCode = stationCode;
        this.createdAt = createdAt;
        this.updatedAt = updatedAt;
    }
}
