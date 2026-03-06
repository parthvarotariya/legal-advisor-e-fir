package com.legal_advisor_e_fir.backend.dto;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

import java.time.LocalDateTime;

/**
 * Data Transfer Object for Subdivision responses.
 * Contains subdivision details along with nested DSP officer information.
 */
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class SubdivisionResponseDto {

    private Long subdivisionId;
    private String subdivisionCode;
    private String subdivisionName;
    private String district;
    private String state;
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;

    /**
     * Details of the DSP officer assigned to this subdivision.
     * Will be null if no DSP is assigned.
     */
    private DspOfficerDto dspOfficer;

    /**
     * Count of police stations under this subdivision's jurisdiction.
     */
    private Integer stationCount;

    /**
     * Nested DTO for DSP officer information.
     */
    @Getter
    @Setter
    @NoArgsConstructor
    @AllArgsConstructor
    public static class DspOfficerDto {
        private Long policeId;
        private String name;
        private String badgeNumber;
        private String rank;
        private String email;
        private String mobileNumber;
    }
}
