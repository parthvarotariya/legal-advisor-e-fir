package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.FirStatus;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class FirFromReportRequestDto {

    @NotNull(message = "Preliminary report ID is required")
    @Positive(message = "Report ID must be positive")
    private Long reportId;

    @NotBlank(message = "FIR number is required")
    private String firNumber;

    @NotBlank(message = "District is required")
    private String district;

    @NotBlank(message = "Incident description is required")
    @Size(max = 2000, message = "Description cannot exceed 2000 characters")
    private String incidentDescription;

    @NotNull(message = "FIR status is required")
    private FirStatus status;

    @NotBlank(message = "FIR writer name is required")
    @Size(min = 2, max = 100, message = "FIR writer name must be between 2 and 100 characters")
    private String firWrittenBy;

    @Size(max = 255, message = "Signature path cannot exceed 255 characters")
    private String informantSignaturePath;

    // ==========================================
    // NEW FIELDS FOR BNSS 2023 COMPLIANCE
    // ==========================================

    // 1. Jurisdiction & Zero FIR
    private Boolean isZeroFir = false;
    private String destinationPoliceStation;

    // 2. Electronic Communication (e-FIR)
    private Boolean isEfir = false;
    private Boolean isSignatureObtained;

    // 3. Vulnerable Victim Protection
    private Boolean isVictimWoman = false;
    private Boolean recordedByWomanOfficer;
    
    private Boolean isDisabledVictim = false;
    private String interpreterOrEducatorName;
    private String videoRecordingPath;
    private Boolean isMagistrateStatementRecorded;
}
