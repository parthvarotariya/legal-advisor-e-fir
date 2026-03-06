package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ChargeSheetRequestDto {

    // Header
    @NotBlank(message = "Charge sheet number is required")
    private String chargeSheetNumber;

    private String district;

    @NotNull(message = "FIR ID is required")
    private Long firId;

    @NotNull(message = "Police station ID is required")
    private Long policeStationId;

    @NotBlank(message = "Report type is required")
    private String reportType;  // CHARGE_SHEET, CLOSURE_UNTRACED, etc.

    // Offence & Legal Classification
    @Size(max = 500)
    private String actsAndSections;

    @NotBlank(message = "Brief facts are required")
    @Size(max = 4000)
    private String briefFacts;

    // Accused (JSON strings)
    private String accusedChargeSheetedJson;
    private String accusedNotChargeSheetedJson;
    private String accusedAbscondingJson;

    // Evidence
    private String seizedPropertyJson;

    @Size(max = 2000)
    private String chainOfCustody;

    @Size(max = 2000)
    private String laboratoryResult;

    // Witnesses (JSON string)
    private String witnessListJson;

    // Verification
    private Boolean complainantNotified;

    // IO who is submitting
    @NotNull(message = "Investigating officer ID is required")
    private Long investigatingOfficerId;
}
