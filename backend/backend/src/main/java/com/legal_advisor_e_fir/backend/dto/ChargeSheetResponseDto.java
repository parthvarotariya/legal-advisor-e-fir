package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.ChargeSheetStatus;
import com.legal_advisor_e_fir.backend.model.FinalReportType;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
public class ChargeSheetResponseDto {

    private Long chargeSheetId;

    // Header & Jurisdictional Data
    private String chargeSheetNumber;
    private String district;
    private FinalReportType reportType;
    private ChargeSheetStatus status;

    // FIR Reference
    private Long firId;
    private String firNumber;
    private String firCrimeCategory;
    private String firIncidentDescription;

    // Police Station
    private Long policeStationId;
    private String policeStationName;
    private String policeStationCode;

    // Offence & Legal Classification
    private String actsAndSections;
    private String briefFacts;

    // Accused (JSON strings – frontend parses)
    private String accusedChargeSheetedJson;
    private String accusedNotChargeSheetedJson;
    private String accusedAbscondingJson;

    // Evidence
    private String seizedPropertyJson;
    private String chainOfCustody;
    private String laboratoryResult;

    // Witnesses
    private String witnessListJson;

    // Verification
    private Boolean complainantNotified;

    // IO details
    private Long investigatingOfficerId;
    private String investigatingOfficerName;
    private String investigatingOfficerBadgeNumber;
    private String investigatingOfficerRank;

    // PI (Approver) details
    private Long approvingOfficerId;
    private String approvingOfficerName;
    private String approvingOfficerBadgeNumber;

    // PI Review
    private String piSuggestions;
    private Integer revisionCount;

    // Timestamps
    private LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private LocalDateTime submittedAt;
    private LocalDateTime approvedAt;
    private LocalDateTime dispatchedAt;

    public ChargeSheetResponseDto() {}
}
