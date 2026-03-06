package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalTime;

@Getter
@Setter
public class PreliminaryReportRequestDto {

    @NotBlank(message = "Investigation narrative is required")
    @Size(min = 20, max = 4000, message = "Investigation narrative must be between 20 and 4000 characters")
    private String investigationNarrative;

    @NotNull(message = "Cognizable offence decision required")
    private Boolean cognizableOffence;

    @NotBlank(message = "Informant name required")
    private String informantName;

    @NotBlank(message = "Informant address required")
    private String informantAddress;

    @Pattern(regexp = "^[0-9]{10}$", message = "Invalid contact number")
    private String informantContact;

    @Email(message = "Invalid email format")
    private String informantEmail;

    @NotBlank(message = "Incident location required")
    private String incidentLocation;

    @PastOrPresent(message = "Incident date cannot be future")
    private LocalDate incidentDate;

    private LocalTime incidentTime;

    @NotBlank(message = "Crime category required")
    private String crimeCategory;

    private String ipcSections;

    @Size(max = 1000, message = "Stolen property details cannot exceed 1000 characters")
    private String stolenPropertyDetails;

    @Size(max = 1000, message = "Draft accused details cannot exceed 1000 characters")
    private String draftAccusedDetails;

    @Size(max = 1000, message = "Draft witness details cannot exceed 1000 characters")
    private String draftWitnessDetails;

    @Size(max = 1000, message = "Witness statement cannot exceed 1000 characters")
    private String witnessStatement;

    @NotNull(message = "Complaint ID is required")
    private Long complaintId;

    @NotNull(message = "Investigating officer ID is required")
    private Long investigatingOfficerId;

    @NotNull(message = "Station ID is required")
    private Long stationId;

    // ==========================================
    // NEW FIELDS FOR BNSS 2023 PE PROTOCOL
    // ==========================================

    // 1. Authorization
    private Long permissionGrantedByDspId; // Tracks the DSP who allowed the PE
    private String peCategory; // e.g., "Matrimonial", "Commercial", "Medical Negligence"

    // 2. Timeline Tracking (Mandatory 14 days)
    private LocalDate peStartDate;
    private LocalDate peDeadline; // Should be calculated as peStartDate + 14 days

    // 3. Closure
    @Size(max = 1000, message = "Reason for refusal cannot exceed 1000 characters")
    private String reasonForRefusal; // If no FIR is registered, why?
    private Boolean informantNotifiedOfRefusal;
}
