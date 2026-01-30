package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.fir_status;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalTime;


@Getter
@Setter
public class FirRequestDto {

    @NotBlank(message = "FIR number is required")
    private String firNumber;

    @NotBlank(message = "District is required")
    private String district;

    @NotBlank(message = "Informant name is required")
    private String informantName;

    @NotBlank(message = "Guardian name is required")
    private String informantGuardianName;   // Father / Husband

    @NotBlank(message = "Informant address is required")
    @Size(max = 500, message = "Address cannot exceed 500 characters")
    private String informantAddress;

    @NotBlank(message = "Contact number is required")
    @Pattern(
            regexp = "^[6-9]\\d{9}$",
            message = "Invalid Indian mobile number"
    )
    private String informantContact;

    @NotBlank(message = "Informant email is recommended")
    @Email(message = "Invalid email format")
    private String informantEmail;

    @Pattern(
            regexp = "^[0-9]{3,15}$",
            message = "Fax number must contain only digits and be between 3-15 characters"
    )
    private String informantFax;

    @NotBlank(message = "Incident location is required")
    @Size(min = 5, max = 200, message = "Incident location must be between 5 and 200 characters")
    private String incidentLocation;

    @NotNull(message = "Incident date is required")
    private LocalDate incidentDate;

    @NotNull(message = "Incident time is required")
    private LocalTime incidentTime;

    @NotBlank(message = "Incident description is required")
    @Size(max = 2000, message = "Description cannot exceed 2000 characters")
    private String incidentDescription;

    @NotBlank(message = "Crime category is required")
    @Size(min = 3, max = 100, message = "Crime category must be between 3 and 100 characters")
    private String crimeCategory;   // theft, murder, cyber crime

    @Size(max = 500, message = "IPC sections cannot exceed 500 characters")
    private String ipcSections;     // filled by police

    @Size(max = 1000, message = "Stolen property details cannot exceed 1000 characters")
    private String stolenPropertyDetails;

    @Size(max = 1000, message = "Accused details too long")
    private String accusedDetails;

    @Size(max = 1000, message = "Witness details too long")
    private String witnessDetails;

    @NotNull(message = "FIR status is required")
    private fir_status status;

    @NotNull(message = "Police station ID is required")
    @Positive(message = "Police station ID must be positive")
    private Long policeStationId;

    @Positive(message = "Investigating officer ID must be positive")
    private Long investigatingOfficerId;

    @Positive(message = "Complaint ID must be positive")
    private Long complaintId;

    @NotBlank(message = "FIR writer name is required")
    @Size(min = 2, max = 100, message = "FIR writer name must be between 2 and 100 characters")
    private String firWrittenBy;

    @Size(max = 255, message = "Signature path cannot exceed 255 characters")
    private String informantSignaturePath;

}
