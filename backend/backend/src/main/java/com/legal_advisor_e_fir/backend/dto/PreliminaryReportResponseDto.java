package com.legal_advisor_e_fir.backend.dto;

import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Getter
@Setter
public class PreliminaryReportResponseDto {

    private Long reportId;
    private LocalDateTime submittedAt;

    // Investigation Details
    private String investigationNarrative;
    private Boolean cognizableOffence;

    // Informant Details
    private String informantName;
    private String informantAddress;
    private String informantContact;
    private String informantEmail;

    // Incident Details
    private String incidentLocation;
    private LocalDate incidentDate;
    private LocalTime incidentTime;

    // Crime Details
    private String crimeCategory;
    private String ipcSections;
    private String stolenPropertyDetails;

    // Draft Details
    private String draftAccusedDetails;
    private String draftWitnessDetails;
    private String witnessStatement;

    // Complaint Details
    private Long complaintId;
    private String complaintDescription;

    // Investigating Officer Details
    private Long investigatingOfficerId;
    private String investigatingOfficerName;
    private String investigatingOfficerBadgeNumber;
    private String investigatingOfficerRank;

    // Police Station Details
    private Long stationId;
    private String stationName;
    private String stationCode;

    // Default constructor
    public PreliminaryReportResponseDto() {
    }

    // Constructor with essential fields
    public PreliminaryReportResponseDto(Long reportId, LocalDateTime submittedAt,
                                        Boolean cognizableOffence, String informantName,
                                        String incidentLocation, LocalDate incidentDate,
                                        String crimeCategory, String investigatingOfficerName,
                                        String stationName) {
        this.reportId = reportId;
        this.submittedAt = submittedAt;
        this.cognizableOffence = cognizableOffence;
        this.informantName = informantName;
        this.incidentLocation = incidentLocation;
        this.incidentDate = incidentDate;
        this.crimeCategory = crimeCategory;
        this.investigatingOfficerName = investigatingOfficerName;
        this.stationName = stationName;
    }
}
