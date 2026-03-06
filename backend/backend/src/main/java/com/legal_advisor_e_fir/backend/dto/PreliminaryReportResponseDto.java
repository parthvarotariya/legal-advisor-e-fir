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

    // ==========================================
    // NEW FIELDS FOR BNSS 2023 PE PROTOCOL
    // ==========================================

    // 1. Authorization
    private Long permissionGrantedByDspId;
    private String permissionGrantedByDspName; // For display purposes
    private String peCategory;

    // 2. Timeline Tracking (Mandatory 14 days)
    private LocalDate peStartDate;
    private LocalDate peDeadline;

    // 3. Closure
    private String reasonForRefusal;
    private Boolean informantNotifiedOfRefusal;

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

    // Full constructor with all fields including BNSS 2023 PE Protocol
    public PreliminaryReportResponseDto(Long reportId, LocalDateTime submittedAt,
                                        String investigationNarrative, Boolean cognizableOffence,
                                        String informantName, String informantAddress,
                                        String informantContact, String informantEmail,
                                        String incidentLocation, LocalDate incidentDate, LocalTime incidentTime,
                                        String crimeCategory, String ipcSections, String stolenPropertyDetails,
                                        String draftAccusedDetails, String draftWitnessDetails, String witnessStatement,
                                        Long complaintId, String complaintDescription,
                                        Long investigatingOfficerId, String investigatingOfficerName,
                                        String investigatingOfficerBadgeNumber, String investigatingOfficerRank,
                                        Long stationId, String stationName, String stationCode,
                                        Long permissionGrantedByDspId, String permissionGrantedByDspName,
                                        String peCategory, LocalDate peStartDate, LocalDate peDeadline,
                                        String reasonForRefusal, Boolean informantNotifiedOfRefusal) {
        this.reportId = reportId;
        this.submittedAt = submittedAt;
        this.investigationNarrative = investigationNarrative;
        this.cognizableOffence = cognizableOffence;
        this.informantName = informantName;
        this.informantAddress = informantAddress;
        this.informantContact = informantContact;
        this.informantEmail = informantEmail;
        this.incidentLocation = incidentLocation;
        this.incidentDate = incidentDate;
        this.incidentTime = incidentTime;
        this.crimeCategory = crimeCategory;
        this.ipcSections = ipcSections;
        this.stolenPropertyDetails = stolenPropertyDetails;
        this.draftAccusedDetails = draftAccusedDetails;
        this.draftWitnessDetails = draftWitnessDetails;
        this.witnessStatement = witnessStatement;
        this.complaintId = complaintId;
        this.complaintDescription = complaintDescription;
        this.investigatingOfficerId = investigatingOfficerId;
        this.investigatingOfficerName = investigatingOfficerName;
        this.investigatingOfficerBadgeNumber = investigatingOfficerBadgeNumber;
        this.investigatingOfficerRank = investigatingOfficerRank;
        this.stationId = stationId;
        this.stationName = stationName;
        this.stationCode = stationCode;
        // BNSS 2023 PE Protocol fields
        this.permissionGrantedByDspId = permissionGrantedByDspId;
        this.permissionGrantedByDspName = permissionGrantedByDspName;
        this.peCategory = peCategory;
        this.peStartDate = peStartDate;
        this.peDeadline = peDeadline;
        this.reasonForRefusal = reasonForRefusal;
        this.informantNotifiedOfRefusal = informantNotifiedOfRefusal;
    }
}
