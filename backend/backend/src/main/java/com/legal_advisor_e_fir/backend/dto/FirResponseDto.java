package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.fir_status;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Getter
@Setter
public class FirResponseDto {

    private Long firId;
    private String firNumber;
    private String district;

    // Informant Details
    private String informantName;
    private String informantGuardianName;
    private String informantAddress;
    private String informantContact;
    private String informantEmail;

    // Incident Details
    private String incidentLocation;
    private LocalDate incidentDate;
    private LocalTime incidentTime;
    private String incidentDescription;

    // Crime Details
    private String crimeCategory;
    private String ipcSections;
    private String stolenPropertyDetails;
    private String accusedDetails;
    private String witnessDetails;

    // Status
    private fir_status status;

    // Police Station Details
    private Long policeStationId;
    private String policeStationName;
    private String policeStationCode;
    private String policeStationAddress;

    // Investigating Officer Details
    private Long investigatingOfficerId;
    private String investigatingOfficerName;
    private String investigatingOfficerBadgeNumber;
    private String investigatingOfficerRank;
    private String investigatingOfficerContact;

    // Complaint Details
    private Long complaintId;
    private String complaintDescription;
    private String complaintPredictedCategory;

    // Additional Info
    private String firWrittenBy;
    private String informantSignaturePath;
    private LocalDateTime registeredAt;

    // Default constructor
    public FirResponseDto() {
    }

    // Constructor with essential fields
    public FirResponseDto(Long firId, String firNumber, String district, 
                         String informantName, String incidentLocation, 
                         LocalDate incidentDate, String crimeCategory, 
                         fir_status status, LocalDateTime registeredAt) {
        this.firId = firId;
        this.firNumber = firNumber;
        this.district = district;
        this.informantName = informantName;
        this.incidentLocation = incidentLocation;
        this.incidentDate = incidentDate;
        this.crimeCategory = crimeCategory;
        this.status = status;
        this.registeredAt = registeredAt;
    }

    // Full constructor
    public FirResponseDto(Long firId, String firNumber, String district,
                         String informantName, String informantGuardianName,
                         String informantAddress, String informantContact,
                         String informantEmail,
                         String incidentLocation, LocalDate incidentDate,
                         LocalTime incidentTime, String incidentDescription,
                         String crimeCategory, String ipcSections,
                         String stolenPropertyDetails, String accusedDetails,
                         String witnessDetails, fir_status status,
                         Long policeStationId, String policeStationName,
                         String policeStationCode, String policeStationAddress,
                         Long investigatingOfficerId, String investigatingOfficerName,
                         String investigatingOfficerBadgeNumber, String investigatingOfficerRank,
                         String investigatingOfficerContact, String firWrittenBy,
                         String informantSignaturePath, LocalDateTime registeredAt) {
        this.firId = firId;
        this.firNumber = firNumber;
        this.district = district;
        this.informantName = informantName;
        this.informantGuardianName = informantGuardianName;
        this.informantAddress = informantAddress;
        this.informantContact = informantContact;
        this.informantEmail = informantEmail;
        this.incidentLocation = incidentLocation;
        this.incidentDate = incidentDate;
        this.incidentTime = incidentTime;
        this.incidentDescription = incidentDescription;
        this.crimeCategory = crimeCategory;
        this.ipcSections = ipcSections;
        this.stolenPropertyDetails = stolenPropertyDetails;
        this.accusedDetails = accusedDetails;
        this.witnessDetails = witnessDetails;
        this.status = status;
        this.policeStationId = policeStationId;
        this.policeStationName = policeStationName;
        this.policeStationCode = policeStationCode;
        this.policeStationAddress = policeStationAddress;
        this.investigatingOfficerId = investigatingOfficerId;
        this.investigatingOfficerName = investigatingOfficerName;
        this.investigatingOfficerBadgeNumber = investigatingOfficerBadgeNumber;
        this.investigatingOfficerRank = investigatingOfficerRank;
        this.investigatingOfficerContact = investigatingOfficerContact;
        this.firWrittenBy = firWrittenBy;
        this.informantSignaturePath = informantSignaturePath;
        this.registeredAt = registeredAt;
    }
}
