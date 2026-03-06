package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.FirStatus;
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
    private FirStatus status;

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

    // ==========================================
    // NEW FIELDS FOR BNSS 2023 COMPLIANCE
    // ==========================================

    // 1. Jurisdiction & Zero FIR
    private Boolean isZeroFir;
    private String destinationPoliceStation;

    // 2. Electronic Communication (e-FIR)
    private Boolean isEfir;
    private LocalDateTime signatureDeadline;
    private Boolean isSignatureObtained;

    // 3. Vulnerable Victim Protection
    private Boolean isVictimWoman;
    private Boolean recordedByWomanOfficer;
    
    private Boolean isDisabledVictim;
    private String interpreterOrEducatorName;
    private String videoRecordingPath;
    private Boolean isMagistrateStatementRecorded;

    // Default constructor
    public FirResponseDto() {
    }

    // Constructor with essential fields
    public FirResponseDto(Long firId, String firNumber, String district, 
                         String informantName, String incidentLocation, 
                         LocalDate incidentDate, String crimeCategory, 
                         FirStatus status, LocalDateTime registeredAt) {
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
                         String witnessDetails, FirStatus status,
                         Long policeStationId, String policeStationName,
                         String policeStationCode, String policeStationAddress,
                         Long investigatingOfficerId, String investigatingOfficerName,
                         String investigatingOfficerBadgeNumber, String investigatingOfficerRank,
                         String investigatingOfficerContact, String firWrittenBy,
                         String informantSignaturePath, LocalDateTime registeredAt,
                         Boolean isZeroFir, String destinationPoliceStation,
                         Boolean isEfir, LocalDateTime signatureDeadline, Boolean isSignatureObtained,
                         Boolean isVictimWoman, Boolean recordedByWomanOfficer,
                         Boolean isDisabledVictim, String interpreterOrEducatorName,
                         String videoRecordingPath, Boolean isMagistrateStatementRecorded) {
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
        // BNSS 2023 fields
        this.isZeroFir = isZeroFir;
        this.destinationPoliceStation = destinationPoliceStation;
        this.isEfir = isEfir;
        this.signatureDeadline = signatureDeadline;
        this.isSignatureObtained = isSignatureObtained;
        this.isVictimWoman = isVictimWoman;
        this.recordedByWomanOfficer = recordedByWomanOfficer;
        this.isDisabledVictim = isDisabledVictim;
        this.interpreterOrEducatorName = interpreterOrEducatorName;
        this.videoRecordingPath = videoRecordingPath;
        this.isMagistrateStatementRecorded = isMagistrateStatementRecorded;
    }
}
