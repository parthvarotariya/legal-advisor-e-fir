package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDate;
import java.time.LocalTime;
import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "fir")
public class Fir {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long firId;


    @Column(unique = true, nullable = false)
    private String firNumber;

    private String district;

    private String informantName;
    private String informantGuardianName;
    private String informantAddress;
    private String informantContact;
    private String informantEmail;

    private String incidentLocation;
    private LocalDate incidentDate;
    private LocalTime incidentTime;

    @Column(length = 2000)
    private String incidentDescription;

    private String crimeCategory;
    private String ipcSections;
    private String stolenPropertyDetails;

    @Column(length = 1000)
    private String accusedDetails;

    @Column(length = 1000)
    private String witnessDetails;

    @Enumerated(EnumType.STRING)
    @Column(columnDefinition = "varchar(50)")
    private FirStatus status;


    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private PoliceStation policeStation;

    @ManyToOne
    @JoinColumn(name = "investigating_officer_id")
    private Police investigatingOfficer;

    @OneToOne
    @JoinColumn(name = "complaint_id")
    private Complaint complaint;

    private String firWrittenBy;
    private String informantSignaturePath;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime registeredAt;

    // ==========================================
    // NEW FIELDS FOR BNSS 2023 COMPLIANCE
    // ==========================================

    // 1. Jurisdiction & Zero FIR
    @Column(nullable = false)
    private Boolean isZeroFir = false;
    private String destinationPoliceStation; // If Zero FIR, where is it going?

    // 2. Electronic Communication (e-FIR)
    @Column(nullable = false)
    private Boolean isEfir = false;
    //private String generalDiaryReference; //not now
    private LocalDateTime signatureDeadline; // Must sign within 3 days
    private Boolean isSignatureObtained;

    // 3. Vulnerable Victim Protection
    @Column(nullable = false)
    private Boolean isVictimWoman = false;
    private Boolean recordedByWomanOfficer;
    
    @Column(nullable = false)
    private Boolean isDisabledVictim = false;
    private String interpreterOrEducatorName;
    private String videoRecordingPath; // Mandatory if victim is disabled
    private Boolean isMagistrateStatementRecorded;
}
