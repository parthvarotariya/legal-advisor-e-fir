package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

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

    /* FIR Reference */
    @Column(unique = true, nullable = false)
    private String firNumber;

    /* Police Jurisdiction */
    private String district;

    /* Informant Details (as per FIR Proforma) */
    private String informantName;
    private String informantGuardianName;   // Father / Husband
    private String informantAddress;
    private String informantContact;
    private String informantEmail;
    private String informantFax;

    /* Place, Date & Time of Occurrence */
    private String incidentLocation;
    private LocalDate incidentDate;
    private LocalTime incidentTime;

    /* Complaint Description */
    @Column(length = 2000)
    private String incidentDescription;

    /* Offence Details */
    private String crimeCategory;           // theft, murder, cyber crime
    private String ipcSections;             // written by police
    private String stolenPropertyDetails;   // if applicable

    /* Accused & Witness Details */
    @Column(length = 1000)
    private String accusedDetails;

    @Column(length = 1000)
    private String witnessDetails;

    /* FIR Status */
    @Enumerated(EnumType.STRING)
    private fir_status status;

    /* Police Mapping */
    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private PoliceStation policeStation;

    @ManyToOne
    @JoinColumn(name = "investigating_officer_id")
    private Police investigatingOfficer;

    /* Office Use */
    private String firWrittenBy;

    /* Informant Signature / Thumb Impression */
    private String informantSignaturePath;

    /* Metadata */
    private LocalDateTime registeredAt;
}
