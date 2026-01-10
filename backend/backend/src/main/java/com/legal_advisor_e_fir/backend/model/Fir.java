package com.legal_advisor_e_fir.backend.model;


import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
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

    /* Informant Details */
    private String informantName;
    private String informantAddress;
    private String informantContact;

    /* Incident Details */
    private LocalDate incidentDate;
    private LocalTime incidentTime;
    private String incidentLocation;

    @Column(length = 2000)
    private String incidentDescription;

    @Column(length = 1000)
    private String accusedDetails;

    @Column(length = 1000)
    private String witnessDetails;

    private String crime_category;
    private String ipcSections;

    @Enumerated(EnumType.STRING)
    private fir_status status;

    /* Police Mapping */
    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private PoliceStation policeStation;

    @ManyToOne
    @JoinColumn(name = "investigating_officer_id")
    private Police investigatingOfficer;

    /* Informant Signature */
    private String informantSignaturePath; // image/pdf path

    /* Metadata */
    private LocalDateTime registeredAt;
}
