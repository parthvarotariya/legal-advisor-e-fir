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
    private String informantGuardianName;   // Father / Husband
    private String informantAddress;
    private String informantContact;
    private String informantEmail;
    private String informantFax;

    private String incidentLocation;
    private LocalDate incidentDate;
    private LocalTime incidentTime;

    @Column(length = 2000)
    private String incidentDescription;

    private String crimeCategory;           // theft, murder, cyber crime
    private String ipcSections;             // written by police
    private String stolenPropertyDetails;   // if applicable

    @Column(length = 1000)
    private String accusedDetails;

    @Column(length = 1000)
    private String witnessDetails;

    @Enumerated(EnumType.STRING)
    private fir_status status;


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
}
