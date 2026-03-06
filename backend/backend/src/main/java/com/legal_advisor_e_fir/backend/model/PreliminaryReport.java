package com.legal_advisor_e_fir.backend.model;
import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;

@Getter
@Setter
@Entity
public class PreliminaryReport {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long reportId;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime submittedAt;


    @NotBlank(message = "Investigation narrative is required")
    @Size(min = 20, max = 4000)
    @Column(length = 4000)
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

    private String stolenPropertyDetails;




    @Size(max = 1000)
    private String draftAccusedDetails;

    @Size(max = 1000)
    private String draftWitnessDetails;

    @Size(max = 1000)
    private String witnessStatement;


    @NotNull(message = "Complaint is required")
    @OneToOne
    @JoinColumn(name = "complaint_id", nullable = false)
    private Complaint complaint;

    @NotNull(message = "Investigating officer required")
    @ManyToOne
    @JoinColumn(name = "investigating_officer_id")
    private Police investigatingOfficer;

    @NotNull(message = "Police station required")
    @ManyToOne
    @JoinColumn(name = "station_id")
    private PoliceStation station;

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
    @Column(length = 1000)
    private String reasonForRefusal; // If no FIR is registered, why?
    private Boolean informantNotifiedOfRefusal;
}
