package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDate;
import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "charge_sheet")
public class ChargeSheet {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long chargeSheetId;

    // =============================================
    // 1. HEADER & JURISDICTIONAL DATA
    // =============================================

    @Column(unique = true, nullable = false)
    private String chargeSheetNumber;  // Unique serial per station per year

    private String district;

    @ManyToOne
    @JoinColumn(name = "fir_id", nullable = false)
    private Fir fir;

    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private PoliceStation policeStation;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, columnDefinition = "varchar(50)")
    private FinalReportType reportType;  // CHARGE_SHEET, UNTRACED, etc.

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, columnDefinition = "varchar(50)")
    private ChargeSheetStatus status;

    // =============================================
    // 2. OFFENCE & LEGAL CLASSIFICATION
    // =============================================

    @Column(length = 500)
    private String actsAndSections;  // BNS/IPC sections, Arms Act, NDPS etc.

    @Column(length = 4000)
    private String briefFacts;  // Chronological narrative of investigation findings

    // =============================================
    // 3. ACCUSED PERSONS (JSON arrays stored as text)
    // =============================================

    @Column(columnDefinition = "TEXT")
    private String accusedChargeSheetedJson;
    // JSON array: [{name, fatherName, dob, nationality, religion, caste, scStStatus,
    //   occupation, arrestDate, bailDate, suretyDetails, dateForwardedToCourt}]

    @Column(columnDefinition = "TEXT")
    private String accusedNotChargeSheetedJson;
    // JSON array: [{name, fatherName, reasonForNotProsecuting}]

    @Column(columnDefinition = "TEXT")
    private String accusedAbscondingJson;
    // JSON array: [{name, fatherName, lastKnownAddress, warrantIssued}]

    // =============================================
    // 4. EVIDENCE & RECOVERY (Panchnama)
    // =============================================

    @Column(columnDefinition = "TEXT")
    private String seizedPropertyJson;
    // JSON array: [{description, estimatedValue, muddamalNumber, psPropertyRegNo}]

    @Column(length = 2000)
    private String chainOfCustody;  // Sec 193 BNSS electronic evidence handling

    @Column(length = 2000)
    private String laboratoryResult;  // FSL findings summary

    // =============================================
    // 5. WITNESS LIST
    // =============================================

    @Column(columnDefinition = "TEXT")
    private String witnessListJson;
    // JSON array: [{serialNo, name, fatherName, address, age, evidenceType}]
    // evidenceType: EYE_WITNESS, SEIZURE_WITNESS, MEDICAL_WITNESS, EXPERT, etc.

    // =============================================
    // 6. VERIFICATION & DISPATCH
    // =============================================

    @Column(nullable = false)
    private Boolean complainantNotified = false;  // Refer notice sent?

    // IO (PSI) who prepared this charge sheet
    @ManyToOne
    @JoinColumn(name = "investigating_officer_id", nullable = false)
    private Police investigatingOfficer;

    // PI (SHO) who approved/forwarded the report
    @ManyToOne
    @JoinColumn(name = "approving_officer_id")
    private Police approvingOfficer;

    private LocalDateTime approvedAt;         // PI approval timestamp
    private LocalDateTime dispatchedAt;       // When sent to court

    // =============================================
    // PI REVIEW
    // =============================================

    @Column(length = 2000)
    private String piSuggestions;  // Suggestions when returned for revision

    private Integer revisionCount = 0;  // How many times returned for revision

    // =============================================
    // TIMESTAMPS
    // =============================================

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    private LocalDateTime submittedAt;  // When IO submitted to PI
}
