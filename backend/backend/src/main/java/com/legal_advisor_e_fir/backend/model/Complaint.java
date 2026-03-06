package com.legal_advisor_e_fir.backend.model;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;
import org.springframework.boot.context.properties.bind.DefaultValue;

import java.time.LocalDateTime;

@Getter
@Setter
@Entity
@Table(name = "complaints")
public class Complaint {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = true, columnDefinition = "TEXT")
    private String description;


    @Column(nullable = true)
    private String actualCategory;

    @Column(nullable = true)
    private String predictedCategory;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, columnDefinition = "varchar(50)")
    private ComplaintStatus status = ComplaintStatus.RECEIVED;

    public Complaint(){

    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name="user_id",nullable = false)
    @JsonIgnoreProperties({"complaintList", "password"})
    private User user;

    @ManyToOne
    @JoinColumn(name="station_id")
    @JsonIgnoreProperties({"complaints", "policeOfficers", "subdivision"})
    private PoliceStation policeStation;

    @ManyToOne
    @JoinColumn(name="assigned_officer_id")
    @JsonIgnoreProperties({"policeStation", "password", "dspSubdivision"})
    private Police assignedOfficer;

    @OneToOne(mappedBy = "complaint", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private Fir fir;

    @OneToOne(mappedBy = "complaint")
    private PreliminaryReport preliminaryReport;
}
