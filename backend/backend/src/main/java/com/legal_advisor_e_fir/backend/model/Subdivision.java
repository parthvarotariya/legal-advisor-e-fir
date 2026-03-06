package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Entity representing a Police Subdivision.
 * A subdivision is an administrative unit that supervises multiple police stations
 * and is typically headed by a Deputy Superintendent of Police (DSP).
 */
@Entity
@Table(name = "subdivision")
@Getter
@Setter
public class Subdivision {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long subdivisionId;

    /**
     * Unique code identifying the subdivision (e.g., "SUB001", "NORTH-DIV", "RJK-TLK").
     */
    @Column(unique = true, nullable = false, length = 15)
    private String subdivisionCode;

    @Column(nullable = false, length = 100)
    private String subdivisionName;

    @Column(nullable = false, length = 50)
    private String district;

    @Column(nullable = false, length = 50)
    private String state;


    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "dsp_officer_id")
    private Police dspOfficer;

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

    @OneToMany(mappedBy = "subdivision", fetch = FetchType.LAZY)
    private List<PoliceStation> policeStations;
}
