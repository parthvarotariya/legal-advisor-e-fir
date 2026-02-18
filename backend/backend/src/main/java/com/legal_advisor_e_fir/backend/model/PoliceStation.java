package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;

import java.util.List;

@Entity
@Table(name = "police_station")
@Getter
@Setter
public class PoliceStation{

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long stationId;

    @Column(unique = true, nullable = false)
    private String stationCode;

    private String stationName;
    private String address;
    private String district;
    private String state;

    @OneToMany(mappedBy = "policeStation",fetch = FetchType.LAZY)
    private List<Police> policeList;

    @OneToMany(mappedBy = "policeStation",fetch = FetchType.LAZY)
    private List<Fir> firList;

    @OneToMany(mappedBy = "policeStation",fetch = FetchType.LAZY)
    private List<Complaint> complaintList;

    @OneToMany(mappedBy = "station")
    private List<PreliminaryReport> preliminaryReports;
}
