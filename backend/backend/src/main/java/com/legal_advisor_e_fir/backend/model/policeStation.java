package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;

import java.util.List;

@Entity
@Table(name = "police_station")
public class policeStation {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long stationId;

    @Column(unique = true, nullable = false)
    private String stationCode;

    private String stationName;
    private String address;
    private String district;
    private String state;

    @OneToMany(mappedBy = "policeStation")
    private List<police> policeList;

//    @OneToMany(mappedBy = "policeStation")
//    private List<FIR> firList;
}
