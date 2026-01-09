package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "police")
public class police {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long policeId;

    @Column(nullable = false)
    private String name;

    @Column(unique = true, nullable = false)
    private String badgeNumber;

    @Column(nullable = false)
    private String rank;

    @Column(unique = true, nullable = false)
    private String email;

    @Column(nullable = false)
    private String mobileNumber;

    @Column(nullable = false)
    private String password;

    @Enumerated(EnumType.STRING)
    private Role role;

//    @Enumerated(EnumType.STRING)
//    private Status status;

    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private policeStation policeStation;

    private LocalDateTime createdAt;
}
