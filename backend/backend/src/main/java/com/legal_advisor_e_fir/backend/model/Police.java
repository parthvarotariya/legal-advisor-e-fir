package com.legal_advisor_e_fir.backend.model;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.List;

@Entity
@Table(name = "police")
@Getter
@Setter
public class Police {

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

    @CreationTimestamp
    @Column(updatable = false)
    private LocalDateTime createdAt;

    @UpdateTimestamp
    private LocalDateTime updatedAt;

//    @Enumerated(EnumType.STRING)
//    private Status status;

    @ManyToOne
    @JoinColumn(name = "station_id", nullable = false)
    private PoliceStation policeStation;

    @OneToMany(mappedBy = "investigatingOfficer")
    private List<Fir> firs;
}
