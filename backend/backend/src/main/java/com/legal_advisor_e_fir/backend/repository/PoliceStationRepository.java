package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.PoliceStation;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PoliceStationRepository
        extends JpaRepository<PoliceStation, Long> {
}
