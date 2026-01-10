package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Fir;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface FirRepository extends JpaRepository<Fir, Long> {

    Fir findByFirNumber(String firNumber);

    List<Fir> findByPoliceStation_StationId(Long stationId);
}
