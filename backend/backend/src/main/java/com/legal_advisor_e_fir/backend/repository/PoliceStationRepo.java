package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.PoliceStation;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PoliceStationRepo
        extends JpaRepository<PoliceStation, Long> {
    
    Optional<PoliceStation> findByStationCode(String stationCode);
    
    List<PoliceStation> findByStationName(String stationName);
    
    List<PoliceStation> findByDistrict(String district);
    
    List<PoliceStation> findByState(String state);
}
