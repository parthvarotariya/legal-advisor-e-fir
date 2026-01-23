package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.Role;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PoliceRepository extends JpaRepository<Police, Long> {

    // Find by unique identifiers
    Optional<Police> findByEmail(String email);
    
    Optional<Police> findByBadgeNumber(String badgeNumber);
    
    // Check existence for validation
    boolean existsByEmail(String email);
    
    boolean existsByBadgeNumber(String badgeNumber);
    
    // Find by police station
    List<Police> findByPoliceStationStationId(Long stationId);
    
    // Find by role
    List<Police> findByRole(Role role);
}
