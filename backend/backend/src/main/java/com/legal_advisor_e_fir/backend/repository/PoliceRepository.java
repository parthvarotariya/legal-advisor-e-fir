package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Police;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PoliceRepository
        extends JpaRepository<Police, Long> {

    Police findByEmail(String email);
}
