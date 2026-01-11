package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Complaint;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ComplaintRepo extends JpaRepository<Complaint,Long> {

}
