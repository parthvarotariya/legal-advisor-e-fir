package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Complaint;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

@Repository
public interface ComplaintRepo extends JpaRepository<Complaint,Long> {
    List<Complaint> findAllByUser(User user);

    List<Complaint> findAllByPoliceStation(PoliceStation policeStation);
}
