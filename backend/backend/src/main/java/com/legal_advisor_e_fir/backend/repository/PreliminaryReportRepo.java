package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.PreliminaryReport;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface PreliminaryReportRepo extends JpaRepository<PreliminaryReport, Long> {

    Optional<PreliminaryReport> findByComplaint_Id(Long complaintId);

    boolean existsByComplaint_Id(Long complaintId);

    List<PreliminaryReport> findByStation_StationId(Long stationId);

    List<PreliminaryReport> findByInvestigatingOfficer_PoliceId(Long policeId);

    List<PreliminaryReport> findByCognizableOffence(Boolean cognizableOffence);

    List<PreliminaryReport> findByCrimeCategory(String crimeCategory);

    List<PreliminaryReport> findByStation_Subdivision_SubdivisionId(Long subdivisionId);
}
