package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Fir;
import com.legal_advisor_e_fir.backend.model.fir_status;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

@Repository
public interface FirRepo extends JpaRepository<Fir, Long> {

    Optional<Fir> findByFirNumber(String firNumber);

    boolean existsByFirNumber(String firNumber);

    List<Fir> findByPoliceStation_StationId(Long stationId);

    List<Fir> findByStatus(fir_status status);

    List<Fir> findByPoliceStation_StationIdAndStatus(Long stationId, fir_status status);

    List<Fir> findByInvestigatingOfficer_PoliceId(Long policeId);

    // Find FIRs by investigating officer and status
    List<Fir> findByInvestigatingOfficer_PoliceIdAndStatus(Long policeId, fir_status status);

    // Find FIRs by district
    List<Fir> findByDistrict(String district);

    // Find FIRs by district and status
    List<Fir> findByDistrictAndStatus(String district, fir_status status);

    // Find FIRs by crime category
    List<Fir> findByCrimeCategory(String crimeCategory);

    // Find FIRs by crime category and status
    List<Fir> findByCrimeCategoryAndStatus(String crimeCategory, fir_status status);

    // Find FIRs by informant contact (to check if citizen has filed FIRs before)
    List<Fir> findByInformantContact(String informantContact);

    // Find FIRs by informant email
    List<Fir> findByInformantEmail(String informantEmail);

    // Find FIRs registered within a date range
    List<Fir> findByRegisteredAtBetween(LocalDateTime startDate, LocalDateTime endDate);

    // Find FIRs by incident date range
    List<Fir> findByIncidentDateBetween(LocalDate startDate, LocalDate endDate);

    // Find FIRs by police station and incident date range
    List<Fir> findByPoliceStation_StationIdAndIncidentDateBetween(Long stationId, LocalDate startDate, LocalDate endDate);

    // Search FIRs by informant name (case-insensitive partial match)
    List<Fir> findByInformantNameContainingIgnoreCase(String informantName);

    // Search FIRs by incident description (case-insensitive partial match)
    List<Fir> findByIncidentDescriptionContainingIgnoreCase(String keyword);

    // Find recent FIRs (ordered by registration date)
    List<Fir> findTop10ByPoliceStation_StationIdOrderByRegisteredAtDesc(Long stationId);

    // Count FIRs by status
    long countByStatus(fir_status status);

    // Count FIRs by police station
    long countByPoliceStation_StationId(Long stationId);

    // Count FIRs by investigating officer
    long countByInvestigatingOfficer_PoliceId(Long policeId);

    List<Fir> findByInvestigatingOfficerIsNull();

    List<Fir> findByPoliceStation_StationIdAndInvestigatingOfficerIsNull(Long stationId);

    List<Fir> findByStatusIn(List<fir_status> statuses);

    List<Fir> findByPoliceStation_StationIdAndStatusIn(Long stationId, List<fir_status> statuses);

    Optional<Fir> findByComplaint_Id(Long complaintId);

    boolean existsByComplaint_Id(Long complaintId);
}
