package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.ChargeSheet;
import com.legal_advisor_e_fir.backend.model.ChargeSheetStatus;
import com.legal_advisor_e_fir.backend.model.FinalReportType;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface ChargeSheetRepo extends JpaRepository<ChargeSheet, Long> {

    Optional<ChargeSheet> findByChargeSheetNumber(String chargeSheetNumber);

    boolean existsByChargeSheetNumber(String chargeSheetNumber);

    List<ChargeSheet> findByFir_FirId(Long firId);

    Optional<ChargeSheet> findFirstByFir_FirIdOrderByCreatedAtDesc(Long firId);

    List<ChargeSheet> findByPoliceStation_StationId(Long stationId);

    List<ChargeSheet> findByStatus(ChargeSheetStatus status);

    List<ChargeSheet> findByPoliceStation_StationIdAndStatus(Long stationId, ChargeSheetStatus status);

    List<ChargeSheet> findByInvestigatingOfficer_PoliceId(Long policeId);

    List<ChargeSheet> findByInvestigatingOfficer_PoliceIdAndStatus(Long policeId, ChargeSheetStatus status);

    List<ChargeSheet> findByApprovingOfficer_PoliceId(Long policeId);

    List<ChargeSheet> findByReportType(FinalReportType reportType);

    List<ChargeSheet> findByPoliceStation_Subdivision_SubdivisionId(Long subdivisionId);

    long countByStatus(ChargeSheetStatus status);

    long countByPoliceStation_StationId(Long stationId);

    long countByInvestigatingOfficer_PoliceId(Long policeId);
}
