package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ChargeSheetRequestDto;
import com.legal_advisor_e_fir.backend.dto.ChargeSheetResponseDto;

import java.util.List;

public interface IChargeSheetService {

    // IO (PSI) actions
    ChargeSheetResponseDto createChargeSheet(ChargeSheetRequestDto request);
    ChargeSheetResponseDto updateChargeSheet(Long chargeSheetId, ChargeSheetRequestDto request);
    ChargeSheetResponseDto submitToPI(Long chargeSheetId);

    // PI actions
    ChargeSheetResponseDto approveChargeSheet(Long chargeSheetId, Long approvingOfficerId);
    ChargeSheetResponseDto returnForRevision(Long chargeSheetId, Long approvingOfficerId, String suggestions);
    ChargeSheetResponseDto dispatchToCourt(Long chargeSheetId);

    // Queries
    ChargeSheetResponseDto getById(Long chargeSheetId);
    ChargeSheetResponseDto getByChargeSheetNumber(String chargeSheetNumber);
    List<ChargeSheetResponseDto> getByFirId(Long firId);
    List<ChargeSheetResponseDto> getByStation(Long stationId);
    List<ChargeSheetResponseDto> getByStationAndStatus(Long stationId, String status);
    List<ChargeSheetResponseDto> getByOfficer(Long policeId);
    List<ChargeSheetResponseDto> getByOfficerAndStatus(Long policeId, String status);
    List<ChargeSheetResponseDto> getBySubdivision(Long subdivisionId);
    List<ChargeSheetResponseDto> getPendingApprovalByStation(Long stationId);
    void deleteChargeSheet(Long chargeSheetId);
}
