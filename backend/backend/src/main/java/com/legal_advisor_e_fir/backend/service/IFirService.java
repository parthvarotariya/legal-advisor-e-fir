package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.FirFromReportRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirResponseDto;
import com.legal_advisor_e_fir.backend.model.Fir;
import com.legal_advisor_e_fir.backend.model.FirStatus;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

public interface IFirService {

    FirResponseDto createFir(FirRequestDto request);

    FirResponseDto createFirFromReport(FirFromReportRequestDto request);
    

    FirResponseDto assignInvestigatingOfficer(Long firId, Long policeId);
    
    FirResponseDto updateFirStatus(Long firId, FirStatus status);
    
    FirResponseDto updateIpcSections(Long firId, String ipcSections);

    Fir getFirById(Long firId);
    
    FirResponseDto getFirResponseById(Long firId);
    
    FirResponseDto getFirByFirNumber(String firNumber);
    
    List<FirResponseDto> getAllFirs();

    List<FirResponseDto> getFirsByStation(Long stationId);
    
    List<FirResponseDto> getFirsByStationAndStatus(Long stationId, FirStatus status);

    List<FirResponseDto> getFirsByInvestigatingOfficer(Long policeId);
    
    List<FirResponseDto> getFirsByInvestigatingOfficerAndStatus(Long policeId, FirStatus status);
    
    List<FirResponseDto> getUnassignedFirsByStation(Long stationId);

    List<FirResponseDto> getFirsByStatus(FirStatus status);

    List<FirResponseDto> getFirsByDistrict(String district);
    
    List<FirResponseDto> getFirsByDistrictAndStatus(String district, FirStatus status);

    List<FirResponseDto> getFirsByCrimeCategory(String crimeCategory);

    List<FirResponseDto> getFirsByInformantContact(String contact);
    
    List<FirResponseDto> getFirsByInformantEmail(String email);
    
    FirResponseDto getFirByComplaintId(Long complaintId);

    List<FirResponseDto> getFirsByRegistrationDateRange(LocalDateTime startDate, LocalDateTime endDate);
    
    List<FirResponseDto> getFirsByIncidentDateRange(LocalDate startDate, LocalDate endDate);

    List<FirResponseDto> searchFirsByInformantName(String name);
    
    List<FirResponseDto> searchFirsByDescription(String keyword);

    long countFirsByStatus(FirStatus status);
    
    long countFirsByStation(Long stationId);
    
    long countFirsByInvestigatingOfficer(Long policeId);

    void deleteFir(Long firId);

    List<FirResponseDto> getFirsBySubdivision(Long subdivisionId);
}
