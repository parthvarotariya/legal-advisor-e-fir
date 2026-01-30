package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.FirRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirResponseDto;
import com.legal_advisor_e_fir.backend.model.Fir;
import com.legal_advisor_e_fir.backend.model.fir_status;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

public interface IFirService {

    FirResponseDto createFir(FirRequestDto request);
    

    FirResponseDto assignInvestigatingOfficer(Long firId, Long policeId);
    
    FirResponseDto updateFirStatus(Long firId, fir_status status);
    
    FirResponseDto updateIpcSections(Long firId, String ipcSections);

    Fir getFirById(Long firId);
    
    FirResponseDto getFirResponseById(Long firId);
    
    FirResponseDto getFirByFirNumber(String firNumber);
    
    List<FirResponseDto> getAllFirs();

    List<FirResponseDto> getFirsByStation(Long stationId);
    
    List<FirResponseDto> getFirsByStationAndStatus(Long stationId, fir_status status);

    List<FirResponseDto> getFirsByInvestigatingOfficer(Long policeId);
    
    List<FirResponseDto> getFirsByInvestigatingOfficerAndStatus(Long policeId, fir_status status);
    
    List<FirResponseDto> getUnassignedFirsByStation(Long stationId);

    List<FirResponseDto> getFirsByStatus(fir_status status);

    List<FirResponseDto> getFirsByDistrict(String district);
    
    List<FirResponseDto> getFirsByDistrictAndStatus(String district, fir_status status);

    List<FirResponseDto> getFirsByCrimeCategory(String crimeCategory);

    List<FirResponseDto> getFirsByInformantContact(String contact);
    
    List<FirResponseDto> getFirsByInformantEmail(String email);
    
    FirResponseDto getFirByComplaintId(Long complaintId);

    List<FirResponseDto> getFirsByRegistrationDateRange(LocalDateTime startDate, LocalDateTime endDate);
    
    List<FirResponseDto> getFirsByIncidentDateRange(LocalDate startDate, LocalDate endDate);

    List<FirResponseDto> searchFirsByInformantName(String name);
    
    List<FirResponseDto> searchFirsByDescription(String keyword);

    long countFirsByStatus(fir_status status);
    
    long countFirsByStation(Long stationId);
    
    long countFirsByInvestigatingOfficer(Long policeId);

    void deleteFir(Long firId);
}
