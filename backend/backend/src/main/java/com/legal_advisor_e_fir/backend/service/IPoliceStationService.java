package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PoliceStationRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceStationResponseDto;

import java.util.List;

public interface IPoliceStationService {
    
    PoliceStationResponseDto createPoliceStation(PoliceStationRequestDto request);
    
    List<PoliceStationResponseDto> getAllPoliceStations();
    
    PoliceStationResponseDto getByStationCode(String stationCode);
    
    List<PoliceStationResponseDto> getByStationName(String stationName);
    
    List<PoliceStationResponseDto> getByDistrict(String district);
    
    List<PoliceStationResponseDto> getByState(String state);
    
    PoliceStationResponseDto getById(Long id);
}
