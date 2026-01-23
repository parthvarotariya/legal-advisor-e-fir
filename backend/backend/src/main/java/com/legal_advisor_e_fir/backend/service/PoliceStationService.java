package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PoliceStationRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceStationResponseDto;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class PoliceStationService implements IPoliceStationService {
    
    private final PoliceStationRepository policeStationRepository;
    
    public PoliceStationService(@Autowired PoliceStationRepository policeStationRepository) {
        this.policeStationRepository = policeStationRepository;
    }
    
    @Override
    public PoliceStationResponseDto createPoliceStation(PoliceStationRequestDto request) {
        PoliceStation policeStation = new PoliceStation();
        policeStation.setStationCode(request.getStationCode());
        policeStation.setStationName(request.getStationName());
        policeStation.setAddress(request.getAddress());
        policeStation.setDistrict(request.getDistrict());
        policeStation.setState(request.getState());
        
        PoliceStation savedStation = policeStationRepository.save(policeStation);
        return mapToResponseDto(savedStation);
    }
    
    @Override
    public List<PoliceStationResponseDto> getAllPoliceStations() {
        return policeStationRepository.findAll()
                .stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }
    
    @Override
    public PoliceStationResponseDto getByStationCode(String stationCode) {
        PoliceStation policeStation = policeStationRepository.findByStationCode(stationCode)
                .orElseThrow(() -> new RuntimeException("Police station not found with code: " + stationCode));
        return mapToResponseDto(policeStation);
    }
    
    @Override
    public List<PoliceStationResponseDto> getByStationName(String stationName) {
        return policeStationRepository.findByStationName(stationName)
                .stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<PoliceStationResponseDto> getByDistrict(String district) {
        return policeStationRepository.findByDistrict(district)
                .stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }
    
    @Override
    public List<PoliceStationResponseDto> getByState(String state) {
        return policeStationRepository.findByState(state)
                .stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }
    
    @Override
    public PoliceStationResponseDto getById(Long id) {
        PoliceStation policeStation = policeStationRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Police station not found with id: " + id));
        return mapToResponseDto(policeStation);
    }
    
    private PoliceStationResponseDto mapToResponseDto(PoliceStation policeStation) {
        PoliceStationResponseDto dto = new PoliceStationResponseDto();
        dto.setStationId(policeStation.getStationId());
        dto.setStationCode(policeStation.getStationCode());
        dto.setStationName(policeStation.getStationName());
        dto.setAddress(policeStation.getAddress());
        dto.setDistrict(policeStation.getDistrict());
        dto.setState(policeStation.getState());
        return dto;
    }
}
