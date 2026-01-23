package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.PoliceStationRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceStationResponseDto;
import com.legal_advisor_e_fir.backend.service.IPoliceStationService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/police-stations")
public class PoliceStationController {
    
    private final IPoliceStationService policeStationService;
    
    public PoliceStationController(@Autowired IPoliceStationService policeStationService) {
        this.policeStationService = policeStationService;
    }
    
    @PostMapping("/register")
    public ResponseEntity<PoliceStationResponseDto> register(
            @Valid @RequestBody PoliceStationRequestDto request) {
        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(policeStationService.createPoliceStation(request));
    }
    
    @GetMapping
    public ResponseEntity<List<PoliceStationResponseDto>> getAllPoliceStations() {
        List<PoliceStationResponseDto> policeStations = policeStationService.getAllPoliceStations();
        return ResponseEntity.ok(policeStations);
    }
    
    @GetMapping("/{id}")
    public ResponseEntity<PoliceStationResponseDto> getById(@PathVariable Long id) {
        PoliceStationResponseDto policeStation = policeStationService.getById(id);
        return ResponseEntity.ok(policeStation);
    }
    
    @GetMapping("/code/{stationCode}")
    public ResponseEntity<PoliceStationResponseDto> getByStationCode(@PathVariable String stationCode) {
        PoliceStationResponseDto policeStation = policeStationService.getByStationCode(stationCode);
        return ResponseEntity.ok(policeStation);
    }
    
    @GetMapping("/name/{stationName}")
    public ResponseEntity<List<PoliceStationResponseDto>> getByStationName(@PathVariable String stationName) {
        List<PoliceStationResponseDto> policeStations = policeStationService.getByStationName(stationName);
        return ResponseEntity.ok(policeStations);
    }
    
    @GetMapping("/district/{district}")
    public ResponseEntity<List<PoliceStationResponseDto>> getByDistrict(@PathVariable String district) {
        List<PoliceStationResponseDto> policeStations = policeStationService.getByDistrict(district);
        return ResponseEntity.ok(policeStations);
    }
    
    @GetMapping("/state/{state}")
    public ResponseEntity<List<PoliceStationResponseDto>> getByState(@PathVariable String state) {
        List<PoliceStationResponseDto> policeStations = policeStationService.getByState(state);
        return ResponseEntity.ok(policeStations);
    }
}
