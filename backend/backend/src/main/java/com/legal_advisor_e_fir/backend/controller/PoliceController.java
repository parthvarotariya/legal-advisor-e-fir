package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceResponseDto;
import com.legal_advisor_e_fir.backend.dto.PoliceUpdateRequestDto;
import com.legal_advisor_e_fir.backend.model.Role;
import com.legal_advisor_e_fir.backend.service.IPoliceService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/police")
public class PoliceController {

    private final IPoliceService policeService;

    public PoliceController(@Autowired IPoliceService policeService) {
        this.policeService = policeService;
    }

    @PostMapping("/register")
    public ResponseEntity<PoliceResponseDto> register(
            @Valid @RequestBody PoliceRequestDto request) {

        PoliceResponseDto response = policeService.registerPolice(request);
        return new ResponseEntity<>(response, HttpStatus.CREATED);
    }

    @GetMapping("/{id}")
    public ResponseEntity<PoliceResponseDto> getPoliceById(
            @PathVariable Long id) {

        PoliceResponseDto response = policeService.getPoliceResponseById(id);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<List<PoliceResponseDto>> getAllPolice() {

        List<PoliceResponseDto> police = policeService.getAllPolice();
        return ResponseEntity.ok(police);
    }

    @PutMapping("/{id}")
    public ResponseEntity<PoliceResponseDto> updatePolice(
            @PathVariable Long id,
            @Valid @RequestBody PoliceUpdateRequestDto request) {

        PoliceResponseDto response = policeService.updatePolice(id, request);
        return ResponseEntity.ok(response);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deletePolice(
            @PathVariable Long id) {

        policeService.deletePoliceById(id);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/email/{email}")
    public ResponseEntity<PoliceResponseDto> getPoliceByEmail(
            @PathVariable String email) {

        PoliceResponseDto response = policeService.getPoliceByEmail(email);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/badge/{badgeNumber}")
    public ResponseEntity<PoliceResponseDto> getPoliceByBadgeNumber(
            @PathVariable String badgeNumber) {

        PoliceResponseDto response = policeService.getPoliceByBadgeNumber(badgeNumber);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/station/{stationId}")
    public ResponseEntity<List<PoliceResponseDto>> getPoliceByStationId(
            @PathVariable Long stationId) {

        List<PoliceResponseDto> police = policeService.getPoliceByStationId(stationId);
        return ResponseEntity.ok(police);
    }

    @GetMapping("/role/{role}")
    public ResponseEntity<List<PoliceResponseDto>> getPoliceByRole(
            @PathVariable Role role) {

        List<PoliceResponseDto> police = policeService.getPoliceByRole(role);
        return ResponseEntity.ok(police);
    }
}
