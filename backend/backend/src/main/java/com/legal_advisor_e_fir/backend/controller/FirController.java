package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.FirRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirResponseDto;
import com.legal_advisor_e_fir.backend.model.fir_status;
import com.legal_advisor_e_fir.backend.service.IFirService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/fir")
public class FirController {

    private final IFirService firService;

    public FirController(@Autowired IFirService firService) {
        this.firService = firService;
    }

    @PostMapping("/register")
    public ResponseEntity<FirResponseDto> createFir(
            @Valid @RequestBody FirRequestDto request) {

        FirResponseDto response = firService.createFir(request);
        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<FirResponseDto> getFirById(
            @PathVariable Long id) {

        FirResponseDto response = firService.getFirResponseById(id);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/number/{firNumber}")
    public ResponseEntity<FirResponseDto> getFirByFirNumber(
            @PathVariable String firNumber) {

        FirResponseDto response = firService.getFirByFirNumber(firNumber);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<List<FirResponseDto>> getAllFirs() {

        List<FirResponseDto> firs = firService.getAllFirs();
        return ResponseEntity.ok(firs);
    }

    @PutMapping("/{id}/assign-officer")
    public ResponseEntity<FirResponseDto> assignInvestigatingOfficer(
            @PathVariable Long id,
            @RequestParam Long policeId) {

        FirResponseDto response = firService.assignInvestigatingOfficer(id, policeId);
        return ResponseEntity.ok(response);
    }

    @PutMapping("/{id}/status")
    public ResponseEntity<FirResponseDto> updateFirStatus(
            @PathVariable Long id,
            @RequestParam fir_status status) {

        FirResponseDto response = firService.updateFirStatus(id, status);
        return ResponseEntity.ok(response);
    }

    @PutMapping("/{id}/ipc-sections")
    public ResponseEntity<FirResponseDto> updateIpcSections(
            @PathVariable Long id,
            @RequestParam String ipcSections) {

        FirResponseDto response = firService.updateIpcSections(id, ipcSections);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/station/{stationId}")
    public ResponseEntity<List<FirResponseDto>> getFirsByStation(
            @PathVariable Long stationId) {

        List<FirResponseDto> firs = firService.getFirsByStation(stationId);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/station/{stationId}/status/{status}")
    public ResponseEntity<List<FirResponseDto>> getFirsByStationAndStatus(
            @PathVariable Long stationId,
            @PathVariable fir_status status) {

        List<FirResponseDto> firs = firService.getFirsByStationAndStatus(stationId, status);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/station/{stationId}/unassigned")
    public ResponseEntity<List<FirResponseDto>> getUnassignedFirsByStation(
            @PathVariable Long stationId) {

        List<FirResponseDto> firs = firService.getUnassignedFirsByStation(stationId);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/officer/{policeId}")
    public ResponseEntity<List<FirResponseDto>> getFirsByInvestigatingOfficer(
            @PathVariable Long policeId) {

        List<FirResponseDto> firs = firService.getFirsByInvestigatingOfficer(policeId);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/officer/{policeId}/status/{status}")
    public ResponseEntity<List<FirResponseDto>> getFirsByOfficerAndStatus(
            @PathVariable Long policeId,
            @PathVariable fir_status status) {

        List<FirResponseDto> firs = firService.getFirsByInvestigatingOfficerAndStatus(policeId, status);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/status/{status}")
    public ResponseEntity<List<FirResponseDto>> getFirsByStatus(
            @PathVariable fir_status status) {

        List<FirResponseDto> firs = firService.getFirsByStatus(status);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/district/{district}")
    public ResponseEntity<List<FirResponseDto>> getFirsByDistrict(
            @PathVariable String district) {

        List<FirResponseDto> firs = firService.getFirsByDistrict(district);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/district/{district}/status/{status}")
    public ResponseEntity<List<FirResponseDto>> getFirsByDistrictAndStatus(
            @PathVariable String district,
            @PathVariable fir_status status) {

        List<FirResponseDto> firs = firService.getFirsByDistrictAndStatus(district, status);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/crime-category/{crimeCategory}")
    public ResponseEntity<List<FirResponseDto>> getFirsByCrimeCategory(
            @PathVariable String crimeCategory) {

        List<FirResponseDto> firs = firService.getFirsByCrimeCategory(crimeCategory);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/informant/contact/{contact}")
    public ResponseEntity<List<FirResponseDto>> getFirsByInformantContact(
            @PathVariable String contact) {

        List<FirResponseDto> firs = firService.getFirsByInformantContact(contact);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/informant/email/{email}")
    public ResponseEntity<List<FirResponseDto>> getFirsByInformantEmail(
            @PathVariable String email) {

        List<FirResponseDto> firs = firService.getFirsByInformantEmail(email);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/complaint/{complaintId}")
    public ResponseEntity<FirResponseDto> getFirByComplaintId(
            @PathVariable Long complaintId) {

        FirResponseDto response = firService.getFirByComplaintId(complaintId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/search/by-name")
    public ResponseEntity<List<FirResponseDto>> searchFirsByInformantName(
            @RequestParam String name) {

        List<FirResponseDto> firs = firService.searchFirsByInformantName(name);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/search/by-description")
    public ResponseEntity<List<FirResponseDto>> searchFirsByDescription(
            @RequestParam String keyword) {

        List<FirResponseDto> firs = firService.searchFirsByDescription(keyword);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/registration-date-range")
    public ResponseEntity<List<FirResponseDto>> getFirsByRegistrationDateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime endDate) {

        List<FirResponseDto> firs = firService.getFirsByRegistrationDateRange(startDate, endDate);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/incident-date-range")
    public ResponseEntity<List<FirResponseDto>> getFirsByIncidentDateRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate) {

        List<FirResponseDto> firs = firService.getFirsByIncidentDateRange(startDate, endDate);
        return ResponseEntity.ok(firs);
    }

    @GetMapping("/count/status/{status}")
    public ResponseEntity<Long> countFirsByStatus(
            @PathVariable fir_status status) {

        long count = firService.countFirsByStatus(status);
        return ResponseEntity.ok(count);
    }

    @GetMapping("/count/station/{stationId}")
    public ResponseEntity<Long> countFirsByStation(
            @PathVariable Long stationId) {

        long count = firService.countFirsByStation(stationId);
        return ResponseEntity.ok(count);
    }

    @GetMapping("/count/officer/{policeId}")
    public ResponseEntity<Long> countFirsByOfficer(
            @PathVariable Long policeId) {

        long count = firService.countFirsByInvestigatingOfficer(policeId);
        return ResponseEntity.ok(count);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteFir(
            @PathVariable Long id) {

        firService.deleteFir(id);
        return ResponseEntity.noContent().build();
    }
}
