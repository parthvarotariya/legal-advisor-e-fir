package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.PreliminaryReportRequestDto;
import com.legal_advisor_e_fir.backend.dto.PreliminaryReportResponseDto;
import com.legal_advisor_e_fir.backend.service.IPreliminaryReportService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/preliminary-report")
public class PreliminaryReportController {

    private final IPreliminaryReportService preliminaryReportService;

    public PreliminaryReportController(@Autowired IPreliminaryReportService preliminaryReportService) {
        this.preliminaryReportService = preliminaryReportService;
    }

    @PostMapping("/create")
    public ResponseEntity<PreliminaryReportResponseDto> createPreliminaryReport(
            @Valid @RequestBody PreliminaryReportRequestDto request) {

        PreliminaryReportResponseDto response = preliminaryReportService.createPreliminaryReport(request);
        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<PreliminaryReportResponseDto> getPreliminaryReportById(
            @PathVariable Long id) {

        PreliminaryReportResponseDto response = preliminaryReportService.getPreliminaryReportResponseById(id);
        return ResponseEntity.ok(response);
    }

    @GetMapping
    public ResponseEntity<List<PreliminaryReportResponseDto>> getAllPreliminaryReports() {

        List<PreliminaryReportResponseDto> reports = preliminaryReportService.getAllPreliminaryReports();
        return ResponseEntity.ok(reports);
    }

    @GetMapping("/complaint/{complaintId}")
    public ResponseEntity<PreliminaryReportResponseDto> getByComplaintId(
            @PathVariable Long complaintId) {

        PreliminaryReportResponseDto response = preliminaryReportService.findByComplaintId(complaintId);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/station/{stationId}")
    public ResponseEntity<List<PreliminaryReportResponseDto>> getByStationId(
            @PathVariable Long stationId) {

        List<PreliminaryReportResponseDto> reports = preliminaryReportService.findByStationId(stationId);
        return ResponseEntity.ok(reports);
    }

    @GetMapping("/officer/{policeId}")
    public ResponseEntity<List<PreliminaryReportResponseDto>> getByInvestigatingOfficerId(
            @PathVariable Long policeId) {

        List<PreliminaryReportResponseDto> reports = preliminaryReportService.findByInvestigatingOfficerId(policeId);
        return ResponseEntity.ok(reports);
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deletePreliminaryReport(
            @PathVariable Long id) {

        preliminaryReportService.deletePreliminaryReport(id);
        return ResponseEntity.noContent().build();
    }

    @GetMapping("/subdivision/{subdivisionId}")
    public ResponseEntity<List<PreliminaryReportResponseDto>> getBySubdivisionId(
            @PathVariable Long subdivisionId) {
        List<PreliminaryReportResponseDto> reports = preliminaryReportService.findBySubdivisionId(subdivisionId);
        return ResponseEntity.ok(reports);
    }
}
