package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.ChargeSheetRequestDto;
import com.legal_advisor_e_fir.backend.dto.ChargeSheetResponseDto;
import com.legal_advisor_e_fir.backend.service.IChargeSheetService;
import jakarta.validation.Valid;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/charge-sheet")
public class ChargeSheetController {

    private static final Logger log = LoggerFactory.getLogger(ChargeSheetController.class);

    private final IChargeSheetService chargeSheetService;

    public ChargeSheetController(@Autowired IChargeSheetService chargeSheetService) {
        this.chargeSheetService = chargeSheetService;
    }

    // ────────────────── IO (PSI) ENDPOINTS ──────────────────

    /** IO creates a new charge sheet / closure report (status = DRAFT) */
    @PostMapping("/create")
    public ResponseEntity<ChargeSheetResponseDto> create(
            @Valid @RequestBody ChargeSheetRequestDto request) {
        log.info("Creating charge sheet: csNumber={}, firId={}, stationId={}, ioId={}, reportType={}",
                request.getChargeSheetNumber(), request.getFirId(), request.getPoliceStationId(),
                request.getInvestigatingOfficerId(), request.getReportType());
        ChargeSheetResponseDto response = chargeSheetService.createChargeSheet(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /** IO updates a DRAFT or RETURNED charge sheet */
    @PutMapping("/{id}")
    public ResponseEntity<ChargeSheetResponseDto> update(
            @PathVariable Long id,
            @Valid @RequestBody ChargeSheetRequestDto request) {
        ChargeSheetResponseDto response = chargeSheetService.updateChargeSheet(id, request);
        return ResponseEntity.ok(response);
    }

    /** IO submits charge sheet to PI for approval */
    @PutMapping("/{id}/submit")
    public ResponseEntity<ChargeSheetResponseDto> submitToPI(@PathVariable Long id) {
        ChargeSheetResponseDto response = chargeSheetService.submitToPI(id);
        return ResponseEntity.ok(response);
    }

    // ────────────────── PI ENDPOINTS ──────────────────

    /** PI approves the charge sheet */
    @PutMapping("/{id}/approve")
    public ResponseEntity<ChargeSheetResponseDto> approve(
            @PathVariable Long id,
            @RequestParam Long approvingOfficerId) {
        ChargeSheetResponseDto response = chargeSheetService.approveChargeSheet(id, approvingOfficerId);
        return ResponseEntity.ok(response);
    }

    /** PI returns charge sheet to IO with suggestions */
    @PutMapping("/{id}/return")
    public ResponseEntity<ChargeSheetResponseDto> returnForRevision(
            @PathVariable Long id,
            @RequestBody Map<String, Object> body) {
        Long approvingOfficerId = Long.valueOf(body.get("approvingOfficerId").toString());
        String suggestions = (String) body.get("suggestions");
        ChargeSheetResponseDto response = chargeSheetService.returnForRevision(id, approvingOfficerId, suggestions);
        return ResponseEntity.ok(response);
    }

    /** PI dispatches approved charge sheet to court */
    @PutMapping("/{id}/dispatch")
    public ResponseEntity<ChargeSheetResponseDto> dispatchToCourt(@PathVariable Long id) {
        ChargeSheetResponseDto response = chargeSheetService.dispatchToCourt(id);
        return ResponseEntity.ok(response);
    }

    // ────────────────── QUERY ENDPOINTS ──────────────────

    @GetMapping("/{id}")
    public ResponseEntity<ChargeSheetResponseDto> getById(@PathVariable Long id) {
        return ResponseEntity.ok(chargeSheetService.getById(id));
    }

    @GetMapping("/number/{chargeSheetNumber}")
    public ResponseEntity<ChargeSheetResponseDto> getByNumber(@PathVariable String chargeSheetNumber) {
        return ResponseEntity.ok(chargeSheetService.getByChargeSheetNumber(chargeSheetNumber));
    }

    @GetMapping("/fir/{firId}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getByFir(@PathVariable Long firId) {
        return ResponseEntity.ok(chargeSheetService.getByFirId(firId));
    }

    @GetMapping("/station/{stationId}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getByStation(@PathVariable Long stationId) {
        return ResponseEntity.ok(chargeSheetService.getByStation(stationId));
    }

    @GetMapping("/station/{stationId}/status/{status}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getByStationAndStatus(
            @PathVariable Long stationId, @PathVariable String status) {
        return ResponseEntity.ok(chargeSheetService.getByStationAndStatus(stationId, status));
    }

    @GetMapping("/station/{stationId}/pending")
    public ResponseEntity<List<ChargeSheetResponseDto>> getPendingByStation(@PathVariable Long stationId) {
        return ResponseEntity.ok(chargeSheetService.getPendingApprovalByStation(stationId));
    }

    @GetMapping("/officer/{policeId}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getByOfficer(@PathVariable Long policeId) {
        return ResponseEntity.ok(chargeSheetService.getByOfficer(policeId));
    }

    @GetMapping("/officer/{policeId}/status/{status}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getByOfficerAndStatus(
            @PathVariable Long policeId, @PathVariable String status) {
        return ResponseEntity.ok(chargeSheetService.getByOfficerAndStatus(policeId, status));
    }

    @GetMapping("/subdivision/{subdivisionId}")
    public ResponseEntity<List<ChargeSheetResponseDto>> getBySubdivision(@PathVariable Long subdivisionId) {
        return ResponseEntity.ok(chargeSheetService.getBySubdivision(subdivisionId));
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> delete(@PathVariable Long id) {
        chargeSheetService.deleteChargeSheet(id);
        return ResponseEntity.noContent().build();
    }
}
