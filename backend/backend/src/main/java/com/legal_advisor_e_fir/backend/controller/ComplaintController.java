package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.service.IComplaintService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/complaints")
public class ComplaintController {
    private final IComplaintService complaintService;

    public ComplaintController(@Autowired IComplaintService complaintService)
    {
        this.complaintService = complaintService;
    }

    @PostMapping("/register")
    public ResponseEntity<ComplaintResponseDto> register(
            @Valid @RequestBody ComplaintRequestDto request) {

        return ResponseEntity
                .status(HttpStatus.CREATED)
                .body(complaintService.createComplaint(request));
    }

    @GetMapping("/{id}")
    public ResponseEntity<ComplaintResponseDto> getById(@PathVariable Long id) {
        ComplaintResponseDto complaint = complaintService.getById(id);
        return ResponseEntity.ok(complaint);
    }


    @GetMapping("/users/{id}")
    public ResponseEntity<List<ComplaintResponseDto>> userComplaints(@PathVariable Long id) {
        List<ComplaintResponseDto> complaints = complaintService.getByUser(id);
        return ResponseEntity.ok(complaints);
    }


    @GetMapping("/station/{stationId}")
    public ResponseEntity<?> stationComplaints(@PathVariable Long stationId) {
        try {
            List<ComplaintResponseDto> complaints = complaintService.getByPoliceStation(stationId);
            return ResponseEntity.ok(complaints);
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Map.of("error", e.getClass().getSimpleName(), 
                                 "message", e.getMessage() != null ? e.getMessage() : "Unknown error",
                                 "cause", e.getCause() != null ? e.getCause().getMessage() : "No cause"));
        }
    }

    @GetMapping("/officer/{officerId}")
    public ResponseEntity<List<ComplaintResponseDto>> officerComplaints(@PathVariable Long officerId) {
        List<ComplaintResponseDto> complaints = complaintService.getByAssignedOfficer(officerId);
        return ResponseEntity.ok(complaints);
    }

    @GetMapping("/subdivision/{subdivisionId}")
    public ResponseEntity<List<ComplaintResponseDto>> subdivisionComplaints(@PathVariable Long subdivisionId) {
        List<ComplaintResponseDto> complaints = complaintService.getBySubdivision(subdivisionId);
        return ResponseEntity.ok(complaints);
    }

    @GetMapping("/subdivision/{subdivisionId}/status/{status}")
    public ResponseEntity<List<ComplaintResponseDto>> subdivisionComplaintsByStatus(
            @PathVariable Long subdivisionId,
            @PathVariable com.legal_advisor_e_fir.backend.model.ComplaintStatus status) {
        List<ComplaintResponseDto> complaints = complaintService.getBySubdivisionAndStatus(subdivisionId, status);
        return ResponseEntity.ok(complaints);
    }
//.
    @PutMapping("/{id}/status")
    public ResponseEntity<ComplaintResponseDto> updateComplaint(
            @PathVariable Long id,
            @RequestBody java.util.Map<String, Object> request) {
        
        String statusStr = (String) request.get("status");
        String actualCategory = (String) request.get("actualCategory");
        Object officerIdObj = request.get("officerId");
        
        Long officerId = null;
        if (officerIdObj != null) {
            if (officerIdObj instanceof Integer) {
                officerId = ((Integer) officerIdObj).longValue();
            } else if (officerIdObj instanceof Long) {
                officerId = (Long) officerIdObj;
            }
        }
        
        com.legal_advisor_e_fir.backend.model.ComplaintStatus status = 
            com.legal_advisor_e_fir.backend.model.ComplaintStatus.valueOf(statusStr);
        
        ComplaintResponseDto updated = complaintService.updateComplaint(id, status, actualCategory, officerId);
        return ResponseEntity.ok(updated);
    }
}
