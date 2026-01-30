package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.service.IComplaintService;
import jakarta.validation.Valid;
import jdk.jfr.Frequency;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

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
    public  ResponseEntity<List<ComplaintResponseDto>> stationComplaints(@PathVariable Long stationId)
    {
        List<ComplaintResponseDto> complaints = complaintService.getByPoliceStation(stationId);
        return ResponseEntity.ok(complaints);
    }
}
