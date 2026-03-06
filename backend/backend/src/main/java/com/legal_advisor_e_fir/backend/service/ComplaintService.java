package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.*;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class ComplaintService implements IComplaintService {
    private final IUserService userService;
    private final ComplaintRepo complaintRepo;
    private final PoliceStationRepo policeStationRepo;
    private final PoliceRepo policeRepo;

    public ComplaintService(@Autowired ComplaintRepo complaintRepo, @Autowired IUserService userService, @Autowired PoliceStationRepo policeStationRepo, @Autowired PoliceRepo policeRepo) {
        this.complaintRepo = complaintRepo;
        this.userService = userService;
        this.policeStationRepo = policeStationRepo;
        this.policeRepo = policeRepo;
    }

    @Override
    public ComplaintResponseDto createComplaint(ComplaintRequestDto request) {
        Complaint comp = mapToComplaint(request);
        comp = complaintRepo.save(comp);
        return mapToResponse(comp);
    }

    @Override
    public ComplaintResponseDto getById(Long id) {
        Complaint complaint = complaintRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(
                        "Complaint not found with id: " + id
                ));
        return mapToResponse(complaint);
    }

    @Override
    public List<ComplaintResponseDto> getByUser(Long id) {
        User u = userService.getUserById(id);
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByUser(u);

        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }

        return complaints;
    }


    @Override
    public List<ComplaintResponseDto> getByPoliceStation(Long id) {
        PoliceStation ps = policeStationRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("PoliceStation not found with id " + id));
        
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByPoliceStation(ps);

        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }
        return complaints;
    }

    @Override
    public List<ComplaintResponseDto> getByAssignedOfficer(Long officerId) {
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByAssignedOfficer_PoliceId(officerId);
        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }
        return complaints;
    }

    @Override
    public List<ComplaintResponseDto> getBySubdivision(Long subdivisionId) {
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByPoliceStation_Subdivision_SubdivisionId(subdivisionId);
        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }
        return complaints;
    }

    @Override
    public List<ComplaintResponseDto> getBySubdivisionAndStatus(Long subdivisionId, ComplaintStatus status) {
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByPoliceStation_Subdivision_SubdivisionIdAndStatus(subdivisionId, status);
        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }
        return complaints;
    }


    private Complaint mapToComplaint(ComplaintRequestDto request) {
        User user = userService.getUserById(request.getUserId());

        PoliceStation policeStation = null;
        if (request.getPoliceStationId() != null) {
            policeStation = policeStationRepo.findById(request.getPoliceStationId())
                    .orElseThrow(() -> new ResourceNotFoundException(
                            "Police station not found with id: " + request.getPoliceStationId()));
        }

        Complaint c = new Complaint();
        c.setDescription(request.getDescription());
        c.setUser(user);
        c.setPoliceStation(policeStation);
        c.setStatus(ComplaintStatus.RECEIVED);
        c.setPredictedCategory(request.getPredictedCategory());

        return c;
    }

    private ComplaintResponseDto mapToResponse(Complaint c) {
        ComplaintResponseDto response = new ComplaintResponseDto();
        response.setId(c.getId());
        response.setDescription(c.getDescription());
        response.setStatus(c.getStatus());
        response.setActualCategory(c.getActualCategory());
        response.setPredictedCategory(c.getPredictedCategory());
        response.setCreatedAt(c.getCreatedAt());

        if (c.getPoliceStation() != null) {
            response.setPoliceStationId(c.getPoliceStation().getStationId());
            response.setPoliceStationName(c.getPoliceStation().getStationName());
        }

        if (c.getAssignedOfficer() != null) {
            response.setAssignedOfficerId(c.getAssignedOfficer().getPoliceId());
            response.setAssignedOfficerName(c.getAssignedOfficer().getName());
            response.setAssignedOfficerBadge(c.getAssignedOfficer().getBadgeNumber());
        }
        
        // Map complainant/user details
        if (c.getUser() != null) {
            response.setUserId(c.getUser().getId());
            response.setComplainantName(c.getUser().getName());
            response.setComplainantMobile(c.getUser().getMobileNumber());
            response.setComplainantEmail(c.getUser().getEmail());
            response.setComplainantAddress(c.getUser().getAddress());
        }

        return response;
    }

    @Override
    public ComplaintResponseDto updateComplaint(Long id, ComplaintStatus status, String actualCategory, Long officerId) {
        Complaint complaint = complaintRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Complaint not found"));

        // Update status
        complaint.setStatus(status);

        // Update actual category if provided
        if (actualCategory != null && !actualCategory.isEmpty()) {
            complaint.setActualCategory(actualCategory);
        }

        // Assign officer if provided
        if (officerId != null) {
            Police officer = policeRepo.findById(officerId)
                    .orElseThrow(() -> new ResourceNotFoundException("Officer not found"));
            complaint.setAssignedOfficer(officer);
        }

        Complaint updated = complaintRepo.save(complaint);
        return mapToResponse(updated);
    }
}
