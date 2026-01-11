package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.model.*;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
public class ComplaintService implements IComplaintService{
    private final IUserService userService;
    private final ComplaintRepo complaintRepo;

    public ComplaintService(@Autowired ComplaintRepo complaintRepo,@Autowired IUserService userService)
    {
        this.complaintRepo = complaintRepo;
        this.userService = userService;
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
                .orElseThrow(() -> new RuntimeException(
                        "Complaint not found with id: " + id
                ));
        return mapToResponse(complaint);
    }

    @Override
    public List<ComplaintResponseDto> getByUser(Long id) {
        User u =userService.getUserById(id);
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByUser(u);

        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }

        return complaints;
    }


    @Override
    public List<ComplaintResponseDto> getByPoliceStation(PoliceStation ps) {
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByPoliceStation(ps);

        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }

        return complaints;
    }


    Complaint mapToComplaint(ComplaintRequestDto request)
    {
        Complaint c = new Complaint();
        c.setDescription(request.getDescription());
        c.setUser(request.getUser());
        c.setPoliceStation(request.getPoliceStation());
        c.setStatus(complaint_status.PENDING);
        c.setPredictedCategory(request.getPredictedCategory());

        return c;
    }
    ComplaintResponseDto mapToResponse(Complaint c)
    {
        ComplaintResponseDto response = new ComplaintResponseDto();
        response.setId(c.getId());
        response.setDescription(c.getDescription());
        response.setStatus(c.getStatus());
        response.setActualCategory(c.getActualCategory());
        response.setPredictedCategory(c.getPredictedCategory());
        response.setCreatedAt(c.getCreatedAt());
        response.setPoliceStation(c.getPoliceStation());
        return response;
    }
}
