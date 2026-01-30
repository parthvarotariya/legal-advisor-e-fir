package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.*;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

@Service
@Transactional
public class ComplaintService implements IComplaintService{
    private final IUserService userService;
    private final ComplaintRepo complaintRepo;
    private final PoliceStationRepo policeStationRepo;

    public ComplaintService(@Autowired ComplaintRepo complaintRepo,@Autowired IUserService userService,@Autowired PoliceStationRepo policeStationRepo)
    {
        this.complaintRepo = complaintRepo;
        this.userService = userService;
        this.policeStationRepo = policeStationRepo;
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
        User u =userService.getUserById(id);
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
                .orElseThrow(() ->new ResourceNotFoundException("PoliceStation not found with id " +id));
        List<ComplaintResponseDto> complaints = new ArrayList<>();
        List<Complaint> comps = complaintRepo.findAllByPoliceStation(ps);

        for (Complaint comp : comps) {
            complaints.add(mapToResponse(comp));
        }
        return complaints;
    }


    private Complaint mapToComplaint(ComplaintRequestDto request)
    {
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
        c.setStatus(complaint_status.PENDING);
        c.setPredictedCategory(request.getPredictedCategory());

        return c;
    }
    private ComplaintResponseDto mapToResponse(Complaint c)
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
