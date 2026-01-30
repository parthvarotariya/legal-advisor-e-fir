package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceResponseDto;
import com.legal_advisor_e_fir.backend.dto.PoliceUpdateRequestDto;
import com.legal_advisor_e_fir.backend.exceptions.DuplicateResourceException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.Role;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class PoliceService implements IPoliceService {

    private final PoliceRepo policeRepo;
    private final PoliceStationRepo policeStationRepo;

    public PoliceService(PoliceRepo policeRepo, 
                        PoliceStationRepo policeStationRepo) {
        this.policeRepo = policeRepo;
        this.policeStationRepo = policeStationRepo;
    }


    @Override
    public Police getPoliceById(Long id) {
        return policeRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Police not found with id: " + id));
    }

    @Override
    public List<PoliceResponseDto> getAllPolice() {
        return policeRepo.findAll()
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public PoliceResponseDto getPoliceResponseById(Long id) {
        Police police = getPoliceById(id);
        return mapToResponse(police);
    }

    @Override
    public PoliceResponseDto updatePolice(Long id, PoliceUpdateRequestDto request) {
        
        Police existingPolice = getPoliceById(id);
        
        // Update email if provided and validate uniqueness
        if (request.getEmail() != null && !request.getEmail().isEmpty()) {
            if (!existingPolice.getEmail().equals(request.getEmail()) 
                    && policeRepo.existsByEmail(request.getEmail())) {
                throw new DuplicateResourceException("Email already registered: " + request.getEmail());
            }
            existingPolice.setEmail(request.getEmail());
        }
        
        // Update badge number if provided and validate uniqueness
        if (request.getBadgeNumber() != null && !request.getBadgeNumber().isEmpty()) {
            if (!existingPolice.getBadgeNumber().equals(request.getBadgeNumber()) 
                    && policeRepo.existsByBadgeNumber(request.getBadgeNumber())) {
                throw new DuplicateResourceException("Badge number already exists: " + request.getBadgeNumber());
            }
            existingPolice.setBadgeNumber(request.getBadgeNumber());
        }
        
        // Update police station if provided
        if (request.getStationId() != null && !existingPolice.getPoliceStation().getStationId().equals(request.getStationId())) {
            PoliceStation policeStation = policeStationRepo.findById(request.getStationId())
                    .orElseThrow(() -> new ResourceNotFoundException("Police station not found with id: " + request.getStationId()));
            existingPolice.setPoliceStation(policeStation);
        }
        
        // Update other fields if provided
        if (request.getName() != null && !request.getName().isEmpty()) {
            existingPolice.setName(request.getName());
        }
        
        if (request.getRank() != null && !request.getRank().isEmpty()) {
            existingPolice.setRank(request.getRank());
        }
        
        if (request.getMobileNumber() != null && !request.getMobileNumber().isEmpty()) {
            existingPolice.setMobileNumber(request.getMobileNumber());
        }
        
        if (request.getRole() != null) {
            existingPolice.setRole(request.getRole());
        }
        
        // Update password if provided
        if (request.getPassword() != null && !request.getPassword().isEmpty()) {
            existingPolice.setPassword(request.getPassword()); // Should be hashed in production
        }
        
        Police updatedPolice = policeRepo.save(existingPolice);
        return mapToResponse(updatedPolice);
    }

    @Override
    public void deletePoliceById(Long id) {
        if (!policeRepo.existsById(id)) {
            throw new ResourceNotFoundException("Police not found with id: " + id);
        }
        policeRepo.deleteById(id);
    }

    @Override
    public PoliceResponseDto getPoliceByEmail(String email) {
        Police police = policeRepo.findByEmail(email)
                .orElseThrow(() -> new ResourceNotFoundException("Police not found with email: " + email));
        return mapToResponse(police);
    }

    @Override
    public PoliceResponseDto getPoliceByBadgeNumber(String badgeNumber) {
        Police police = policeRepo.findByBadgeNumber(badgeNumber)
                .orElseThrow(() -> new ResourceNotFoundException("Police not found with badge number: " + badgeNumber));
        return mapToResponse(police);
    }

    @Override
    public List<PoliceResponseDto> getPoliceByStationId(Long stationId) {
        // Validate station exists
        if (!policeStationRepo.existsById(stationId)) {
            throw new ResourceNotFoundException("Police station not found with id: " + stationId);
        }
        
        return policeRepo.findByPoliceStation_StationId(stationId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<PoliceResponseDto> getPoliceByRole(Role role) {
        return policeRepo.findByRole(role)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    private PoliceResponseDto mapToResponse(Police police) {
        PoliceResponseDto response = new PoliceResponseDto();
        response.setPoliceId(police.getPoliceId());
        response.setName(police.getName());
        response.setBadgeNumber(police.getBadgeNumber());
        response.setRank(police.getRank());
        response.setEmail(police.getEmail());
        response.setMobileNumber(police.getMobileNumber());
        response.setRole(police.getRole());
        response.setStationId(police.getPoliceStation().getStationId());
        response.setStationName(police.getPoliceStation().getStationName());
        response.setStationCode(police.getPoliceStation().getStationCode());
        response.setCreatedAt(police.getCreatedAt());
        response.setUpdatedAt(police.getUpdatedAt());
        return response;
    }

    private Police mapToEntity(PoliceRequestDto request, PoliceStation policeStation) {
        Police police = new Police();
        police.setName(request.getName());
        police.setBadgeNumber(request.getBadgeNumber());
        police.setRank(request.getRank());
        police.setEmail(request.getEmail());
        police.setMobileNumber(request.getMobileNumber());
        police.setPassword(request.getPassword()); // Should be hashed in production
        police.setRole(request.getRole());
        police.setPoliceStation(policeStation);
        return police;
    }
}
