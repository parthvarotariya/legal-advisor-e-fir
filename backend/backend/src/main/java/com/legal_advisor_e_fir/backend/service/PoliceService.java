package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceResponseDto;
import com.legal_advisor_e_fir.backend.dto.PoliceUpdateRequestDto;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.Role;
import com.legal_advisor_e_fir.backend.repository.PoliceRepository;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

@Service
public class PoliceService implements IPoliceService {

    private final PoliceRepository policeRepository;
    private final PoliceStationRepository policeStationRepository;

    public PoliceService(PoliceRepository policeRepository, 
                        PoliceStationRepository policeStationRepository) {
        this.policeRepository = policeRepository;
        this.policeStationRepository = policeStationRepository;
    }

    @Override
    public PoliceResponseDto registerPolice(PoliceRequestDto request) {
        
        // Validate unique constraints
        if (policeRepository.existsByEmail(request.getEmail())) {
            throw new RuntimeException("Email already registered: " + request.getEmail());
        }
        
        if (policeRepository.existsByBadgeNumber(request.getBadgeNumber())) {
            throw new RuntimeException("Badge number already exists: " + request.getBadgeNumber());
        }
        
        // Fetch and validate police station
        PoliceStation policeStation = policeStationRepository.findById(request.getStationId())
                .orElseThrow(() -> new RuntimeException("Police station not found with id: " + request.getStationId()));
        
        // Create police entity
        Police police = mapToEntity(request, policeStation);
        Police savedPolice = policeRepository.save(police);
        
        return mapToResponse(savedPolice);
    }

    @Override
    public Police getPoliceById(Long id) {
        return policeRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Police not found with id: " + id));
    }

    @Override
    public List<PoliceResponseDto> getAllPolice() {
        return policeRepository.findAll()
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
                    && policeRepository.existsByEmail(request.getEmail())) {
                throw new RuntimeException("Email already registered: " + request.getEmail());
            }
            existingPolice.setEmail(request.getEmail());
        }
        
        // Update badge number if provided and validate uniqueness
        if (request.getBadgeNumber() != null && !request.getBadgeNumber().isEmpty()) {
            if (!existingPolice.getBadgeNumber().equals(request.getBadgeNumber()) 
                    && policeRepository.existsByBadgeNumber(request.getBadgeNumber())) {
                throw new RuntimeException("Badge number already exists: " + request.getBadgeNumber());
            }
            existingPolice.setBadgeNumber(request.getBadgeNumber());
        }
        
        // Update police station if provided
        if (request.getStationId() != null && !existingPolice.getPoliceStation().getStationId().equals(request.getStationId())) {
            PoliceStation policeStation = policeStationRepository.findById(request.getStationId())
                    .orElseThrow(() -> new RuntimeException("Police station not found with id: " + request.getStationId()));
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
        
        Police updatedPolice = policeRepository.save(existingPolice);
        return mapToResponse(updatedPolice);
    }

    @Override
    public void deletePoliceById(Long id) {
        if (!policeRepository.existsById(id)) {
            throw new RuntimeException("Police not found with id: " + id);
        }
        policeRepository.deleteById(id);
    }

    @Override
    public PoliceResponseDto getPoliceByEmail(String email) {
        Police police = policeRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("Police not found with email: " + email));
        return mapToResponse(police);
    }

    @Override
    public PoliceResponseDto getPoliceByBadgeNumber(String badgeNumber) {
        Police police = policeRepository.findByBadgeNumber(badgeNumber)
                .orElseThrow(() -> new RuntimeException("Police not found with badge number: " + badgeNumber));
        return mapToResponse(police);
    }

    @Override
    public List<PoliceResponseDto> getPoliceByStationId(Long stationId) {
        // Validate station exists
        if (!policeStationRepository.existsById(stationId)) {
            throw new RuntimeException("Police station not found with id: " + stationId);
        }
        
        return policeRepository.findByPoliceStationStationId(stationId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<PoliceResponseDto> getPoliceByRole(Role role) {
        return policeRepository.findByRole(role)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    // Private helper methods for mapping
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
