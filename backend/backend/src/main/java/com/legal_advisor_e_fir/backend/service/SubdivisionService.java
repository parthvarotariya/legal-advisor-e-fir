package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.SubdivisionRequestDto;
import com.legal_advisor_e_fir.backend.dto.SubdivisionResponseDto;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.Subdivision;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import com.legal_advisor_e_fir.backend.repository.SubdivisionRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Service implementation for Subdivision management operations.
 * Handles business logic, validation, and data transformation.
 */
@Service
public class SubdivisionService implements ISubdivisionService {

    private final SubdivisionRepository subdivisionRepository;
    private final PoliceRepo policeRepo;
    private final PoliceStationRepo policeStationRepo;

    @Autowired
    public SubdivisionService(
            SubdivisionRepository subdivisionRepository,
            PoliceRepo policeRepo,
            PoliceStationRepo policeStationRepo
    ) {
        this.subdivisionRepository = subdivisionRepository;
        this.policeRepo = policeRepo;
        this.policeStationRepo = policeStationRepo;
    }

    @Override
    @Transactional
    public SubdivisionResponseDto createSubdivision(SubdivisionRequestDto requestDto) {
        // Validate subdivision code uniqueness
        if (subdivisionRepository.existsBySubdivisionCode(requestDto.getSubdivisionCode())) {
            throw new RuntimeException("Subdivision with code " + requestDto.getSubdivisionCode() + " already exists");
        }

        Subdivision subdivision = new Subdivision();
        subdivision.setSubdivisionCode(requestDto.getSubdivisionCode());
        subdivision.setSubdivisionName(requestDto.getSubdivisionName());
        subdivision.setDistrict(requestDto.getDistrict());
        subdivision.setState(requestDto.getState());

        // Assign DSP officer if provided
        if (requestDto.getDspOfficerId() != null) {
            Police dspOfficer = policeRepo.findById(requestDto.getDspOfficerId())
                    .orElseThrow(() -> new RuntimeException("DSP Officer not found with ID: " + requestDto.getDspOfficerId()));

            // Validate officer rank or role
            boolean isDsp = "DEPUTY_SUPERINTENDENT".equalsIgnoreCase(dspOfficer.getRank()) ||
                    (dspOfficer.getRole() != null && dspOfficer.getRole().name().equals("DEPUTY_SUPRINTENDENT"));
            
            if (!isDsp) {
                throw new RuntimeException("Officer must have DEPUTY_SUPERINTENDENT rank or DEPUTY_SUPRINTENDENT role to be assigned as DSP");
            }

            // Check if DSP is already assigned to another subdivision
            subdivisionRepository.findByDspOfficer_PoliceId(requestDto.getDspOfficerId())
                    .ifPresent(existingSub -> {
                        throw new RuntimeException("DSP Officer is already assigned to subdivision: " + existingSub.getSubdivisionName());
                    });

            subdivision.setDspOfficer(dspOfficer);
        }

        Subdivision savedSubdivision = subdivisionRepository.save(subdivision);
        return mapToResponseDto(savedSubdivision);
    }

    @Override
    @Transactional(readOnly = true)
    public SubdivisionResponseDto getById(Long id) {
        Subdivision subdivision = subdivisionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + id));
        return mapToResponseDto(subdivision);
    }

    @Override
    @Transactional(readOnly = true)
    public SubdivisionResponseDto getByCode(String code) {
        Subdivision subdivision = subdivisionRepository.findBySubdivisionCode(code)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with code: " + code));
        return mapToResponseDto(subdivision);
    }

    @Override
    @Transactional(readOnly = true)
    public List<SubdivisionResponseDto> getAllSubdivisions() {
        return subdivisionRepository.findAll().stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional(readOnly = true)
    public List<SubdivisionResponseDto> getByDistrict(String district) {
        return subdivisionRepository.findByDistrict(district).stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional(readOnly = true)
    public List<SubdivisionResponseDto> getByState(String state) {
        return subdivisionRepository.findByState(state).stream()
                .map(this::mapToResponseDto)
                .collect(Collectors.toList());
    }

    @Override
    @Transactional
    public SubdivisionResponseDto updateSubdivision(Long id, SubdivisionRequestDto requestDto) {
        Subdivision subdivision = subdivisionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + id));

        // Check if code is being changed and validate uniqueness
        if (!subdivision.getSubdivisionCode().equals(requestDto.getSubdivisionCode())) {
            if (subdivisionRepository.existsBySubdivisionCode(requestDto.getSubdivisionCode())) {
                throw new RuntimeException("Subdivision with code " + requestDto.getSubdivisionCode() + " already exists");
            }
            subdivision.setSubdivisionCode(requestDto.getSubdivisionCode());
        }

        subdivision.setSubdivisionName(requestDto.getSubdivisionName());
        subdivision.setDistrict(requestDto.getDistrict());
        subdivision.setState(requestDto.getState());

        Subdivision updatedSubdivision = subdivisionRepository.save(subdivision);
        return mapToResponseDto(updatedSubdivision);
    }

    @Override
    @Transactional
    public SubdivisionResponseDto assignDspOfficer(Long subdivisionId, Long dspOfficerId) {
        Subdivision subdivision = subdivisionRepository.findById(subdivisionId)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + subdivisionId));

        Police dspOfficer = policeRepo.findById(dspOfficerId)
                .orElseThrow(() -> new RuntimeException("DSP Officer not found with ID: " + dspOfficerId));

        // Validate officer rank or role
        boolean isDsp = "DEPUTY_SUPERINTENDENT".equalsIgnoreCase(dspOfficer.getRank()) ||
                (dspOfficer.getRole() != null && dspOfficer.getRole().name().equals("DEPUTY_SUPRINTENDENT"));
        
        if (!isDsp) {
            throw new RuntimeException("Officer must have DEPUTY_SUPERINTENDENT rank or DEPUTY_SUPRINTENDENT role to be assigned as DSP. Current rank: " + dspOfficer.getRank() + ", role: " + dspOfficer.getRole());
        }

        // Check if DSP is already assigned to another subdivision
        subdivisionRepository.findByDspOfficer_PoliceId(dspOfficerId)
                .ifPresent(existingSub -> {
                    if (!existingSub.getSubdivisionId().equals(subdivisionId)) {
                        throw new RuntimeException("DSP Officer is already assigned to subdivision: " + existingSub.getSubdivisionName());
                    }
                });

        subdivision.setDspOfficer(dspOfficer);
        Subdivision updatedSubdivision = subdivisionRepository.save(subdivision);
        return mapToResponseDto(updatedSubdivision);
    }

    @Override
    @Transactional
    public SubdivisionResponseDto removeDspOfficer(Long subdivisionId) {
        Subdivision subdivision = subdivisionRepository.findById(subdivisionId)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + subdivisionId));

        subdivision.setDspOfficer(null);
        Subdivision updatedSubdivision = subdivisionRepository.save(subdivision);
        return mapToResponseDto(updatedSubdivision);
    }

    @Override
    @Transactional
    public SubdivisionResponseDto addPoliceStation(Long subdivisionId, Long policeStationId) {
        Subdivision subdivision = subdivisionRepository.findById(subdivisionId)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + subdivisionId));

        PoliceStation policeStation = policeStationRepo.findById(policeStationId)
                .orElseThrow(() -> new RuntimeException("Police Station not found with ID: " + policeStationId));

        // Check if station is already assigned to another subdivision
        if (policeStation.getSubdivision() != null && 
            !policeStation.getSubdivision().getSubdivisionId().equals(subdivisionId)) {
            throw new RuntimeException("Police Station is already assigned to subdivision: " + 
                    policeStation.getSubdivision().getSubdivisionName());
        }

        policeStation.setSubdivision(subdivision);
        policeStationRepo.save(policeStation);

        return mapToResponseDto(subdivision);
    }

    @Override
    @Transactional
    public SubdivisionResponseDto removePoliceStation(Long subdivisionId, Long policeStationId) {
        Subdivision subdivision = subdivisionRepository.findById(subdivisionId)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + subdivisionId));

        PoliceStation policeStation = policeStationRepo.findById(policeStationId)
                .orElseThrow(() -> new RuntimeException("Police Station not found with ID: " + policeStationId));

        // Verify that station belongs to this subdivision
        if (policeStation.getSubdivision() == null || 
            !policeStation.getSubdivision().getSubdivisionId().equals(subdivisionId)) {
            throw new RuntimeException("Police Station is not assigned to this subdivision");
        }

        policeStation.setSubdivision(null);
        policeStationRepo.save(policeStation);

        return mapToResponseDto(subdivision);
    }

    @Override
    @Transactional
    public void deleteSubdivision(Long id) {
        Subdivision subdivision = subdivisionRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Subdivision not found with ID: " + id));

        // Check if subdivision has assigned stations
        if (subdivision.getPoliceStations() != null && !subdivision.getPoliceStations().isEmpty()) {
            throw new RuntimeException("Cannot delete subdivision with assigned police stations. Remove all stations first.");
        }

        subdivisionRepository.delete(subdivision);
    }

    /**
     * Maps Subdivision entity to SubdivisionResponseDto.
     *
     * @param subdivision Subdivision entity
     * @return SubdivisionResponseDto
     */
    private SubdivisionResponseDto mapToResponseDto(Subdivision subdivision) {
        SubdivisionResponseDto responseDto = new SubdivisionResponseDto();
        responseDto.setSubdivisionId(subdivision.getSubdivisionId());
        responseDto.setSubdivisionCode(subdivision.getSubdivisionCode());
        responseDto.setSubdivisionName(subdivision.getSubdivisionName());
        responseDto.setDistrict(subdivision.getDistrict());
        responseDto.setState(subdivision.getState());
        responseDto.setCreatedAt(subdivision.getCreatedAt());
        responseDto.setUpdatedAt(subdivision.getUpdatedAt());

        // Map DSP officer if assigned
        if (subdivision.getDspOfficer() != null) {
            Police dspOfficer = subdivision.getDspOfficer();
            SubdivisionResponseDto.DspOfficerDto dspDto = new SubdivisionResponseDto.DspOfficerDto();
            dspDto.setPoliceId(dspOfficer.getPoliceId());
            dspDto.setName(dspOfficer.getName());
            dspDto.setBadgeNumber(dspOfficer.getBadgeNumber());
            dspDto.setRank(dspOfficer.getRank());
            dspDto.setEmail(dspOfficer.getEmail());
            dspDto.setMobileNumber(dspOfficer.getMobileNumber());
            responseDto.setDspOfficer(dspDto);
        }

        // Set station count
        int stationCount = subdivision.getPoliceStations() != null ? 
                subdivision.getPoliceStations().size() : 0;
        responseDto.setStationCount(stationCount);

        return responseDto;
    }
}
