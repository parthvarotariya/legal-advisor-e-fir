package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.FirRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.DuplicateResourceException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.Complaint;
import com.legal_advisor_e_fir.backend.model.Fir;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.fir_status;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import com.legal_advisor_e_fir.backend.repository.FirRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class FirService implements IFirService {

    private final FirRepo firRepo;
    private final PoliceStationRepo policeStationRepo;
    private final PoliceRepo policeRepo;
    private final ComplaintRepo complaintRepo;

    public FirService(FirRepo firRepo,
                     PoliceStationRepo policeStationRepo,
                     PoliceRepo policeRepo,
                     ComplaintRepo complaintRepo) {
        this.firRepo = firRepo;
        this.policeStationRepo = policeStationRepo;
        this.policeRepo = policeRepo;
        this.complaintRepo = complaintRepo;
    }

    @Override
    public FirResponseDto createFir(FirRequestDto request) {
        if (firRepo.existsByFirNumber(request.getFirNumber())) {
            throw new DuplicateResourceException("FIR number already exists: " + request.getFirNumber());
        }

        PoliceStation policeStation = policeStationRepo.findById(request.getPoliceStationId())
                .orElseThrow(() -> new ResourceNotFoundException("Police station not found with id: " + request.getPoliceStationId()));

        Police investigatingOfficer = null;
        if (request.getInvestigatingOfficerId() != null) {
            investigatingOfficer = policeRepo.findById(request.getInvestigatingOfficerId())
                    .orElseThrow(() -> new ResourceNotFoundException("Police officer not found with id: " + request.getInvestigatingOfficerId()));
        }

        Complaint complaint = null;
        if (request.getComplaintId() != null) {
            complaint = complaintRepo.findById(request.getComplaintId())
                    .orElseThrow(() -> new ResourceNotFoundException("Complaint not found with id: " + request.getComplaintId()));
            
            if (firRepo.existsByComplaint_Id(request.getComplaintId())) {
                throw new DuplicateResourceException("Complaint already has a FIR associated with it");
            }
        }

        Fir fir = mapToEntity(request, policeStation, investigatingOfficer, complaint);
Fir savedFir = firRepo.save(fir);
        return mapToResponse(savedFir);
    }


    @Override
    public FirResponseDto assignInvestigatingOfficer(Long firId, Long policeId) {
        Fir fir = getFirById(firId);
        Police officer = policeRepo.findById(policeId)
                .orElseThrow(() -> new ResourceNotFoundException("Police officer not found with id: " + policeId));
        
        fir.setInvestigatingOfficer(officer);
        Fir updatedFir = firRepo.save(fir);
        return mapToResponse(updatedFir);
    }

    @Override
    public FirResponseDto updateFirStatus(Long firId, fir_status status) {
        Fir fir = getFirById(firId);
        fir.setStatus(status);
        Fir updatedFir = firRepo.save(fir);
        return mapToResponse(updatedFir);
    }

    @Override
    public FirResponseDto updateIpcSections(Long firId, String ipcSections) {
        Fir fir = getFirById(firId);
        fir.setIpcSections(ipcSections);
        Fir updatedFir = firRepo.save(fir);
        return mapToResponse(updatedFir);
    }

    @Override
    public Fir getFirById(Long firId) {
        return firRepo.findById(firId)
                .orElseThrow(() -> new ResourceNotFoundException("FIR not found with id: " + firId));
    }

    @Override
    public FirResponseDto getFirResponseById(Long firId) {
        Fir fir = getFirById(firId);
        return mapToResponse(fir);
    }

    @Override
    public FirResponseDto getFirByFirNumber(String firNumber) {
        Fir fir = firRepo.findByFirNumber(firNumber)
                .orElseThrow(() -> new ResourceNotFoundException("FIR not found with number: " + firNumber));
        return mapToResponse(fir);
    }

    @Override
    public List<FirResponseDto> getAllFirs() {
        return firRepo.findAll()
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByStation(Long stationId) {
        // Validate station exists
        if (!policeStationRepo.existsById(stationId)) {
            throw new ResourceNotFoundException("Police station not found with id: " + stationId);
        }
        
        return firRepo.findByPoliceStation_StationId(stationId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByStationAndStatus(Long stationId, fir_status status) {
        if (!policeStationRepo.existsById(stationId)) {
            throw new ResourceNotFoundException("Police station not found with id: " + stationId);
        }
        
        return firRepo.findByPoliceStation_StationIdAndStatus(stationId, status)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByInvestigatingOfficer(Long policeId) {
        if (!policeRepo.existsById(policeId)) {
            throw new ResourceNotFoundException("Police officer not found with id: " + policeId);
        }
        
        return firRepo.findByInvestigatingOfficer_PoliceId(policeId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByInvestigatingOfficerAndStatus(Long policeId, fir_status status) {
        if (!policeRepo.existsById(policeId)) {
            throw new ResourceNotFoundException("Police officer not found with id: " + policeId);
        }
        
        return firRepo.findByInvestigatingOfficer_PoliceIdAndStatus(policeId, status)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getUnassignedFirsByStation(Long stationId) {
        if (!policeStationRepo.existsById(stationId)) {
            throw new ResourceNotFoundException("Police station not found with id: " + stationId);
        }
        
        return firRepo.findByPoliceStation_StationIdAndInvestigatingOfficerIsNull(stationId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByStatus(fir_status status) {
        return firRepo.findByStatus(status)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByDistrict(String district) {
        return firRepo.findByDistrict(district)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByDistrictAndStatus(String district, fir_status status) {
        return firRepo.findByDistrictAndStatus(district, status)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByCrimeCategory(String crimeCategory) {
        return firRepo.findByCrimeCategory(crimeCategory)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByInformantContact(String contact) {
        return firRepo.findByInformantContact(contact)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByInformantEmail(String email) {
        return firRepo.findByInformantEmail(email)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public FirResponseDto getFirByComplaintId(Long complaintId) {
        Fir fir = firRepo.findByComplaint_Id(complaintId)
                .orElseThrow(() -> new ResourceNotFoundException("FIR not found for complaint id: " + complaintId));
        return mapToResponse(fir);
    }

    @Override
    public List<FirResponseDto> getFirsByRegistrationDateRange(LocalDateTime startDate, LocalDateTime endDate) {
        return firRepo.findByRegisteredAtBetween(startDate, endDate)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> getFirsByIncidentDateRange(LocalDate startDate, LocalDate endDate) {
        return firRepo.findByIncidentDateBetween(startDate, endDate)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> searchFirsByInformantName(String name) {
        return firRepo.findByInformantNameContainingIgnoreCase(name)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<FirResponseDto> searchFirsByDescription(String keyword) {
        return firRepo.findByIncidentDescriptionContainingIgnoreCase(keyword)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public long countFirsByStatus(fir_status status) {
        return firRepo.countByStatus(status);
    }

    @Override
    public long countFirsByStation(Long stationId) {
        return firRepo.countByPoliceStation_StationId(stationId);
    }

    @Override
    public long countFirsByInvestigatingOfficer(Long policeId) {
        return firRepo.countByInvestigatingOfficer_PoliceId(policeId);
    }

    @Override
    public void deleteFir(Long firId) {
        if (!firRepo.existsById(firId)) {
            throw new ResourceNotFoundException("FIR not found with id: " + firId);
        }
        firRepo.deleteById(firId);
    }

    // Helper method to map entity to response DTO
    private FirResponseDto mapToResponse(Fir fir) {
        FirResponseDto response = new FirResponseDto();
        
        // Basic FIR details
        response.setFirId(fir.getFirId());
        response.setFirNumber(fir.getFirNumber());
        response.setDistrict(fir.getDistrict());
        
        // Informant details
        response.setInformantName(fir.getInformantName());
        response.setInformantGuardianName(fir.getInformantGuardianName());
        response.setInformantAddress(fir.getInformantAddress());
        response.setInformantContact(fir.getInformantContact());
        response.setInformantEmail(fir.getInformantEmail());
        response.setInformantFax(fir.getInformantFax());
        
        // Incident details
        response.setIncidentLocation(fir.getIncidentLocation());
        response.setIncidentDate(fir.getIncidentDate());
        response.setIncidentTime(fir.getIncidentTime());
        response.setIncidentDescription(fir.getIncidentDescription());
        
        // Crime details
        response.setCrimeCategory(fir.getCrimeCategory());
        response.setIpcSections(fir.getIpcSections());
        response.setStolenPropertyDetails(fir.getStolenPropertyDetails());
        response.setAccusedDetails(fir.getAccusedDetails());
        response.setWitnessDetails(fir.getWitnessDetails());
        
        // Status
        response.setStatus(fir.getStatus());
        
        // Police station details
        if (fir.getPoliceStation() != null) {
            response.setPoliceStationId(fir.getPoliceStation().getStationId());
            response.setPoliceStationName(fir.getPoliceStation().getStationName());
            response.setPoliceStationCode(fir.getPoliceStation().getStationCode());
            response.setPoliceStationAddress(fir.getPoliceStation().getAddress());
        }
        
        // Investigating officer details
        if (fir.getInvestigatingOfficer() != null) {
            response.setInvestigatingOfficerId(fir.getInvestigatingOfficer().getPoliceId());
            response.setInvestigatingOfficerName(fir.getInvestigatingOfficer().getName());
            response.setInvestigatingOfficerBadgeNumber(fir.getInvestigatingOfficer().getBadgeNumber());
            response.setInvestigatingOfficerRank(fir.getInvestigatingOfficer().getRank());
            response.setInvestigatingOfficerContact(fir.getInvestigatingOfficer().getMobileNumber());
        }
        
        // Complaint details
        if (fir.getComplaint() != null) {
            response.setComplaintId(fir.getComplaint().getId());
            response.setComplaintDescription(fir.getComplaint().getDescription());
            response.setComplaintPredictedCategory(fir.getComplaint().getPredictedCategory());
        }
        
        // Additional info
        response.setFirWrittenBy(fir.getFirWrittenBy());
        response.setInformantSignaturePath(fir.getInformantSignaturePath());
        response.setRegisteredAt(fir.getRegisteredAt());
        
        return response;
    }

    // Helper method to map request DTO to entity
    private Fir mapToEntity(FirRequestDto request, PoliceStation policeStation, Police investigatingOfficer, Complaint complaint) {
        Fir fir = new Fir();
        
        fir.setFirNumber(request.getFirNumber());
        fir.setDistrict(request.getDistrict());
        fir.setInformantName(request.getInformantName());
        fir.setInformantGuardianName(request.getInformantGuardianName());
        fir.setInformantAddress(request.getInformantAddress());
        fir.setInformantContact(request.getInformantContact());
        fir.setInformantEmail(request.getInformantEmail());
        fir.setInformantFax(request.getInformantFax());
        fir.setIncidentLocation(request.getIncidentLocation());
        fir.setIncidentDate(request.getIncidentDate());
        fir.setIncidentTime(request.getIncidentTime());
        fir.setIncidentDescription(request.getIncidentDescription());
        fir.setCrimeCategory(request.getCrimeCategory());
        fir.setIpcSections(request.getIpcSections());
        fir.setStolenPropertyDetails(request.getStolenPropertyDetails());
        fir.setAccusedDetails(request.getAccusedDetails());
        fir.setWitnessDetails(request.getWitnessDetails());
        fir.setStatus(request.getStatus());
        fir.setPoliceStation(policeStation);
        fir.setInvestigatingOfficer(investigatingOfficer);
        fir.setComplaint(complaint);
        fir.setFirWrittenBy(request.getFirWrittenBy());
        fir.setInformantSignaturePath(request.getInformantSignaturePath());
        
        return fir;
    }
}
