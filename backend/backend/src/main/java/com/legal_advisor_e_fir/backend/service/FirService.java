package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.FirFromReportRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirRequestDto;
import com.legal_advisor_e_fir.backend.dto.FirResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.DuplicateResourceException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.Complaint;
import com.legal_advisor_e_fir.backend.model.Fir;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.PreliminaryReport;
import com.legal_advisor_e_fir.backend.model.FirStatus;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import com.legal_advisor_e_fir.backend.repository.FirRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import com.legal_advisor_e_fir.backend.repository.PreliminaryReportRepo;
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
    private final PreliminaryReportRepo preliminaryReportRepo;

    public FirService(FirRepo firRepo,
                     PoliceStationRepo policeStationRepo,
                     PoliceRepo policeRepo,
                     ComplaintRepo complaintRepo,
                     PreliminaryReportRepo preliminaryReportRepo) {
        this.firRepo = firRepo;
        this.policeStationRepo = policeStationRepo;
        this.policeRepo = policeRepo;
        this.complaintRepo = complaintRepo;
        this.preliminaryReportRepo = preliminaryReportRepo;
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
    public FirResponseDto createFirFromReport(FirFromReportRequestDto request) {
        if (firRepo.existsByFirNumber(request.getFirNumber())) {
            throw new DuplicateResourceException("FIR number already exists: " + request.getFirNumber());
        }

        PreliminaryReport report = preliminaryReportRepo.findById(request.getReportId())
                .orElseThrow(() -> new ResourceNotFoundException("Preliminary report not found with id: " + request.getReportId()));

        // Check if complaint already has a FIR
        if (report.getComplaint() != null && firRepo.existsByComplaint_Id(report.getComplaint().getId())) {
            throw new DuplicateResourceException("Complaint already has a FIR associated with it");
        }

        Fir fir = mapReportToFir(request, report);
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
    public FirResponseDto updateFirStatus(Long firId, FirStatus status) {
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
    public List<FirResponseDto> getFirsByStationAndStatus(Long stationId, FirStatus status) {
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
    public List<FirResponseDto> getFirsByInvestigatingOfficerAndStatus(Long policeId, FirStatus status) {
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
    public List<FirResponseDto> getFirsByStatus(FirStatus status) {
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
    public List<FirResponseDto> getFirsByDistrictAndStatus(String district, FirStatus status) {
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
    public long countFirsByStatus(FirStatus status) {
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

    @Override
    public List<FirResponseDto> getFirsBySubdivision(Long subdivisionId) {
        List<Fir> firs = firRepo.findByPoliceStation_Subdivision_SubdivisionId(subdivisionId);
        return firs.stream().map(this::mapToResponse).toList();
    }

    // Helper method to map preliminary report to FIR entity
    private Fir mapReportToFir(FirFromReportRequestDto request, PreliminaryReport report) {
        Fir fir = new Fir();

        // From request
        fir.setFirNumber(request.getFirNumber());
        fir.setDistrict(request.getDistrict());
        fir.setIncidentDescription(request.getIncidentDescription());
        fir.setStatus(request.getStatus());
        fir.setFirWrittenBy(request.getFirWrittenBy());
        fir.setInformantSignaturePath(request.getInformantSignaturePath());

        // From report
        fir.setInformantName(report.getInformantName());
        fir.setInformantAddress(report.getInformantAddress());
        fir.setInformantContact(report.getInformantContact());
        fir.setInformantEmail(report.getInformantEmail());
        fir.setIncidentLocation(report.getIncidentLocation());
        fir.setIncidentDate(report.getIncidentDate());
        fir.setIncidentTime(report.getIncidentTime());
        fir.setCrimeCategory(report.getCrimeCategory());
        fir.setIpcSections(report.getIpcSections());
        fir.setStolenPropertyDetails(report.getStolenPropertyDetails());
        fir.setAccusedDetails(report.getDraftAccusedDetails());
        fir.setWitnessDetails(report.getDraftWitnessDetails());
        fir.setPoliceStation(report.getStation());
        fir.setInvestigatingOfficer(report.getInvestigatingOfficer());
        fir.setComplaint(report.getComplaint());

        // BNSS 2023 Compliance Fields from request
        fir.setIsZeroFir(request.getIsZeroFir() != null ? request.getIsZeroFir() : false);
        fir.setDestinationPoliceStation(request.getDestinationPoliceStation());
        fir.setIsEfir(request.getIsEfir() != null ? request.getIsEfir() : false);
        fir.setIsSignatureObtained(request.getIsSignatureObtained());
        fir.setIsVictimWoman(request.getIsVictimWoman() != null ? request.getIsVictimWoman() : false);
        fir.setRecordedByWomanOfficer(request.getRecordedByWomanOfficer());
        fir.setIsDisabledVictim(request.getIsDisabledVictim() != null ? request.getIsDisabledVictim() : false);
        fir.setInterpreterOrEducatorName(request.getInterpreterOrEducatorName());
        fir.setVideoRecordingPath(request.getVideoRecordingPath());
        fir.setIsMagistrateStatementRecorded(request.getIsMagistrateStatementRecorded());

        return fir;
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
        
        // BNSS 2023 Compliance Fields
        response.setIsZeroFir(fir.getIsZeroFir());
        response.setDestinationPoliceStation(fir.getDestinationPoliceStation());
        response.setIsEfir(fir.getIsEfir());
        response.setSignatureDeadline(fir.getSignatureDeadline());
        response.setIsSignatureObtained(fir.getIsSignatureObtained());
        response.setIsVictimWoman(fir.getIsVictimWoman());
        response.setRecordedByWomanOfficer(fir.getRecordedByWomanOfficer());
        response.setIsDisabledVictim(fir.getIsDisabledVictim());
        response.setInterpreterOrEducatorName(fir.getInterpreterOrEducatorName());
        response.setVideoRecordingPath(fir.getVideoRecordingPath());
        response.setIsMagistrateStatementRecorded(fir.getIsMagistrateStatementRecorded());
        
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
        
        // BNSS 2023 Compliance Fields
        fir.setIsZeroFir(request.getIsZeroFir() != null ? request.getIsZeroFir() : false);
        fir.setDestinationPoliceStation(request.getDestinationPoliceStation());
        fir.setIsEfir(request.getIsEfir() != null ? request.getIsEfir() : false);
        fir.setIsSignatureObtained(request.getIsSignatureObtained());
        fir.setIsVictimWoman(request.getIsVictimWoman() != null ? request.getIsVictimWoman() : false);
        fir.setRecordedByWomanOfficer(request.getRecordedByWomanOfficer());
        fir.setIsDisabledVictim(request.getIsDisabledVictim() != null ? request.getIsDisabledVictim() : false);
        fir.setInterpreterOrEducatorName(request.getInterpreterOrEducatorName());
        fir.setVideoRecordingPath(request.getVideoRecordingPath());
        fir.setIsMagistrateStatementRecorded(request.getIsMagistrateStatementRecorded());
        
        return fir;
    }
}
