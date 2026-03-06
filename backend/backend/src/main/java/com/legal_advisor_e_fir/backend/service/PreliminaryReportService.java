package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PreliminaryReportRequestDto;
import com.legal_advisor_e_fir.backend.dto.PreliminaryReportResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.DuplicateResourceException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.Complaint;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.PreliminaryReport;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import com.legal_advisor_e_fir.backend.repository.PreliminaryReportRepo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class PreliminaryReportService implements IPreliminaryReportService {

    private final PreliminaryReportRepo preliminaryReportRepo;
    private final ComplaintRepo complaintRepo;
    private final PoliceRepo policeRepo;
    private final PoliceStationRepo policeStationRepo;

    public PreliminaryReportService(PreliminaryReportRepo preliminaryReportRepo,
                                    ComplaintRepo complaintRepo,
                                    PoliceRepo policeRepo,
                                    PoliceStationRepo policeStationRepo) {
        this.preliminaryReportRepo = preliminaryReportRepo;
        this.complaintRepo = complaintRepo;
        this.policeRepo = policeRepo;
        this.policeStationRepo = policeStationRepo;
    }

    @Override
    public PreliminaryReportResponseDto createPreliminaryReport(PreliminaryReportRequestDto request) {
        // Check if complaint already has a preliminary report
        if (preliminaryReportRepo.existsByComplaint_Id(request.getComplaintId())) {
            throw new DuplicateResourceException("Complaint already has a preliminary report associated with it");
        }

        Complaint complaint = complaintRepo.findById(request.getComplaintId())
                .orElseThrow(() -> new ResourceNotFoundException("Complaint not found with id: " + request.getComplaintId()));

        Police investigatingOfficer = policeRepo.findById(request.getInvestigatingOfficerId())
                .orElseThrow(() -> new ResourceNotFoundException("Police officer not found with id: " + request.getInvestigatingOfficerId()));

        PoliceStation station = policeStationRepo.findById(request.getStationId())
                .orElseThrow(() -> new ResourceNotFoundException("Police station not found with id: " + request.getStationId()));

        PreliminaryReport report = mapToEntity(request, complaint, investigatingOfficer, station);
        PreliminaryReport savedReport = preliminaryReportRepo.save(report);
        return mapToResponse(savedReport);
    }

    @Override
    public PreliminaryReport getPreliminaryReportById(Long reportId) {
        return preliminaryReportRepo.findById(reportId)
                .orElseThrow(() -> new ResourceNotFoundException("Preliminary report not found with id: " + reportId));
    }

    @Override
    public PreliminaryReportResponseDto getPreliminaryReportResponseById(Long reportId) {
        PreliminaryReport report = getPreliminaryReportById(reportId);
        return mapToResponse(report);
    }

    @Override
    public PreliminaryReportResponseDto findByComplaintId(Long complaintId) {
        PreliminaryReport report = preliminaryReportRepo.findByComplaint_Id(complaintId)
                .orElseThrow(() -> new ResourceNotFoundException("Preliminary report not found for complaint id: " + complaintId));
        return mapToResponse(report);
    }

    @Override
    public List<PreliminaryReportResponseDto> findByStationId(Long stationId) {
        if (!policeStationRepo.existsById(stationId)) {
            throw new ResourceNotFoundException("Police station not found with id: " + stationId);
        }

        return preliminaryReportRepo.findByStation_StationId(stationId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<PreliminaryReportResponseDto> findByInvestigatingOfficerId(Long policeId) {
        if (!policeRepo.existsById(policeId)) {
            throw new ResourceNotFoundException("Police officer not found with id: " + policeId);
        }

        return preliminaryReportRepo.findByInvestigatingOfficer_PoliceId(policeId)
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public List<PreliminaryReportResponseDto> getAllPreliminaryReports() {
        return preliminaryReportRepo.findAll()
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public void deletePreliminaryReport(Long reportId) {
        if (!preliminaryReportRepo.existsById(reportId)) {
            throw new ResourceNotFoundException("Preliminary report not found with id: " + reportId);
        }
        preliminaryReportRepo.deleteById(reportId);
    }

    @Override
    public List<PreliminaryReportResponseDto> findBySubdivisionId(Long subdivisionId) {
        return preliminaryReportRepo.findByStation_Subdivision_SubdivisionId(subdivisionId)
                .stream().map(this::mapToResponse).toList();
    }

    private PreliminaryReportResponseDto mapToResponse(PreliminaryReport report) {
        PreliminaryReportResponseDto response = new PreliminaryReportResponseDto();

        // Report details
        response.setReportId(report.getReportId());
        response.setSubmittedAt(report.getSubmittedAt());

        // Investigation details
        response.setInvestigationNarrative(report.getInvestigationNarrative());
        response.setCognizableOffence(report.getCognizableOffence());

        // Informant details
        response.setInformantName(report.getInformantName());
        response.setInformantAddress(report.getInformantAddress());
        response.setInformantContact(report.getInformantContact());
        response.setInformantEmail(report.getInformantEmail());

        // Incident details
        response.setIncidentLocation(report.getIncidentLocation());
        response.setIncidentDate(report.getIncidentDate());
        response.setIncidentTime(report.getIncidentTime());

        // Crime details
        response.setCrimeCategory(report.getCrimeCategory());
        response.setIpcSections(report.getIpcSections());
        response.setStolenPropertyDetails(report.getStolenPropertyDetails());

        // Draft details
        response.setDraftAccusedDetails(report.getDraftAccusedDetails());
        response.setDraftWitnessDetails(report.getDraftWitnessDetails());
        response.setWitnessStatement(report.getWitnessStatement());

        // Complaint details
        if (report.getComplaint() != null) {
            response.setComplaintId(report.getComplaint().getId());
            response.setComplaintDescription(report.getComplaint().getDescription());
        }

        // Investigating officer details
        if (report.getInvestigatingOfficer() != null) {
            response.setInvestigatingOfficerId(report.getInvestigatingOfficer().getPoliceId());
            response.setInvestigatingOfficerName(report.getInvestigatingOfficer().getName());
            response.setInvestigatingOfficerBadgeNumber(report.getInvestigatingOfficer().getBadgeNumber());
            response.setInvestigatingOfficerRank(report.getInvestigatingOfficer().getRank());
        }

        // Police station details
        if (report.getStation() != null) {
            response.setStationId(report.getStation().getStationId());
            response.setStationName(report.getStation().getStationName());
            response.setStationCode(report.getStation().getStationCode());
        }

        // BNSS 2023 PE Protocol fields
        response.setPermissionGrantedByDspId(report.getPermissionGrantedByDspId());
        // TODO: If you want to populate DSP name, you'll need to fetch the Police entity
        response.setPeCategory(report.getPeCategory());
        response.setPeStartDate(report.getPeStartDate());
        response.setPeDeadline(report.getPeDeadline());
        response.setReasonForRefusal(report.getReasonForRefusal());
        response.setInformantNotifiedOfRefusal(report.getInformantNotifiedOfRefusal());

        return response;
    }

    private PreliminaryReport mapToEntity(PreliminaryReportRequestDto request, Complaint complaint,
                                          Police investigatingOfficer, PoliceStation station) {
        PreliminaryReport report = new PreliminaryReport();

        report.setInvestigationNarrative(request.getInvestigationNarrative());
        report.setCognizableOffence(request.getCognizableOffence());
        report.setInformantName(request.getInformantName());
        report.setInformantAddress(request.getInformantAddress());
        report.setInformantContact(request.getInformantContact());
        report.setInformantEmail(request.getInformantEmail());
        report.setIncidentLocation(request.getIncidentLocation());
        report.setIncidentDate(request.getIncidentDate());
        report.setIncidentTime(request.getIncidentTime());
        report.setCrimeCategory(request.getCrimeCategory());
        report.setIpcSections(request.getIpcSections());
        report.setStolenPropertyDetails(request.getStolenPropertyDetails());
        report.setDraftAccusedDetails(request.getDraftAccusedDetails());
        report.setDraftWitnessDetails(request.getDraftWitnessDetails());
        report.setWitnessStatement(request.getWitnessStatement());
        report.setComplaint(complaint);
        report.setInvestigatingOfficer(investigatingOfficer);
        report.setStation(station);

        // BNSS 2023 PE Protocol fields
        report.setPermissionGrantedByDspId(request.getPermissionGrantedByDspId());
        report.setPeCategory(request.getPeCategory());
        report.setPeStartDate(request.getPeStartDate());
        report.setPeDeadline(request.getPeDeadline());
        report.setReasonForRefusal(request.getReasonForRefusal());
        report.setInformantNotifiedOfRefusal(request.getInformantNotifiedOfRefusal());

        return report;
    }
}
