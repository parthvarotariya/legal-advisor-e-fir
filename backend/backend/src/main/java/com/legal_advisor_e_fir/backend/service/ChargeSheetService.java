package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ChargeSheetRequestDto;
import com.legal_advisor_e_fir.backend.dto.ChargeSheetResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.DuplicateResourceException;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.*;
import com.legal_advisor_e_fir.backend.repository.ChargeSheetRepo;
import com.legal_advisor_e_fir.backend.repository.FirRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceRepo;
import com.legal_advisor_e_fir.backend.repository.PoliceStationRepo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.stream.Collectors;

@Service
@Transactional
public class ChargeSheetService implements IChargeSheetService {

    private final ChargeSheetRepo chargeSheetRepo;
    private final FirRepo firRepo;
    private final PoliceRepo policeRepo;
    private final PoliceStationRepo policeStationRepo;

    public ChargeSheetService(ChargeSheetRepo chargeSheetRepo, FirRepo firRepo,
                              PoliceRepo policeRepo, PoliceStationRepo policeStationRepo) {
        this.chargeSheetRepo = chargeSheetRepo;
        this.firRepo = firRepo;
        this.policeRepo = policeRepo;
        this.policeStationRepo = policeStationRepo;
    }

    // ────────────────── IO (PSI) ACTIONS ──────────────────

    @Override
    public ChargeSheetResponseDto createChargeSheet(ChargeSheetRequestDto request) {
        if (chargeSheetRepo.existsByChargeSheetNumber(request.getChargeSheetNumber())) {
            throw new DuplicateResourceException("Charge sheet number already exists: " + request.getChargeSheetNumber());
        }

        Fir fir = firRepo.findById(request.getFirId())
                .orElseThrow(() -> new ResourceNotFoundException("FIR not found with id: " + request.getFirId()));

        PoliceStation station = policeStationRepo.findById(request.getPoliceStationId())
                .orElseThrow(() -> new ResourceNotFoundException("Police station not found with id: " + request.getPoliceStationId()));

        Police io = policeRepo.findById(request.getInvestigatingOfficerId())
                .orElseThrow(() -> new ResourceNotFoundException("Officer not found with id: " + request.getInvestigatingOfficerId()));

        ChargeSheet cs = mapRequestToEntity(request, fir, station, io);
        cs.setStatus(ChargeSheetStatus.DRAFT);
        cs.setRevisionCount(0);

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    @Override
    public ChargeSheetResponseDto updateChargeSheet(Long chargeSheetId, ChargeSheetRequestDto request) {
        ChargeSheet cs = getEntityById(chargeSheetId);

        // Only allow edits when DRAFT or RETURNED_FOR_REVISION
        if (cs.getStatus() != ChargeSheetStatus.DRAFT &&
            cs.getStatus() != ChargeSheetStatus.RETURNED_FOR_REVISION) {
            throw new IllegalStateException("Charge sheet can only be edited in DRAFT or RETURNED_FOR_REVISION status. Current: " + cs.getStatus());
        }

        // Update fields
        cs.setReportType(FinalReportType.valueOf(request.getReportType()));
        cs.setActsAndSections(request.getActsAndSections());
        cs.setBriefFacts(request.getBriefFacts());
        cs.setAccusedChargeSheetedJson(request.getAccusedChargeSheetedJson());
        cs.setAccusedNotChargeSheetedJson(request.getAccusedNotChargeSheetedJson());
        cs.setAccusedAbscondingJson(request.getAccusedAbscondingJson());
        cs.setSeizedPropertyJson(request.getSeizedPropertyJson());
        cs.setChainOfCustody(request.getChainOfCustody());
        cs.setLaboratoryResult(request.getLaboratoryResult());
        cs.setWitnessListJson(request.getWitnessListJson());
        cs.setComplainantNotified(request.getComplainantNotified() != null ? request.getComplainantNotified() : false);
        cs.setDistrict(request.getDistrict());

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    @Override
    public ChargeSheetResponseDto submitToPI(Long chargeSheetId) {
        ChargeSheet cs = getEntityById(chargeSheetId);

        if (cs.getStatus() != ChargeSheetStatus.DRAFT &&
            cs.getStatus() != ChargeSheetStatus.RETURNED_FOR_REVISION) {
            throw new IllegalStateException("Only DRAFT or RETURNED charge sheets can be submitted. Current: " + cs.getStatus());
        }

        cs.setStatus(ChargeSheetStatus.SUBMITTED_TO_PI);
        cs.setSubmittedAt(LocalDateTime.now());

        // Update FIR status to CHARGE_SHEETED
        Fir fir = cs.getFir();
        fir.setStatus(FirStatus.CHARGE_SHEET_SUBMITTED);
        firRepo.save(fir);

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    // ────────────────── PI ACTIONS ──────────────────

    @Override
    public ChargeSheetResponseDto approveChargeSheet(Long chargeSheetId, Long approvingOfficerId) {
        ChargeSheet cs = getEntityById(chargeSheetId);

        if (cs.getStatus() != ChargeSheetStatus.SUBMITTED_TO_PI) {
            throw new IllegalStateException("Only SUBMITTED_TO_PI charge sheets can be approved. Current: " + cs.getStatus());
        }

        Police approver = policeRepo.findById(approvingOfficerId)
                .orElseThrow(() -> new ResourceNotFoundException("Approving officer not found with id: " + approvingOfficerId));

        cs.setStatus(ChargeSheetStatus.APPROVED_BY_PI);
        cs.setApprovingOfficer(approver);
        cs.setApprovedAt(LocalDateTime.now());
        cs.setPiSuggestions(null);

        // Update FIR status
        Fir fir = cs.getFir();
        fir.setStatus(FirStatus.CHARGE_SHEET_APPROVED);
        firRepo.save(fir);

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    @Override
    public ChargeSheetResponseDto returnForRevision(Long chargeSheetId, Long approvingOfficerId, String suggestions) {
        ChargeSheet cs = getEntityById(chargeSheetId);

        if (cs.getStatus() != ChargeSheetStatus.SUBMITTED_TO_PI) {
            throw new IllegalStateException("Only SUBMITTED_TO_PI charge sheets can be returned. Current: " + cs.getStatus());
        }

        Police approver = policeRepo.findById(approvingOfficerId)
                .orElseThrow(() -> new ResourceNotFoundException("Officer not found with id: " + approvingOfficerId));

        cs.setStatus(ChargeSheetStatus.RETURNED_FOR_REVISION);
        cs.setApprovingOfficer(approver);
        cs.setPiSuggestions(suggestions);
        cs.setRevisionCount(cs.getRevisionCount() != null ? cs.getRevisionCount() + 1 : 1);

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    @Override
    public ChargeSheetResponseDto dispatchToCourt(Long chargeSheetId) {
        ChargeSheet cs = getEntityById(chargeSheetId);

        if (cs.getStatus() != ChargeSheetStatus.APPROVED_BY_PI) {
            throw new IllegalStateException("Only APPROVED charge sheets can be dispatched. Current: " + cs.getStatus());
        }

        cs.setStatus(ChargeSheetStatus.DISPATCHED_TO_COURT);
        cs.setDispatchedAt(LocalDateTime.now());

        // Update FIR status
        Fir fir = cs.getFir();
        fir.setStatus(FirStatus.CHARGE_SHEET_FILED);
        firRepo.save(fir);

        ChargeSheet saved = chargeSheetRepo.save(cs);
        return mapToResponse(saved);
    }

    // ────────────────── QUERIES ──────────────────

    @Override
    public ChargeSheetResponseDto getById(Long chargeSheetId) {
        return mapToResponse(getEntityById(chargeSheetId));
    }

    @Override
    public ChargeSheetResponseDto getByChargeSheetNumber(String chargeSheetNumber) {
        ChargeSheet cs = chargeSheetRepo.findByChargeSheetNumber(chargeSheetNumber)
                .orElseThrow(() -> new ResourceNotFoundException("Charge sheet not found: " + chargeSheetNumber));
        return mapToResponse(cs);
    }

    @Override
    public List<ChargeSheetResponseDto> getByFirId(Long firId) {
        return chargeSheetRepo.findByFir_FirId(firId).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getByStation(Long stationId) {
        return chargeSheetRepo.findByPoliceStation_StationId(stationId).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getByStationAndStatus(Long stationId, String status) {
        ChargeSheetStatus csStatus = ChargeSheetStatus.valueOf(status);
        return chargeSheetRepo.findByPoliceStation_StationIdAndStatus(stationId, csStatus).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getByOfficer(Long policeId) {
        return chargeSheetRepo.findByInvestigatingOfficer_PoliceId(policeId).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getByOfficerAndStatus(Long policeId, String status) {
        ChargeSheetStatus csStatus = ChargeSheetStatus.valueOf(status);
        return chargeSheetRepo.findByInvestigatingOfficer_PoliceIdAndStatus(policeId, csStatus).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getBySubdivision(Long subdivisionId) {
        return chargeSheetRepo.findByPoliceStation_Subdivision_SubdivisionId(subdivisionId).stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public List<ChargeSheetResponseDto> getPendingApprovalByStation(Long stationId) {
        return chargeSheetRepo.findByPoliceStation_StationIdAndStatus(stationId, ChargeSheetStatus.SUBMITTED_TO_PI)
                .stream().map(this::mapToResponse).collect(Collectors.toList());
    }

    @Override
    public void deleteChargeSheet(Long chargeSheetId) {
        ChargeSheet cs = getEntityById(chargeSheetId);
        if (cs.getStatus() != ChargeSheetStatus.DRAFT) {
            throw new IllegalStateException("Only DRAFT charge sheets can be deleted.");
        }
        chargeSheetRepo.deleteById(chargeSheetId);
    }

    // ────────────────── HELPERS ──────────────────

    private ChargeSheet getEntityById(Long id) {
        return chargeSheetRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Charge sheet not found with id: " + id));
    }

    private ChargeSheet mapRequestToEntity(ChargeSheetRequestDto req, Fir fir, PoliceStation station, Police io) {
        ChargeSheet cs = new ChargeSheet();
        cs.setChargeSheetNumber(req.getChargeSheetNumber());
        cs.setDistrict(req.getDistrict());
        cs.setFir(fir);
        cs.setPoliceStation(station);
        cs.setReportType(FinalReportType.valueOf(req.getReportType()));
        cs.setActsAndSections(req.getActsAndSections());
        cs.setBriefFacts(req.getBriefFacts());
        cs.setAccusedChargeSheetedJson(req.getAccusedChargeSheetedJson());
        cs.setAccusedNotChargeSheetedJson(req.getAccusedNotChargeSheetedJson());
        cs.setAccusedAbscondingJson(req.getAccusedAbscondingJson());
        cs.setSeizedPropertyJson(req.getSeizedPropertyJson());
        cs.setChainOfCustody(req.getChainOfCustody());
        cs.setLaboratoryResult(req.getLaboratoryResult());
        cs.setWitnessListJson(req.getWitnessListJson());
        cs.setComplainantNotified(req.getComplainantNotified() != null ? req.getComplainantNotified() : false);
        cs.setInvestigatingOfficer(io);
        return cs;
    }

    private ChargeSheetResponseDto mapToResponse(ChargeSheet cs) {
        ChargeSheetResponseDto dto = new ChargeSheetResponseDto();
        dto.setChargeSheetId(cs.getChargeSheetId());
        dto.setChargeSheetNumber(cs.getChargeSheetNumber());
        dto.setDistrict(cs.getDistrict());
        dto.setReportType(cs.getReportType());
        dto.setStatus(cs.getStatus());
        dto.setActsAndSections(cs.getActsAndSections());
        dto.setBriefFacts(cs.getBriefFacts());
        dto.setAccusedChargeSheetedJson(cs.getAccusedChargeSheetedJson());
        dto.setAccusedNotChargeSheetedJson(cs.getAccusedNotChargeSheetedJson());
        dto.setAccusedAbscondingJson(cs.getAccusedAbscondingJson());
        dto.setSeizedPropertyJson(cs.getSeizedPropertyJson());
        dto.setChainOfCustody(cs.getChainOfCustody());
        dto.setLaboratoryResult(cs.getLaboratoryResult());
        dto.setWitnessListJson(cs.getWitnessListJson());
        dto.setComplainantNotified(cs.getComplainantNotified());
        dto.setPiSuggestions(cs.getPiSuggestions());
        dto.setRevisionCount(cs.getRevisionCount());
        dto.setCreatedAt(cs.getCreatedAt());
        dto.setUpdatedAt(cs.getUpdatedAt());
        dto.setSubmittedAt(cs.getSubmittedAt());
        dto.setApprovedAt(cs.getApprovedAt());
        dto.setDispatchedAt(cs.getDispatchedAt());

        // FIR info
        if (cs.getFir() != null) {
            dto.setFirId(cs.getFir().getFirId());
            dto.setFirNumber(cs.getFir().getFirNumber());
            dto.setFirCrimeCategory(cs.getFir().getCrimeCategory());
            dto.setFirIncidentDescription(cs.getFir().getIncidentDescription());
        }

        // Station info
        if (cs.getPoliceStation() != null) {
            dto.setPoliceStationId(cs.getPoliceStation().getStationId());
            dto.setPoliceStationName(cs.getPoliceStation().getStationName());
            dto.setPoliceStationCode(cs.getPoliceStation().getStationCode());
        }

        // IO info
        if (cs.getInvestigatingOfficer() != null) {
            dto.setInvestigatingOfficerId(cs.getInvestigatingOfficer().getPoliceId());
            dto.setInvestigatingOfficerName(cs.getInvestigatingOfficer().getName());
            dto.setInvestigatingOfficerBadgeNumber(cs.getInvestigatingOfficer().getBadgeNumber());
            dto.setInvestigatingOfficerRank(cs.getInvestigatingOfficer().getRank());
        }

        // Approving officer info
        if (cs.getApprovingOfficer() != null) {
            dto.setApprovingOfficerId(cs.getApprovingOfficer().getPoliceId());
            dto.setApprovingOfficerName(cs.getApprovingOfficer().getName());
            dto.setApprovingOfficerBadgeNumber(cs.getApprovingOfficer().getBadgeNumber());
        }

        return dto;
    }
}
