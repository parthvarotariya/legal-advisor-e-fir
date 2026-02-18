package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PreliminaryReportRequestDto;
import com.legal_advisor_e_fir.backend.dto.PreliminaryReportResponseDto;
import com.legal_advisor_e_fir.backend.model.PreliminaryReport;

import java.util.List;

public interface IPreliminaryReportService {

    PreliminaryReportResponseDto createPreliminaryReport(PreliminaryReportRequestDto request);

    PreliminaryReport getPreliminaryReportById(Long reportId);

    PreliminaryReportResponseDto getPreliminaryReportResponseById(Long reportId);

    PreliminaryReportResponseDto findByComplaintId(Long complaintId);

    List<PreliminaryReportResponseDto> findByStationId(Long stationId);

    List<PreliminaryReportResponseDto> findByInvestigatingOfficerId(Long policeId);

    List<PreliminaryReportResponseDto> getAllPreliminaryReports();

    void deletePreliminaryReport(Long reportId);
}
