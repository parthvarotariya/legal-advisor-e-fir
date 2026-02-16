package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.dto.ComplaintResponseDto;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.User;
import com.legal_advisor_e_fir.backend.model.complaint_status;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;
import jakarta.validation.constraints.AssertTrue;

import java.util.List;

public interface IComplaintService {
    ComplaintResponseDto createComplaint(ComplaintRequestDto request);

    ComplaintResponseDto getById(Long id);

    List<ComplaintResponseDto> getByUser(Long id);

    List<ComplaintResponseDto> getByPoliceStation(Long  stationId);
    //.
    ComplaintResponseDto updateComplaint(Long id, complaint_status status, String actualCategory, Long officerId);

}
