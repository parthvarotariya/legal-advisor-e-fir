package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.PoliceRequestDto;
import com.legal_advisor_e_fir.backend.dto.PoliceResponseDto;
import com.legal_advisor_e_fir.backend.dto.PoliceUpdateRequestDto;
import com.legal_advisor_e_fir.backend.model.Police;
import com.legal_advisor_e_fir.backend.model.Role;

import java.util.List;

public interface IPoliceService {

    PoliceResponseDto registerPolice(PoliceRequestDto request);

    Police getPoliceById(Long id);

    List<PoliceResponseDto> getAllPolice();

    PoliceResponseDto getPoliceResponseById(Long id);

    PoliceResponseDto updatePolice(Long id, PoliceUpdateRequestDto request);

    void deletePoliceById(Long id);

    PoliceResponseDto getPoliceByEmail(String email);

    PoliceResponseDto getPoliceByBadgeNumber(String badgeNumber);

    List<PoliceResponseDto> getPoliceByStationId(Long stationId);

    List<PoliceResponseDto> getPoliceByRole(Role role);
}
