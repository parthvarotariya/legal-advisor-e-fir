package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.SubdivisionRequestDto;
import com.legal_advisor_e_fir.backend.dto.SubdivisionResponseDto;

import java.util.List;

public interface ISubdivisionService {

    SubdivisionResponseDto createSubdivision(SubdivisionRequestDto requestDto);

    SubdivisionResponseDto getById(Long id);

    SubdivisionResponseDto getByCode(String code);

    List<SubdivisionResponseDto> getAllSubdivisions();

    List<SubdivisionResponseDto> getByDistrict(String district);

    List<SubdivisionResponseDto> getByState(String state);

    SubdivisionResponseDto updateSubdivision(Long id, SubdivisionRequestDto requestDto);

    SubdivisionResponseDto assignDspOfficer(Long subdivisionId, Long dspOfficerId);

    SubdivisionResponseDto removeDspOfficer(Long subdivisionId);

    SubdivisionResponseDto addPoliceStation(Long subdivisionId, Long policeStationId);

    SubdivisionResponseDto removePoliceStation(Long subdivisionId, Long policeStationId);

    void deleteSubdivision(Long id);
}
