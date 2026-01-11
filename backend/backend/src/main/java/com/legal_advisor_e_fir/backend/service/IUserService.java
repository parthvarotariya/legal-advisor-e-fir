package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.UpdateRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserResponseDto;

import java.util.List;

public interface IUserService {

    UserResponseDto createUser(UserRequestDto request);

    List<UserResponseDto> getAllUsers();

    UserResponseDto getUserById(Long id);

    UserResponseDto updateUser(Long id, UpdateRequestDto request);

    void deleteUserById(Long id);
}
