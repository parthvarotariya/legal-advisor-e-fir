package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.UpdateRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserResponseDto;
import com.legal_advisor_e_fir.backend.model.User;

import java.util.List;

public interface IUserService {

    User getUserById(Long id);
    List<UserResponseDto> getAllUsers();

    UserResponseDto getUserResponseById(Long id);

    UserResponseDto updateUser(Long id, UpdateRequestDto request);

    void deleteUserById(Long id);
}
