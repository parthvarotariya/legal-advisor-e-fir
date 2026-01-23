package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.*;

public interface  IAuthService {

    UserLoginResponseDto userLogin(LoginRequestDto request);

    UserLoginResponseDto userRegister(UserRequestDto request);

    PoliceLoginResponseDto policeLogin(LoginRequestDto request);
    PoliceLoginResponseDto policeRegister(PoliceRequestDto request);
}
