package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.ComplaintRequestDto;
import com.legal_advisor_e_fir.backend.repository.ComplaintRepo;

public interface IComplaint {
    void createComplaint(ComplaintRequestDto request);
}
