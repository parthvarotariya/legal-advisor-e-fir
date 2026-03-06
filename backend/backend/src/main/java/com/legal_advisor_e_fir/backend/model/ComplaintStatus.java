package com.legal_advisor_e_fir.backend.model;

public enum ComplaintStatus {
    RECEIVED,                  // Fresh complaint
    PE_PENDING_DSP_APPROVAL,   // PI decided to do PE, waiting for DSP nod
    PE_ASSIGNED,               // SI is currently doing the PE
    PE_SUBMITTED,              // SI finished PE, PI needs to review
    FIR_REGISTERED,            // FIR was created (either directly or after PE)
    CLOSED_NO_CRIME            // PI closed it after PE showed no crime
}
