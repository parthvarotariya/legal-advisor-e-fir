package com.legal_advisor_e_fir.backend.model;

public enum ChargeSheetStatus {
    DRAFT,                      // IO is still editing
    SUBMITTED_TO_PI,            // IO submitted to PI for approval
    RETURNED_FOR_REVISION,      // PI returned with suggestions
    APPROVED_BY_PI,             // PI approved – ready for court dispatch
    DISPATCHED_TO_COURT         // Forwarded to magistrate
}
