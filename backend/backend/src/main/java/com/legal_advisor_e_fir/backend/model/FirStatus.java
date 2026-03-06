package com.legal_advisor_e_fir.backend.model;

public enum FirStatus {
    REGISTERED,
    UNDER_INVESTIGATION,
    CHARGE_SHEET_SUBMITTED,   // IO submitted charge sheet to PI
    CHARGE_SHEET_APPROVED,    // PI approved the charge sheet
    CHARGE_SHEET_FILED,       // Dispatched to court
    CLOSED
}
