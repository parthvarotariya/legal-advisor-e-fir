package com.legal_advisor_e_fir.backend.dto;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;
import java.util.Map;

@Getter
@Setter
public class ErrorResponse {

    private int status;
    private String errorCode;
    private String message;
    private LocalDateTime timestamp;
    private Map<String, String> fieldErrors; // for validation

    public ErrorResponse(int status, String errorCode, String message) {
        this.status = status;
        this.errorCode = errorCode;
        this.message = message;
        this.timestamp = LocalDateTime.now();
    }

    public ErrorResponse(int status, String errorCode, String message,
                         Map<String, String> fieldErrors) {
        this(status, errorCode, message);
        this.fieldErrors = fieldErrors;
    }

}
