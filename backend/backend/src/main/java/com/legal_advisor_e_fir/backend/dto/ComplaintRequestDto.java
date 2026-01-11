package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ComplaintRequestDto {
    @NotBlank
    private String description;
}
