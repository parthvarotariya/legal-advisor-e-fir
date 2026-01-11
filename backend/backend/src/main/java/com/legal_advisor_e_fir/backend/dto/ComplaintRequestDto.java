package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.PoliceStation;
import com.legal_advisor_e_fir.backend.model.User;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ComplaintRequestDto {
    @NotBlank
    private String description;

    @NotBlank
    private String predictedCategory;

    @NotNull
    private User user;

    private PoliceStation policeStation;
}
