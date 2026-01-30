package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Positive;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class ComplaintRequestDto {
    @NotBlank(message = "Description is required")
    @Size(min = 20, max = 2000, message = "Description must be between 20 and 2000 characters")
    private String description;

    @NotBlank(message = "Predicted category is required")
    @Size(min = 3, max = 100, message = "Predicted category must be between 3 and 100 characters")
    private String predictedCategory;

    @NotNull(message = "User ID is required")
    @Positive(message = "User ID must be positive")
    private Long userId;

    @Positive(message = "Police Station ID must be positive")
    private Long policeStationId;
}
