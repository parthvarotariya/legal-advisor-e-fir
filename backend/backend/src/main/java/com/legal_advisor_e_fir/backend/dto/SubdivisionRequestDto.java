package com.legal_advisor_e_fir.backend.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;
import lombok.Getter;
import lombok.Setter;

/**
 * Data Transfer Object for creating or updating a Subdivision.
 * Used for incoming requests from the frontend.
 */
@Getter
@Setter
public class SubdivisionRequestDto {

    /**
     * Unique code for the subdivision (e.g., "SUB001", "NORTHDIV").
     * Must be 4-15 uppercase alphanumeric characters.
     */
    @NotBlank(message = "Subdivision code is required")
    @Pattern(
            regexp = "^[A-Z0-9-]{4,15}$",
            message = "Subdivision code must be 4-15 characters long and contain only uppercase letters, numbers, and hyphens"
    )
    private String subdivisionCode;

    /**
     * Display name of the subdivision.
     */
    @NotBlank(message = "Subdivision name is required")
    @Size(min = 3, max = 100, message = "Subdivision name must be between 3 and 100 characters")
    private String subdivisionName;

    /**
     * District where the subdivision is located.
     */
    @NotBlank(message = "District is required")
    @Size(min = 2, max = 50, message = "District name must be between 2 and 50 characters")
    private String district;

    /**
     * State where the subdivision is located.
     */
    @NotBlank(message = "State is required")
    @Size(min = 2, max = 50, message = "State name must be between 2 and 50 characters")
    private String state;

    /**
     * Optional ID of the DSP officer to assign to this subdivision.
     * Can be null if no DSP is assigned during creation.
     */
    private Long dspOfficerId;
}
