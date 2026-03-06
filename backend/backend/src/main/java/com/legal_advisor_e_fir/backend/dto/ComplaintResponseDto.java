package com.legal_advisor_e_fir.backend.dto;

import com.legal_advisor_e_fir.backend.model.ComplaintStatus;
import jakarta.persistence.Column;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import lombok.Getter;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;

@Getter
@Setter
public class ComplaintResponseDto {
    private Long id;

    private String description;
    private String actualCategory;

    private String predictedCategory;

    private LocalDateTime createdAt;

    private ComplaintStatus status;

    private Long policeStationId;
    private String policeStationName;
    
    private Long assignedOfficerId;
    private String assignedOfficerName;
    private String assignedOfficerBadge;
    
    // Complainant details
    private Long userId;
    private String complainantName;
    private String complainantMobile;
    private String complainantEmail;
    private String complainantAddress;
}
