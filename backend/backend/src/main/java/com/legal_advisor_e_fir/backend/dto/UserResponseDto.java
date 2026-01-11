package com.legal_advisor_e_fir.backend.dto;


import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class UserResponseDto {
    private Long id;
    private String name;
    private String mobileNumber;
    private String email;
    private String address;

    public UserResponseDto(){}
    public UserResponseDto(Long id, String name, String mobileNumber, String email, String address) {
        this.id = id;
        this.name = name;
        this.mobileNumber = mobileNumber;
        this.email = email;
        this.address = address;
    }
}
