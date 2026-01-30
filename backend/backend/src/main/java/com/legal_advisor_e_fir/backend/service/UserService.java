package com.legal_advisor_e_fir.backend.service;

import com.legal_advisor_e_fir.backend.dto.UpdateRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserRequestDto;
import com.legal_advisor_e_fir.backend.dto.UserResponseDto;
import com.legal_advisor_e_fir.backend.exceptions.ResourceNotFoundException;
import com.legal_advisor_e_fir.backend.model.User;
import com.legal_advisor_e_fir.backend.repository.UserRepo;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

@Service
@Transactional
public class UserService implements IUserService {

    private final UserRepo userRepo;

    public UserService(UserRepo userRepo) {
        this.userRepo = userRepo;
    }


    @Override
    public User getUserById(Long id) {
        return userRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));
    }

    @Override
    public List<UserResponseDto> getAllUsers() {

        List<User> users = userRepo.findAll();
        List<UserResponseDto> response = new ArrayList<>();

        for (User user : users) {
            response.add(mapToResponse(user));
        }
        return response;
    }

    @Override
    public UserResponseDto getUserResponseById(Long id) {

        User user = userRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));

        return mapToResponse(user);
    }

    @Override
    public UserResponseDto updateUser(Long id, UpdateRequestDto request) {

        User existingUser = userRepo.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("User not found with id: " + id));

        existingUser.setName(request.getName());
        existingUser.setEmail(request.getEmail());
        existingUser.setMobileNumber(request.getMobileNumber());
        existingUser.setAddress(request.getAddress());

        User updatedUser = userRepo.save(existingUser);

        return mapToResponse(updatedUser);
    }

    @Override
    public void deleteUserById(Long id) {

        if (!userRepo.existsById(id)) {
            throw new ResourceNotFoundException("User not found with id: " + id);
        }

        userRepo.deleteById(id);
    }


    private UserResponseDto mapToResponse(User user) {

        UserResponseDto response = new UserResponseDto();
        response.setId(user.getId());
        response.setName(user.getName());
        response.setEmail(user.getEmail());
        response.setMobileNumber(user.getMobileNumber());
        response.setAddress(user.getAddress());

        return response;
    }

    private User mapToEntity(UserRequestDto request) {

        User user = new User();
        user.setName(request.getName());
        user.setEmail(request.getEmail());
        user.setPassword(request.getPassword()); // hashed later
        user.setMobileNumber(request.getMobileNumber());
        user.setAddress(request.getAddress());

        return user;
    }
}
