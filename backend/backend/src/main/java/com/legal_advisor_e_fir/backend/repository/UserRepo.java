package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.User;
import org.springframework.data.jpa.repository.JpaRepository;

public interface UserRepo extends JpaRepository<User,Long> {

    boolean existsByEmail(String email);

    User findByEmail(String email);

}
