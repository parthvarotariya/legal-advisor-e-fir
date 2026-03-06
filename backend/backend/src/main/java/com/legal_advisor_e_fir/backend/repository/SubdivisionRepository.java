package com.legal_advisor_e_fir.backend.repository;

import com.legal_advisor_e_fir.backend.model.Subdivision;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository interface for Subdivision entity.
 * Provides database access methods for subdivision management.
 */
@Repository
public interface SubdivisionRepository extends JpaRepository<Subdivision, Long> {

    /**
     * Find subdivision by its unique code.
     *
     * @param subdivisionCode Unique subdivision code
     * @return Optional containing the subdivision if found
     */
    Optional<Subdivision> findBySubdivisionCode(String subdivisionCode);

    /**
     * Find all subdivisions in a specific district.
     *
     * @param district District name
     * @return List of subdivisions in the district
     */
    List<Subdivision> findByDistrict(String district);

    /**
     * Find all subdivisions in a specific state.
     *
     * @param state State name
     * @return List of subdivisions in the state
     */
    List<Subdivision> findByState(String state);

    /**
     * Find subdivisions by name (case-insensitive partial match).
     *
     * @param subdivisionName Subdivision name to search
     * @return List of matching subdivisions
     */
    List<Subdivision> findBySubdivisionNameContainingIgnoreCase(String subdivisionName);

    /**
     * Find subdivision by the DSP officer assigned to it.
     *
     * @param dspOfficerId ID of the DSP officer
     * @return Optional containing the subdivision if found
     */
    Optional<Subdivision> findByDspOfficer_PoliceId(Long dspOfficerId);

    /**
     * Check if a subdivision code already exists.
     *
     * @param subdivisionCode Subdivision code to check
     * @return true if exists, false otherwise
     */
    boolean existsBySubdivisionCode(String subdivisionCode);
}
