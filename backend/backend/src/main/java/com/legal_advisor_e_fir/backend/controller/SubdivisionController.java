package com.legal_advisor_e_fir.backend.controller;

import com.legal_advisor_e_fir.backend.dto.SubdivisionRequestDto;
import com.legal_advisor_e_fir.backend.dto.SubdivisionResponseDto;
import com.legal_advisor_e_fir.backend.service.ISubdivisionService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * REST Controller for Subdivision management.
 * Handles all subdivision-related HTTP requests.
 */
@RestController
@RequestMapping("/api/subdivisions")
public class SubdivisionController {

    private final ISubdivisionService subdivisionService;

    @Autowired
    public SubdivisionController(ISubdivisionService subdivisionService) {
        this.subdivisionService = subdivisionService;
    }

    /**
     * Create a new subdivision.
     *
     * @param requestDto Subdivision details
     * @return Created subdivision response
     */
    @PostMapping("/register")
    public ResponseEntity<SubdivisionResponseDto> createSubdivision(
            @Valid @RequestBody SubdivisionRequestDto requestDto
    ) {
        SubdivisionResponseDto response = subdivisionService.createSubdivision(requestDto);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /**
     * Get all subdivisions.
     *
     * @return List of all subdivisions
     */
    @GetMapping
    public ResponseEntity<List<SubdivisionResponseDto>> getAllSubdivisions() {
        List<SubdivisionResponseDto> subdivisions = subdivisionService.getAllSubdivisions();
        return ResponseEntity.ok(subdivisions);
    }

    /**
     * Get subdivision by ID.
     *
     * @param id Subdivision ID
     * @return Subdivision details
     */
    @GetMapping("/{id}")
    public ResponseEntity<SubdivisionResponseDto> getSubdivisionById(@PathVariable Long id) {
        SubdivisionResponseDto subdivision = subdivisionService.getById(id);
        return ResponseEntity.ok(subdivision);
    }

    /**
     * Get subdivision by code.
     *
     * @param code Subdivision code
     * @return Subdivision details
     */
    @GetMapping("/code/{code}")
    public ResponseEntity<SubdivisionResponseDto> getSubdivisionByCode(@PathVariable String code) {
        SubdivisionResponseDto subdivision = subdivisionService.getByCode(code);
        return ResponseEntity.ok(subdivision);
    }

    /**
     * Get subdivisions by district.
     *
     * @param district District name
     * @return List of subdivisions in the district
     */
    @GetMapping("/district/{district}")
    public ResponseEntity<List<SubdivisionResponseDto>> getSubdivisionsByDistrict(
            @PathVariable String district
    ) {
        List<SubdivisionResponseDto> subdivisions = subdivisionService.getByDistrict(district);
        return ResponseEntity.ok(subdivisions);
    }

    /**
     * Get subdivisions by state.
     *
     * @param state State name
     * @return List of subdivisions in the state
     */
    @GetMapping("/state/{state}")
    public ResponseEntity<List<SubdivisionResponseDto>> getSubdivisionsByState(
            @PathVariable String state
    ) {
        List<SubdivisionResponseDto> subdivisions = subdivisionService.getByState(state);
        return ResponseEntity.ok(subdivisions);
    }

    /**
     * Update subdivision details.
     *
     * @param id         Subdivision ID
     * @param requestDto Updated subdivision details
     * @return Updated subdivision response
     */
    @PutMapping("/{id}")
    public ResponseEntity<SubdivisionResponseDto> updateSubdivision(
            @PathVariable Long id,
            @Valid @RequestBody SubdivisionRequestDto requestDto
    ) {
        SubdivisionResponseDto response = subdivisionService.updateSubdivision(id, requestDto);
        return ResponseEntity.ok(response);
    }

    /**
     * Assign a DSP officer to a subdivision.
     *
     * @param id      Subdivision ID
     * @param payload Request body containing dspOfficerId
     * @return Updated subdivision response
     */
    @PutMapping("/{id}/assign-dsp")
    public ResponseEntity<SubdivisionResponseDto> assignDspOfficer(
            @PathVariable Long id,
            @RequestBody Map<String, Long> payload
    ) {
        Long dspOfficerId = payload.get("dspOfficerId");
        if (dspOfficerId == null) {
            return ResponseEntity.badRequest().build();
        }
        SubdivisionResponseDto response = subdivisionService.assignDspOfficer(id, dspOfficerId);
        return ResponseEntity.ok(response);
    }

    /**
     * Remove DSP officer from a subdivision.
     *
     * @param id Subdivision ID
     * @return Updated subdivision response
     */
    @PutMapping("/{id}/remove-dsp")
    public ResponseEntity<SubdivisionResponseDto> removeDspOfficer(@PathVariable Long id) {
        SubdivisionResponseDto response = subdivisionService.removeDspOfficer(id);
        return ResponseEntity.ok(response);
    }

    /**
     * Add a police station to a subdivision's jurisdiction.
     *
     * @param id      Subdivision ID
     * @param payload Request body containing policeStationId
     * @return Updated subdivision response
     */
    @PutMapping("/{id}/add-station")
    public ResponseEntity<SubdivisionResponseDto> addPoliceStation(
            @PathVariable Long id,
            @RequestBody Map<String, Long> payload
    ) {
        Long policeStationId = payload.get("policeStationId");
        if (policeStationId == null) {
            return ResponseEntity.badRequest().build();
        }
        SubdivisionResponseDto response = subdivisionService.addPoliceStation(id, policeStationId);
        return ResponseEntity.ok(response);
    }

    /**
     * Remove a police station from a subdivision's jurisdiction.
     *
     * @param id      Subdivision ID
     * @param payload Request body containing policeStationId
     * @return Updated subdivision response
     */
    @PutMapping("/{id}/remove-station")
    public ResponseEntity<SubdivisionResponseDto> removePoliceStation(
            @PathVariable Long id,
            @RequestBody Map<String, Long> payload
    ) {
        Long policeStationId = payload.get("policeStationId");
        if (policeStationId == null) {
            return ResponseEntity.badRequest().build();
        }
        SubdivisionResponseDto response = subdivisionService.removePoliceStation(id, policeStationId);
        return ResponseEntity.ok(response);
    }

    /**
     * Delete a subdivision.
     *
     * @param id Subdivision ID
     * @return No content response
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteSubdivision(@PathVariable Long id) {
        subdivisionService.deleteSubdivision(id);
        return ResponseEntity.noContent().build();
    }
}
