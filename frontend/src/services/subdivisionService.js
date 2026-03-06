import api from './api';

/**
 * Service for subdivision management operations.
 * Handles all subdivision-related API calls.
 */

// Create a new subdivision
export const createSubdivision = async (subdivisionData) => {
  try {
    const response = await api.post('/subdivisions/register', subdivisionData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get all subdivisions
export const getAllSubdivisions = async () => {
  try {
    const response = await api.get('/subdivisions');
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get subdivision by ID
export const getSubdivisionById = async (id) => {
  try {
    const response = await api.get(`/subdivisions/${id}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get subdivision by code
export const getSubdivisionByCode = async (code) => {
  try {
    const response = await api.get(`/subdivisions/code/${code}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get subdivisions by district
export const getSubdivisionsByDistrict = async (district) => {
  try {
    const response = await api.get(`/subdivisions/district/${district}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get subdivisions by state
export const getSubdivisionsByState = async (state) => {
  try {
    const response = await api.get(`/subdivisions/state/${state}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Update subdivision details
export const updateSubdivision = async (id, subdivisionData) => {
  try {
    const response = await api.put(`/subdivisions/${id}`, subdivisionData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Assign DSP officer to subdivision
export const assignDspOfficer = async (subdivisionId, dspOfficerId) => {
  try {
    const response = await api.put(`/subdivisions/${subdivisionId}/assign-dsp`, {
      dspOfficerId: dspOfficerId
    });
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Remove DSP officer from subdivision
export const removeDspOfficer = async (subdivisionId) => {
  try {
    const response = await api.put(`/subdivisions/${subdivisionId}/remove-dsp`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Add police station to subdivision
export const addStationToSubdivision = async (subdivisionId, policeStationId) => {
  try {
    const response = await api.put(`/subdivisions/${subdivisionId}/add-station`, {
      policeStationId: policeStationId
    });
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Remove police station from subdivision
export const removeStationFromSubdivision = async (subdivisionId, policeStationId) => {
  try {
    const response = await api.put(`/subdivisions/${subdivisionId}/remove-station`, {
      policeStationId: policeStationId
    });
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Delete subdivision
export const deleteSubdivision = async (id) => {
  try {
    await api.delete(`/subdivisions/${id}`);
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};
