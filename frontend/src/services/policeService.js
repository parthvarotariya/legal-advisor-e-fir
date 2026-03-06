import api from './api';

// Register a new police officer (super admin only)
export const registerPolice = async (policeData) => {
  try {
    const response = await api.post('/auth/police/register', policeData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get all police officers
export const getAllPolice = async () => {
  try {
    const response = await api.get('/police');
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get police by station
export const getPoliceByStation = async (stationId) => {
  try {
    const response = await api.get(`/police/station/${stationId}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get police by role
export const getPoliceByRole = async (role) => {
  try {
    const response = await api.get(`/police/role/${role}`);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Update police officer
export const updatePolice = async (id, updateData) => {
  try {
    const response = await api.put(`/police/${id}`, updateData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Delete police officer
export const deletePolice = async (id) => {
  try {
    await api.delete(`/police/${id}`);
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Get all police stations
export const getAllStations = async () => {
  try {
    const response = await api.get('/police-stations');
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};

// Create a new police station (super admin only)
export const createPoliceStation = async (stationData) => {
  try {
    const response = await api.post('/police-stations/register', stationData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || error.message;
  }
};
