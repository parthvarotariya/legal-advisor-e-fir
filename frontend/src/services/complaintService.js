import api from './api';

// Submit a new complaint (citizen side - no auth required)
export const submitComplaint = async (complaintData) => {
  try {
    const response = await api.post('/complaints', complaintData);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

// Track complaint by complaint ID
export const trackComplaint = async (complaintId) => {
  try {
    const response = await api.get(`/complaints/${complaintId}`);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

// Get complaint status
export const getComplaintStatus = async (complaintId) => {
  try {
    const response = await api.get(`/complaints/${complaintId}/status`);
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};
