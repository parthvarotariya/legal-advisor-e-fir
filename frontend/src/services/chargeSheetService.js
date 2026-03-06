import api from './api';

// ── IO (PSI) Actions ──

export const createChargeSheet = async (data) => {
  const response = await api.post('/charge-sheet/create', data);
  return response.data;
};

export const updateChargeSheet = async (id, data) => {
  const response = await api.put(`/charge-sheet/${id}`, data);
  return response.data;
};

export const submitChargeSheetToPI = async (id) => {
  const response = await api.put(`/charge-sheet/${id}/submit`);
  return response.data;
};

// ── PI Actions ──

export const approveChargeSheet = async (id, approvingOfficerId) => {
  const response = await api.put(`/charge-sheet/${id}/approve?approvingOfficerId=${approvingOfficerId}`);
  return response.data;
};

export const returnChargeSheet = async (id, approvingOfficerId, suggestions) => {
  const response = await api.put(`/charge-sheet/${id}/return`, {
    approvingOfficerId,
    suggestions
  });
  return response.data;
};

export const dispatchChargeSheet = async (id) => {
  const response = await api.put(`/charge-sheet/${id}/dispatch`);
  return response.data;
};

// ── Queries ──

export const getChargeSheetById = async (id) => {
  const response = await api.get(`/charge-sheet/${id}`);
  return response.data;
};

export const getChargeSheetsByFir = async (firId) => {
  const response = await api.get(`/charge-sheet/fir/${firId}`);
  return response.data;
};

export const getChargeSheetsByStation = async (stationId) => {
  const response = await api.get(`/charge-sheet/station/${stationId}`);
  return response.data;
};

export const getPendingChargeSheetsByStation = async (stationId) => {
  const response = await api.get(`/charge-sheet/station/${stationId}/pending`);
  return response.data;
};

export const getChargeSheetsByOfficer = async (policeId) => {
  const response = await api.get(`/charge-sheet/officer/${policeId}`);
  return response.data;
};

export const getChargeSheetsByOfficerAndStatus = async (policeId, status) => {
  const response = await api.get(`/charge-sheet/officer/${policeId}/status/${status}`);
  return response.data;
};

export const deleteChargeSheet = async (id) => {
  await api.delete(`/charge-sheet/${id}`);
};
