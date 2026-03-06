import api from './api';

// Helper function to extract complaints from potentially corrupted JSON string
const extractComplaints = (data) => {
  if (Array.isArray(data)) {
    return data;
  }
  
  if (typeof data === 'string') {
    try {
      // Try to parse as JSON first
      const parsed = JSON.parse(data);
      if (Array.isArray(parsed)) {
        return parsed;
      }
      return [parsed];
    } catch (e) {
      console.log('JSON parse failed, trying regex extraction:', e.message);
    }
    
    try {
      // Find all complaint IDs first
      const idMatches = [...data.matchAll(/"id":(\d+)/g)];
      console.log('Found', idMatches.length, 'id fields in response');
      
      const extractedComplaints = [];
      const seenIds = new Set();
      
      for (const idMatch of idMatches) {
        const id = parseInt(idMatch[1]);
        const startPos = idMatch.index;
        
        // Skip if we've already processed this ID
        if (seenIds.has(id)) continue;
        seenIds.add(id);
        
        // Extract a chunk around this ID (look ahead ~2000 chars for the complaint data)
        const chunk = data.substring(startPos, startPos + 2000);
        
        const descMatch = chunk.match(/"description":"([^"]*)"/);
        const actualCatMatch = chunk.match(/"actualCategory":(null|"[^"]*")/);
        const predCatMatch = chunk.match(/"predictedCategory":"([^"]*)"/);
        const createdMatch = chunk.match(/"createdAt":"([^"]*)"/);
        const statusMatch = chunk.match(/"status":"(\w+)"/);
        
        // Only add if we found the description (confirms it's a complaint object, not nested id)
        if (descMatch) {
          console.log('Extracting complaint ID:', id);
          extractedComplaints.push({
            id: id,
            description: descMatch[1],
            actualCategory: actualCatMatch ? (actualCatMatch[1] === 'null' ? null : actualCatMatch[1].replace(/"/g, '')) : null,
            predictedCategory: predCatMatch ? predCatMatch[1] : '',
            createdAt: createdMatch ? createdMatch[1] : '',
            status: statusMatch ? statusMatch[1] : 'RECEIVED'
          });
        }
      }
      
      console.log('Extracted', extractedComplaints.length, 'complaints from string');
      return extractedComplaints;
    } catch (e) {
      console.error('Failed to extract complaints:', e);
    }
  }
  
  return [];
};

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
    // Handle potentially corrupted JSON due to circular references
    const complaints = extractComplaints(response.data);
    if (complaints.length > 0) {
      return complaints[0];
    }
    // If extraction failed, try using response.data directly (if it's already an object)
    if (response.data && typeof response.data === 'object' && !Array.isArray(response.data)) {
      return response.data;
    }
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

// Get all complaints for a user by user ID
export const getUserComplaints = async (userId) => {
  try {
    const response = await api.get(`/complaints/users/${userId}`);
    // Handle potentially corrupted JSON due to circular references
    const complaints = extractComplaints(response.data);
    return complaints;
  } catch (error) {
    if (error.response?.status === 404 || error.response?.status === 204) {
      return [];
    }
    throw error.response?.data || error.message;
  }
};

// Get complaints by station ID (for police officers)
export const getComplaintsByStation = async (stationId) => {
  try {
    const response = await api.get(`/complaints/station/${stationId}`);
    // Handle potentially corrupted JSON due to circular references
    const complaints = extractComplaints(response.data);
    return complaints;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};

// Update complaint status
export const updateComplaintStatus = async (complaintId, status) => {
  try {
    const response = await api.put(`/complaints/${complaintId}/status`, { status });
    return response.data;
  } catch (error) {
    throw error.response?.data || error.message;
  }
};
