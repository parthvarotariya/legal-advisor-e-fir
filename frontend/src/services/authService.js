import api from './api';

// Register a new user (citizen)
export const register = async (userData) => {
  try {
    // Map frontend fields to backend DTO fields
    const mappedData = {
      name: userData.name,
      email: userData.email,
      mobileNumber: userData.phone,
      password: userData.password,
      address: userData.address
    };
    const response = await api.post('/users/register', mappedData);
    return response.data;
  } catch (error) {
    throw error.response?.data?.message || 'Registration failed';
  }
};

// Login user (mock authentication - no backend endpoint available)
export const login = async (credentials) => {
  try {
    // Since backend has no login endpoint, simulate login
    // In production, this should call backend API
    
    // For now, just create a mock user session
    const mockUser = {
      id: 1,
      name: credentials.email.split('@')[0],
      email: credentials.email,
      mobileNumber: '9999999999'
    };
    
    // Store in localStorage
    localStorage.setItem('authToken', 'mock-token-' + Date.now());
    localStorage.setItem('user', JSON.stringify(mockUser));
    
    return {
      token: localStorage.getItem('authToken'),
      user: mockUser
    };
  } catch (error) {
    throw error.response?.data?.message || 'Login failed';
  }
};

// Logout user
export const logout = () => {
  localStorage.removeItem('authToken');
  localStorage.removeItem('user');
};

// Get current user from localStorage
export const getCurrentUser = () => {
  const userStr = localStorage.getItem('user');
  return userStr ? JSON.parse(userStr) : null;
};

// Check if user is authenticated
export const isAuthenticated = () => {
  return !!localStorage.getItem('authToken');
};
