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
    const response = await api.post('/auth/user/register', mappedData);
    
    const { token, user } = response.data;
    
    // Store in localStorage
    localStorage.setItem('authToken', token);
    localStorage.setItem('user', JSON.stringify(user));
    
    return {
      token,
      user
    };
  } catch (error) {
    throw error.response?.data?.message || 'Registration failed';
  }
};

// Login user (citizen)
export const login = async (credentials) => {
  try {
    const response = await api.post('/auth/user/login', {
      email: credentials.email,
      password: credentials.password
    });
    
    const { token, user } = response.data;
    
    // Store in localStorage
    localStorage.setItem('authToken', token);
    localStorage.setItem('user', JSON.stringify(user));
    
    return {
      token,
      user
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
