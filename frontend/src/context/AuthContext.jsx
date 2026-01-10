import { createContext, useContext, useState, useEffect } from 'react';
import { login as loginService, logout as logoutService, getCurrentUser, isAuthenticated } from '../services/authService';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Check if user is logged in on mount
  useEffect(() => {
    const currentUser = getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
    }
    setLoading(false);
  }, []);

  const login = async (credentials) => {
    try {
      const data = await loginService(credentials);
      setUser(data.user);
      return { success: true };
    } catch (error) {
      return { success: false, error };
    }
  };

  const logout = () => {
    logoutService();
    setUser(null);
  };

  const value = {
    user,
    login,
    logout,
    isAuthenticated: isAuthenticated(),
    loading
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

// Custom hook to use auth context
export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within AuthProvider');
  }
  return context;
};
