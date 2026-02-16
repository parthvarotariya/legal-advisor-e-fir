import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import './SuperAdminPage.css';

const AdminLoginPage = () => {
  const navigate = useNavigate();
  const [credentials, setCredentials] = useState({ username: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      console.log('Sending admin login request:', credentials);
      
      // Call backend admin login endpoint
      const response = await api.post('/auth/admin/login', {
        username: credentials.username,
        password: credentials.password
      });

      console.log('Admin login response:', response.data);
      
      const { token, admin } = response.data;

      if (!token) {
        throw new Error('No token received');
      }

      // Store token and admin info
      localStorage.setItem('authToken', token);
      localStorage.setItem('adminUser', JSON.stringify(admin));
      
      console.log('Admin login successful');
      navigate('/super-admin');
    } catch (err) {
      console.error('Admin login error:', err);
      setError(err.response?.data?.message || 'Invalid username or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="super-admin-page">
      <div className="admin-container">
        <div className="admin-header">
          <h1>🔒 Admin Login</h1>
          <p>Station Administration Portal</p>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit} className="admin-form">
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={credentials.username}
              onChange={(e) => setCredentials({ ...credentials, username: e.target.value })}
              placeholder="Enter username"
              required
              autoFocus
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={credentials.password}
              onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
              placeholder="Enter password"
              required
            />
          </div>

          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={loading}
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="login-footer">
          <p>Default credentials: admin / admin123</p>
        </div>
      </div>
    </div>
  );
};

export default AdminLoginPage;
