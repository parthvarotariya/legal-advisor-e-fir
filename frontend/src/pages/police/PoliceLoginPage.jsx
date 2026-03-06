import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import api from '../../services/api';
import './PoliceLoginPage.css';

const PoliceLoginPage = () => {
  const navigate = useNavigate();
  const [credentials, setCredentials] = useState({ email: '', password: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      console.log('Attempting login with:', credentials.email);
      
      const response = await api.post('/auth/police/login', credentials);
      console.log('Login response:', response.data);
      const { token, police } = response.data;

      console.log('Extracted token:', token);
      console.log('Token type:', typeof token);

      if (!police) {
        throw new Error('Police officer not found');
      }

      if (!token) {
        throw new Error('No token received from server');
      }

      // Store police session with JWT token
      localStorage.setItem('authToken', token);
      localStorage.setItem('policeUser', JSON.stringify(police));

      console.log('Stored token in localStorage:', localStorage.getItem('authToken'));
      console.log('Police login successful:', police);

      // Redirect based on role
      if (police.role === 'STATION_ADMIN') {
        navigate('/police/pi-dashboard');
      } else if (police.role === 'INVESTIGATING_OFFICER') {
        navigate('/police/psi-dashboard');
      } else if (police.role === 'DEPUTY_SUPRINTENDENT') {
        navigate('/police/dsp-dashboard');
      } else {
        navigate('/police/dashboard');
      }
    } catch (err) {
      console.error('Police login error:', err);
      setError(err.response?.data?.message || err.message || 'Invalid email or password. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="police-login-page">
      <div className="login-container">
        <div className="login-header">
          <div className="police-badge">👮</div>
          <h1>Police Portal</h1>
          <p>E-FIR Management System</p>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label>Email Address</label>
            <input
              type="email"
              value={credentials.email}
              onChange={(e) => setCredentials({ ...credentials, email: e.target.value })}
              placeholder="Enter your email"
              required
            />
          </div>

          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={credentials.password}
              onChange={(e) => setCredentials({ ...credentials, password: e.target.value })}
              placeholder="Enter your password"
              required
            />
          </div>

          <button type="submit" className="btn-login" disabled={loading}>
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <div className="login-footer">
          <Link to="/" className="back-link">← Back to Home</Link>
          <Link to="/admin/login" className="superadmin-link">🔐 Super Admin</Link>
        </div>
      </div>
    </div>
  );
};

export default PoliceLoginPage;
