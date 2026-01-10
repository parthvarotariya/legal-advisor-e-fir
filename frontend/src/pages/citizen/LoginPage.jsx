import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import './AuthPage.css';

const LoginPage = () => {
  const navigate = useNavigate();
  const { login } = useAuth();
  
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    if (!formData.email || !formData.password) {
      setError('Please fill in all fields');
      setLoading(false);
      return;
    }

    // Temporary hardcoded login for testing
    if (formData.email === 'admin@gmail.com' && formData.password === 'admin@123') {
      localStorage.setItem('authToken', 'test-token-123');
      localStorage.setItem('user', JSON.stringify({ 
        name: 'Admin User', 
        email: 'admin@gmail.com',
        role: 'CITIZEN'
      }));
      setLoading(false);
      window.location.href = '/';
      return;
    }

    try {
      const result = await login(formData);
      if (result.success) {
        navigate('/');
      } else {
        setError(result.error);
      }
    } catch (err) {
      setError('Invalid email or password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <div className="auth-icon">🔐</div>
            <h1 className="auth-title">Welcome Back</h1>
            <p className="auth-subtitle">Access your account to file complaints and track status</p>
          </div>

          {error && (
            <div className="alert alert-error">
              <span className="alert-icon">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          <form className="auth-form" onSubmit={handleSubmit}>
            <div className="form-group">
              <label className="form-label">Email Address</label>
              <input
                type="email"
                name="email"
                className="form-input"
                placeholder="your@email.com"
                value={formData.email}
                onChange={handleChange}
                required
                autoFocus
              />
            </div>

            <div className="form-group">
              <label className="form-label">Password</label>
              <input
                type="password"
                name="password"
                className="form-input"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                required
              />
            </div>

            <button 
              type="submit" 
              className="btn btn-primary btn-large btn-block"
              disabled={loading}
            >
              {loading ? (
                <span className="spinner"></span>
              ) : (
                'Login'
              )}
            </button>

            <div className="auth-footer">
              <p className="auth-footer-text">
                Don't have an account?{' '}
                <Link to="/register" className="auth-link">Register here</Link>
              </p>
            </div>
          </form>
        </div>

        <div className="auth-aside">
          <div className="aside-content">
            <div className="aside-icon">⚖️</div>
            <h2 className="aside-title">Legal Advisor e-FIR</h2>
            <p className="aside-text">
              Access AI-powered legal guidance and file complaints online with our secure platform
            </p>
            <div className="aside-features">
              <div className="feature-item">
                <span className="feature-icon">✓</span>
                <span>AI-Assisted Guidance</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">✓</span>
                <span>24/7 Complaint Filing</span>
              </div>
              <div className="feature-item">
                <span className="feature-icon">✓</span>
                <span>Real-time Tracking</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;
