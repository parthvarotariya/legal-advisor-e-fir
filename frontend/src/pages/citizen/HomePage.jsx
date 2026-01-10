import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import './HomePage.css';

const HomePage = () => {
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuth();

  return (
    <div className="homepage">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="nav-brand">
          <span className="brand-icon">⚖️</span>
          <span className="brand-name">Legal Advisor e-FIR</span>
        </div>
        <div className="nav-menu">
          {isAuthenticated ? (
            <>
              <span className="user-welcome">Hi, {user?.name}</span>
              <button onClick={logout} className="btn-logout">Logout</button>
            </>
          ) : (
            <>
              <button onClick={() => navigate('/login')} className="btn-login">Login</button>
              <button onClick={() => navigate('/register')} className="btn-register">Sign Up</button>
            </>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-icon-large">⚖️</div>
          <h1 className="hero-title">AI-Powered Legal Advisor & e-FIR System</h1>
          <p className="hero-description">
            File complaints online, get AI-assisted legal guidance, and track your case status in real-time
          </p>
          <div className="hero-badge">
            <span className="badge-icon">✓</span>
            <span>Section 154 CrPC Compliant</span>
          </div>
        </div>
      </section>

      {/* Services Section */}
      <section className="services-section">
        <h2 className="section-heading">What We Offer</h2>
        <p className="section-subheading">Choose a service to get started</p>
        
        <div className="services-grid">
          {/* Service Card 1 */}
          <div className="service-card card-purple">
            <div className="card-header">
              <div className="service-icon">⚖️</div>
              <h3 className="service-title">Legal Advice</h3>
            </div>
            <p className="service-desc">
              Get instant AI-powered crime classification and legal guidance for your complaint
            </p>
            <button 
              onClick={() => navigate('/legal-advice')} 
              className="service-btn"
              disabled={!isAuthenticated}
            >
              {isAuthenticated ? 'Get Advice' : 'Login Required'}
            </button>
          </div>

          {/* Service Card 2 */}
          <div className="service-card card-green">
            <div className="card-header">
              <div className="service-icon">📄</div>
              <h3 className="service-title">File Complaint</h3>
            </div>
            <p className="service-desc">
              Submit your complaint online. Police will review and register FIR if required
            </p>
            <button 
              onClick={() => navigate('/file-complaint')} 
              className="service-btn"
              disabled={!isAuthenticated}
            >
              {isAuthenticated ? 'File Now' : 'Login Required'}
            </button>
          </div>

          {/* Service Card 3 */}
          <div className="service-card card-blue">
            <div className="card-header">
              <div className="service-icon">🔍</div>
              <h3 className="service-title">Track Status</h3>
            </div>
            <p className="service-desc">
              Monitor your complaint status and get real-time updates on your case
            </p>
            <button 
              onClick={() => navigate('/track-complaint')} 
              className="service-btn"
            >
              Track Now
            </button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="features-container">
          <div className="feature-item">
            <div className="feature-icon">🤖</div>
            <div className="feature-content">
              <h4 className="feature-title">AI-Powered Classification</h4>
              <p className="feature-text">Automatic crime categorization using advanced machine learning</p>
            </div>
          </div>
          
          <div className="feature-item">
            <div className="feature-icon">⚡</div>
            <div className="feature-content">
              <h4 className="feature-title">Fast & Secure</h4>
              <p className="feature-text">Quick complaint filing with end-to-end encryption</p>
            </div>
          </div>
          
          <div className="feature-item">
            <div className="feature-icon">📊</div>
            <div className="feature-content">
              <h4 className="feature-title">Real-Time Tracking</h4>
              <p className="feature-text">Monitor your case progress at every step</p>
            </div>
          </div>
          
          <div className="feature-item">
            <div className="feature-icon">👮</div>
            <div className="feature-content">
              <h4 className="feature-title">Police Integration</h4>
              <p className="feature-text">Direct connection with law enforcement authorities</p>
            </div>
          </div>
        </div>
      </section>

      {/* Info Banner */}
      <section className="info-banner">
        <div className="info-content">
          <h3 className="info-heading">Important Guidelines</h3>
          <div className="info-grid">
            <div className="info-point">
              <span className="check-icon">✓</span>
              <p>System compliant with Section 154 of the Criminal Procedure Code (CrPC)</p>
            </div>
            <div className="info-point">
              <span className="check-icon">✓</span>
              <p>Citizens file complaints - Police officers register official FIR</p>
            </div>
            <div className="info-point">
              <span className="check-icon">✓</span>
              <p>AI provides guidance only - Final authority rests with police</p>
            </div>
            <div className="info-point">
              <span className="check-icon">✓</span>
              <p>Track your complaint status anytime, anywhere</p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="footer">
        <p className="footer-text">© 2026 Legal Advisor e-FIR System. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default HomePage;
