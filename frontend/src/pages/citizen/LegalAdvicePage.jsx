import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { legalAdviceMap } from '../../data/legalAdviceData';
import { bnsSections } from '../../data/bnsSections';
import './LegalAdvicePage.css';

const LegalAdvicePage = () => {
  const navigate = useNavigate();
  const resultsRef = useRef(null);
  const [complaint, setComplaint] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [serverStatus, setServerStatus] = useState('checking');

  // Scroll to results when analysis is complete
  useEffect(() => {
    if (result && resultsRef.current) {
      const yOffset = -100; // Offset to scroll a bit higher
      const element = resultsRef.current;
      const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
      window.scrollTo({ top: y, behavior: 'smooth' });
    }
  }, [result]);

  const checkServerHealth = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/health', {
        method: 'GET',
      });
      if (response.ok) {
        setServerStatus('online');
        return true;
      }
    } catch (err) {
      setServerStatus('offline');
      return false;
    }
    return false;
  };

  const startServer = () => {
    setServerStatus('starting');
    setError('');
    
    // Show instructions to start server
    setError(
      'Please start the ML API server manually by running this command in a new terminal:\n\n' +
      'cd ml\npython api_server.py\n\n' +
      'Then click "Get Legal Advice" again.'
    );
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!complaint.trim()) {
      setError('Please describe your complaint');
      return;
    }

    if (complaint.trim().length < 20) {
      setError('Please provide more details (at least 20 characters)');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);

    // Check if server is running
    const isServerRunning = await checkServerHealth();
    
    if (!isServerRunning) {
      setLoading(false);
      setError(
        '⚠️ ML API Server is not running!\n\n' +
        'Please start the server by opening a new terminal and running:\n\n' +
        'cd c:\\Users\\Parth\\Desktop\\SDP_Final\\legal-advisor-e-fir\\ml\n' +
        'python api_server.py\n\n' +
        'Wait for "Model loaded successfully" message, then try again.'
      );
      return;
    }

    try {
      const response = await fetch('http://localhost:5000/api/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ complaint: complaint }),
      });

      if (!response.ok) {
        throw new Error('Failed to get classification');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError('Unable to connect to AI service. Please ensure the ML server is running.');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setComplaint('');
    setResult(null);
    setError('');
  };

  return (
    <div className="legal-advice-page">
      {/* Navigation */}
      <nav className="advice-nav">
        <button onClick={() => navigate('/')} className="back-btn">
          ← Back to Home
        </button>
        <h1 className="page-title">AI Legal Advisor</h1>
      </nav>

      <div className="advice-container">
        {/* Left Panel - Input Form */}
        <div className="input-panel">
          <div className="panel-header">
            <div className="header-icon">⚖️</div>
            <h2>Describe Your Complaint</h2>
            <p>Our AI will analyze and provide legal guidance</p>
          </div>

          <form onSubmit={handleSubmit} className="complaint-form">
            <div className="form-field">
              <label className="field-label">Complaint Details *</label>
              <textarea
                className="complaint-input"
                placeholder="Describe your complaint in detail. Include what happened, when, where, and who was involved..."
                value={complaint}
                onChange={(e) => setComplaint(e.target.value)}
                rows={6}
                disabled={loading}
              />
              <div className="char-count">
                {complaint.length} characters {complaint.length >= 20 ? '✓' : '(min 20)'}
              </div>
            </div>

            {error && (
              <div className="error-alert">
                <span className="alert-icon">⚠️</span>
                <pre className="error-message">{error}</pre>
              </div>
            )}

            {serverStatus === 'offline' && !error && (
              <div className="server-alert">
                <span className="alert-icon">🔴</span>
                <div className="alert-content">
                  <strong>ML Server Offline</strong>
                  <p>The AI service is not running. Start it to use this feature.</p>
                  <button 
                    type="button"
                    onClick={startServer}
                    className="start-server-btn"
                  >
                    📋 Show Start Instructions
                  </button>
                </div>
              </div>
            )}

            {serverStatus === 'online' && !error && !result && (
              <div className="success-alert">
                <span className="alert-icon">✓</span>
                <span>ML Server is ready</span>
              </div>
            )}

            <div className="form-actions">
              <button 
                type="submit" 
                className="submit-btn"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <span className="btn-icon">🔍</span>
                    Get Legal Advice
                  </>
                )}
              </button>
              
              {(result || complaint) && (
                <button 
                  type="button" 
                  onClick={handleReset}
                  className="reset-btn"
                  disabled={loading}
                >
                  Clear & Start Over
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Right Panel - Results */}
        <div className="results-panel">
          {!result && !loading && (
            <div className="placeholder">
              <div className="placeholder-icon">🔮</div>
              <h3>Awaiting Your Input</h3>
              <p>Describe your complaint on the left to receive AI-powered legal analysis and guidance</p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <h3>Analyzing Your Complaint</h3>
              <p>Our AI is processing your information...</p>
            </div>
          )}

          {result && (
            <div className="results-content" ref={resultsRef}>
              <div className="result-header">
                <div className="success-badge">
                  <span className="badge-icon">✓</span>
                  Analysis Complete
                </div>
              </div>

              {/* Crime Classification */}
              <div className="result-card classification-card">
                <div className="card-header">
                  <span className="card-icon">🎯</span>
                  <h3>Crime Classification</h3>
                </div>
                <div className="classification-result">
                  <div className="crime-type">{result.category_full}</div>
                  <div className="confidence">
                    Confidence: {result.confidence}%
                  </div>
                </div>
              </div>

              {/* Category Details
              <div className="result-card ipc-card">
                <div className="card-header">
                  <span className="card-icon">📋</span>
                  <h3>Crime Category</h3>
                </div>
                <div className="ipc-content">
                  <div className="category-badge">{result.category}</div>
                  <p className="category-description">{result.category_full}</p>
                </div>
              </div> */}

              {/* Legal Guidance */}
              <div className="result-card guidance-card">
                <div className="card-header">
                  <span className="card-icon">💡</span>
                  <h3>AI Legal Advisory</h3>
                </div>
                <div className="guidance-content">
                  {legalAdviceMap[result.category] && (
                    <>
                      {/* Urgency Badge */}
                      <div className={`urgency-badge urgency-${legalAdviceMap[result.category].urgency.toLowerCase().replace(/\s+/g, '-')}`}>
                        ⚠️ {legalAdviceMap[result.category].urgency}
                      </div>

                      {/* Legal Basis */}
                      <div className="legal-basis-section">
                        <h4>📚 Legal Framework</h4>
                        <div className="legal-info-grid">
                          <div className="legal-info-item">
                            <span className="info-label">Law:</span>
                            <span className="info-value">{legalAdviceMap[result.category].legalBasis.law}</span>
                          </div>
                          <div className="legal-info-item">
                            <span className="info-label">Sections:</span>
                            <span className="info-value">{legalAdviceMap[result.category].legalBasis.sections}</span>
                          </div>
                          <div className="legal-info-item">
                            <span className="info-label">Type:</span>
                            <span className="info-value">{legalAdviceMap[result.category].legalBasis.offenceType}</span>
                          </div>
                          <div className="legal-info-item">
                            <span className="info-label">AI Confidence:</span>
                            <span className={`info-value confidence-${legalAdviceMap[result.category].aiConfidence.toLowerCase()}`}>
                              {legalAdviceMap[result.category].aiConfidence}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Summary */}
                      <div className="summary-section">
                        <h4>📋 Summary</h4>
                        <p className="summary-text">{legalAdviceMap[result.category].summary}</p>
                      </div>

                      {/* Detailed Advice */}
                      <div className="advice-section">
                        <h4>💬 Detailed Guidance</h4>
                        <p className="advice-text">{legalAdviceMap[result.category].detailedAdvice}</p>
                      </div>

                      {/* BNS Sections */}
                      {bnsSections[result.category] && bnsSections[result.category].sections && (
                        <div className="bns-sections-section">
                          <h4>⚖️ Applicable BNS Sections</h4>
                          <div className="bns-sections-list">
                            {bnsSections[result.category].sections.map((section, index) => (
                              <div key={index} className="bns-section-item">
                                <div className="bns-section-header">
                                  <span className="bns-section-number">Section {section.sectionNumber}</span>
                                  <span className="bns-section-title">{section.title}</span>
                                </div>
                                <p className="bns-section-description">{section.description}</p>
                                <div className="bns-section-punishment">
                                  <strong>Punishment:</strong> {section.punishment}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Citizen Rights */}
                      {legalAdviceMap[result.category].citizenRights && (
                        <div className="rights-section">
                          <h4>✅ Your Rights</h4>
                          <ul className="rights-list">
                            {legalAdviceMap[result.category].citizenRights.map((right, index) => (
                              <li key={index}>{right}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Citizen Limitations */}
                      {legalAdviceMap[result.category].citizenLimitations && (
                        <div className="limitations-section">
                          <h4>⚠️ Important Limitations</h4>
                          <ul className="limitations-list">
                            {legalAdviceMap[result.category].citizenLimitations.map((limit, index) => (
                              <li key={index}>{limit}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Steps to Take */}
                      <div className="action-steps">
                        <h4>📝 Steps to Take</h4>
                        <ol className="steps-list">
                          {legalAdviceMap[result.category].stepsToTake.map((step, index) => (
                            <li key={index}>{step}</li>
                          ))}
                        </ol>
                      </div>

                      {/* Evidence Checklist */}
                      {legalAdviceMap[result.category].evidenceChecklist && (
                        <div className="evidence-section">
                          <h4>🔍 Evidence Checklist</h4>
                          <div className="evidence-grid">
                            {legalAdviceMap[result.category].evidenceChecklist.map((evidence, index) => (
                              <div key={index} className="evidence-item">
                                <span className="check-icon">✓</span>
                                {evidence}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Expected Police Action */}
                      {legalAdviceMap[result.category].expectedPoliceAction && (
                        <div className="police-action-section">
                          <h4>👮 Expected Police Action</h4>
                          <ul className="police-action-list">
                            {legalAdviceMap[result.category].expectedPoliceAction.map((action, index) => (
                              <li key={index}>{action}</li>
                            ))}
                          </ul>
                        </div>
                      )}

                      {/* Escalation Path */}
                      {legalAdviceMap[result.category].escalationPath && (
                        <div className="escalation-section">
                          <h4>⚖️ Escalation Path</h4>
                          <p className="escalation-text">{legalAdviceMap[result.category].escalationPath}</p>
                        </div>
                      )}

                      {/* Quick Info Tags */}
                      <div className="info-tags">
                        {legalAdviceMap[result.category].canFileComplaintOnline && (
                          <span className="info-tag online">📱 Online Filing Available</span>
                        )}
                        {legalAdviceMap[result.category].requiresImmediatePoliceAction && (
                          <span className="info-tag urgent">⚡ Immediate Action Required</span>
                        )}
                      </div>
                    </>
                  )}
                </div>
              </div>

              {/* Next Steps */}
              <div className="next-steps">
                <h4>Next Steps</h4>
                <div className="steps-actions">
                  <button 
                    onClick={() => navigate('/file-complaint')}
                    className="action-btn primary-action"
                  >
                    <span className="action-icon">📄</span>
                    File Official Complaint
                  </button>
                  <button 
                    onClick={handleReset}
                    className="action-btn secondary-action"
                  >
                    <span className="action-icon">🔄</span>
                    New Analysis
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LegalAdvicePage;
