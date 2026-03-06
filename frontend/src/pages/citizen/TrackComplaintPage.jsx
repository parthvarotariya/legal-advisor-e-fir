import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { trackComplaint, getUserComplaints } from '../../services/complaintService';
import api from '../../services/api';
import { generateFirPdf } from '../../utils/generateFirPdf';
import './TrackComplaintPage.css';

const TrackComplaintPage = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated } = useAuth();
  const [complaintId, setComplaintId] = useState('');
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [userComplaints, setUserComplaints] = useState([]);
  const [filteredComplaints, setFilteredComplaints] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [searchMode, setSearchMode] = useState('all'); // 'all' or 'id'
  const [firData, setFirData] = useState(null);
  const [firLoading, setFirLoading] = useState(false);

  // Load user complaints on mount if authenticated
  useEffect(() => {
    if (isAuthenticated && user?.id) {
      loadUserComplaints();
    }
  }, [isAuthenticated, user]);

  const loadUserComplaints = async () => {
    setLoading(true);
    setError('');
    try {
      console.log('Fetching complaints for user:', user);
      console.log('User ID:', user?.id);
      
      if (!user?.id) {
        setError('User not logged in properly. Please login again.');
        setUserComplaints([]);
        setFilteredComplaints([]);
        setLoading(false);
        return;
      }
      
      const data = await getUserComplaints(user.id);
      console.log('Received complaints:', data);
      setUserComplaints(Array.isArray(data) ? data : []);
      setFilteredComplaints(Array.isArray(data) ? data : []);
      
      // No error if empty array - just show empty state
      if (data.length === 0) {
        console.log('No complaints found for user');
      }
    } catch (err) {
      console.error('Error loading complaints:', err);
      const errorMessage = err?.message || err?.error || 'Failed to load your complaints. Please ensure backend is running.';
      setError(errorMessage);
      setUserComplaints([]);
      setFilteredComplaints([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchById = async (e) => {
    e.preventDefault();
    
    if (!complaintId.trim()) {
      setError('Please enter a complaint ID');
      return;
    }

    setLoading(true);
    setError('');
    setSelectedComplaint(null);

    try {
      const data = await trackComplaint(complaintId.trim());
      setSelectedComplaint(data);
      setSearchMode('id');
      setFirData(null);
      // Auto-fetch FIR if status is FIR_REGISTERED
      if (data.status === 'FIR_REGISTERED' && data.id) {
        fetchFirForComplaint(data.id);
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch complaint. Please check the ID and try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFilterChange = (e) => {
    const searchTerm = e.target.value.toLowerCase();
    if (searchTerm === '') {
      setFilteredComplaints(userComplaints);
    } else {
      const filtered = userComplaints.filter(complaint => 
        complaint.id?.toString().includes(searchTerm) ||
        complaint.actualCategory?.toLowerCase().includes(searchTerm) ||
        complaint.predictedCategory?.toLowerCase().includes(searchTerm) ||
        complaint.status?.toLowerCase().includes(searchTerm) ||
        complaint.description?.toLowerCase().includes(searchTerm)
      );
      setFilteredComplaints(filtered);
    }
  };

  const viewComplaintDetails = (complaint) => {
    setSelectedComplaint(complaint);
    setSearchMode('id');
    setFirData(null);
    // Auto-fetch FIR if status is FIR_REGISTERED
    if (complaint.status === 'FIR_REGISTERED' && complaint.id) {
      fetchFirForComplaint(complaint.id);
    }
  };

  const fetchFirForComplaint = async (complaintId) => {
    setFirLoading(true);
    setFirData(null);
    try {
      const res = await api.get(`/fir/complaint/${complaintId}`);
      setFirData(res.data);
    } catch (err) {
      console.log('No FIR found for complaint:', complaintId);
    } finally {
      setFirLoading(false);
    }
  };

  const getStatusColor = (status) => {
    const statusColors = {
      RECEIVED: '#f59e0b',
      PE_PENDING_DSP_APPROVAL: '#3b82f6',
      PE_ASSIGNED: '#8b5cf6',
      PE_SUBMITTED: '#6366f1',
      FIR_REGISTERED: '#10b981',
      CLOSED_NO_CRIME: '#ef4444'
    };
    return statusColors[status] || '#6b7280';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-IN', {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <div className="track-complaint-page">
      {/* Navigation Bar */}
      <nav className="navbar">
        <div className="nav-brand" onClick={() => navigate('/')}>
          <span className="brand-icon">⚖️</span>
          <span className="brand-name">Legal Advisor e-FIR</span>
        </div>
        <div className="nav-menu">
          <button onClick={() => navigate('/')} className="btn-back">
            ← Back to Home
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <div className="track-complaint-container">
        <div className="track-header">
          <div className="header-icon">🔍</div>
          <h1>Track Your Complaints</h1>
          <p>{isAuthenticated ? 'View all your complaints or search by ID' : 'Enter your complaint ID to view status and details'}</p>
        </div>

        {/* Mode Toggle */}
        {isAuthenticated && (
          <div className="mode-toggle">
            <button 
              className={`toggle-btn ${searchMode === 'all' ? 'active' : ''}`}
              onClick={() => {
                setSearchMode('all');
                setSelectedComplaint(null);
                setError('');
              }}
            >
              📋 My Complaints
            </button>
            <button 
              className={`toggle-btn ${searchMode === 'id' ? 'active' : ''}`}
              onClick={() => {
                setSearchMode('id');
                setError('');
              }}
            >
              🔍 Search by ID
            </button>
          </div>
        )}

        {/* Search by ID Section */}
        {searchMode === 'id' && (
          <div className="search-section">
            <form onSubmit={handleSearchById} className="search-form">
              <div className="input-group">
                <input
                  type="text"
                  placeholder="Enter Complaint ID (e.g., 12345)"
                  value={complaintId}
                  onChange={(e) => setComplaintId(e.target.value)}
                  className="complaint-input"
                />
                <button type="submit" disabled={loading} className="btn-track">
                  {loading ? 'Searching...' : 'Track'}
                </button>
              </div>
              {error && <div className="error-message">{error}</div>}
            </form>
          </div>
        )}

        {/* User Complaints List */}
        {searchMode === 'all' && isAuthenticated && (
          <div className="complaints-list-section">
            <div className="list-header">
              <h2>Your Complaints ({userComplaints.length})</h2>
              <input
                type="text"
                placeholder="Search complaints..."
                onChange={handleFilterChange}
                className="filter-input"
              />
            </div>

            {loading && (
              <div className="loading-state">
                <div className="spinner"></div>
                <p>Loading your complaints...</p>
              </div>
            )}

            {error && <div className="error-message">{error}</div>}

            {!loading && filteredComplaints.length === 0 && !error && (
              <div className="empty-state">
                <div className="empty-icon">📭</div>
                <h3>No Complaints Found</h3>
                <p>You haven't filed any complaints yet.</p>
                <button onClick={() => navigate('/file-complaint')} className="btn-file">
                  File a Complaint
                </button>
              </div>
            )}

            {!loading && filteredComplaints.length > 0 && (
              <div className="complaints-grid">
                {filteredComplaints.map((complaint) => (
                  <div key={complaint.id} className="complaint-card">
                    <div className="card-header">
                      <span className="card-id">#{complaint.id}</span>
                      <span 
                        className="card-status"
                        style={{ backgroundColor: getStatusColor(complaint.status) }}
                      >
                        {complaint.status?.replace(/_/g, ' ')}
                      </span>
                    </div>
                    <div className="card-body">
                      <h3 className="card-title">{complaint.actualCategory || complaint.predictedCategory || 'General Complaint'}</h3>
                      <p className="card-description">
                        {complaint.description?.substring(0, 100)}
                        {complaint.description?.length > 100 ? '...' : ''}
                      </p>
                      <div className="card-meta">
                        <div className="meta-item">
                          <span className="meta-icon">📅</span>
                          <span>{formatDate(complaint.createdAt)}</span>
                        </div>
                        {complaint.policeStation && (
                          <div className="meta-item">
                            <span className="meta-icon">🏛️</span>
                            <span>{complaint.policeStation.stationName}</span>
                          </div>
                        )}
                      </div>
                    </div>
                    <button 
                      onClick={() => viewComplaintDetails(complaint)}
                      className="btn-view-details"
                    >
                      View Details →
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Complaint Details */}
        {selectedComplaint && (
          <div className="complaint-details">
            <div className="status-card">
              <div className="status-header">
                <h2>Complaint Status</h2>
                <span 
                  className="status-badge"
                  style={{ backgroundColor: getStatusColor(selectedComplaint.status) }}
                >
                  {selectedComplaint.status?.replace(/_/g, ' ')}
                </span>
              </div>
            </div>

            <div className="info-grid">
              <div className="info-card">
                <div className="info-icon">📋</div>
                <div className="info-content">
                  <label>Complaint ID</label>
                  <p>{selectedComplaint.id}</p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon">📅</div>
                <div className="info-content">
                  <label>Filed On</label>
                  <p>{formatDate(selectedComplaint.createdAt)}</p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon">👤</div>
                <div className="info-content">
                  <label>Complainant Name</label>
                  <p>{user?.name || 'N/A'}</p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon">📞</div>
                <div className="info-content">
                  <label>Contact</label>
                  <p>{user?.mobileNumber || user?.phone || 'N/A'}</p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon">⚖️</div>
                <div className="info-content">
                  <label>Crime Type</label>
                  <p>{selectedComplaint.actualCategory || selectedComplaint.predictedCategory || 'Not Specified'}</p>
                </div>
              </div>

              <div className="info-card">
                <div className="info-icon">📍</div>
                <div className="info-content">
                  <label>Police Station</label>
                  <p>{selectedComplaint.policeStation?.stationName || 'Not Assigned'}</p>
                </div>
              </div>
            </div>

            <div className="description-section">
              <h3>Complaint Description</h3>
              <div className="description-box">
                <p>{selectedComplaint.description}</p>
              </div>
            </div>

            {selectedComplaint.incidentDate && (
              <div className="incident-details">
                <h3>Incident Details</h3>
                <div className="detail-row">
                  <span className="detail-label">Date of Incident:</span>
                  <span className="detail-value">{formatDate(selectedComplaint.incidentDate)}</span>
                </div>
                {selectedComplaint.incidentLocation && (
                  <div className="detail-row">
                    <span className="detail-label">Location:</span>
                    <span className="detail-value">{selectedComplaint.incidentLocation}</span>
                  </div>
                )}
              </div>
            )}

            {selectedComplaint.assignedOfficer && (
              <div className="officer-info">
                <h3>Assigned Officer</h3>
                <div className="officer-card">
                  <div className="officer-icon">👮</div>
                  <div>
                    <p className="officer-name">{selectedComplaint.assignedOfficer.name}</p>
                    <p className="officer-badge">Badge: {selectedComplaint.assignedOfficer.badgeNumber}</p>
                  </div>
                </div>
              </div>
            )}

            {/* FIR Details Section */}
            {selectedComplaint.status === 'FIR_REGISTERED' && (
              <div className="fir-details-section">
                <h3>📄 FIR Details</h3>
                {firLoading && (
                  <div className="loading-state" style={{ padding: '20px 0' }}>
                    <div className="spinner"></div>
                    <p>Loading FIR details...</p>
                  </div>
                )}
                {!firLoading && !firData && (
                  <div className="info-box" style={{ background: '#fef3cd', border: '1px solid #ffc107', borderRadius: 8, padding: 16, marginTop: 8 }}>
                    <p>FIR has been registered for this complaint. Details are being processed.</p>
                    <button
                      onClick={() => fetchFirForComplaint(selectedComplaint.id)}
                      style={{ marginTop: 8, padding: '6px 16px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 6, cursor: 'pointer' }}
                    >
                      🔄 Retry Loading FIR
                    </button>
                  </div>
                )}
                {!firLoading && firData && (
                  <div className="fir-info-grid">
                    <div className="fir-info-card" style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: 16, marginTop: 8 }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>FIR Number</label>
                          <p style={{ margin: '2px 0 0', fontWeight: 700, fontSize: '1.05rem', color: '#1e40af' }}>{firData.firNumber}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Status</label>
                          <p style={{ margin: '2px 0 0' }}>
                            <span style={{ background: '#10b981', color: '#fff', padding: '2px 10px', borderRadius: 12, fontSize: '0.85rem' }}>{firData.status}</span>
                          </p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>District</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.district || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Registered On</label>
                          <p style={{ margin: '2px 0 0' }}>{formatDate(firData.registeredAt)}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Crime Category</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.crimeCategory || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>IPC / BNS Sections</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.ipcSections || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Police Station</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.policeStationName || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Investigating Officer</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.investigatingOfficerName || 'Not Assigned Yet'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>FIR Written By</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.firWrittenBy || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Incident Location</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.incidentLocation || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Incident Date</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.incidentDate || 'N/A'}</p>
                        </div>
                        <div>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Incident Time</label>
                          <p style={{ margin: '2px 0 0' }}>{firData.incidentTime || 'N/A'}</p>
                        </div>
                      </div>
                      {firData.incidentDescription && (
                        <div style={{ marginTop: 16 }}>
                          <label style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600, textTransform: 'uppercase' }}>Incident Description</label>
                          <div style={{ marginTop: 4, background: '#fff', padding: 12, borderRadius: 6, border: '1px solid #e5e7eb', lineHeight: 1.6, fontSize: '0.95rem' }}>
                            {firData.incidentDescription}
                          </div>
                        </div>
                      )}
                      {firData.isEfir && (
                        <div style={{ marginTop: 12, background: '#f5f3ff', padding: '8px 12px', borderRadius: 6, fontSize: '0.85rem', color: '#6d28d9' }}>
                          📜 This is an e-FIR filed electronically under BNSS 2023
                        </div>
                      )}
                      <div style={{ marginTop: 16, textAlign: 'center' }}>
                        <button
                          onClick={() => generateFirPdf(firData)}
                          style={{
                            padding: '10px 24px',
                            background: 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                            color: '#fff',
                            border: 'none',
                            borderRadius: 8,
                            cursor: 'pointer',
                            fontWeight: 600,
                            fontSize: '0.95rem',
                            boxShadow: '0 2px 8px rgba(59,130,246,0.3)',
                          }}
                        >
                          📄 Download FIR PDF
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="action-buttons">
              <button 
                onClick={() => {
                  if (searchMode === 'all') {
                    setSelectedComplaint(null);
                  } else {
                    setComplaintId('');
                    setSelectedComplaint(null);
                    setError('');
                  }
                }}
                className="btn-new-search"
              >
                {searchMode === 'all' ? '← Back to List' : 'Track Another Complaint'}
              </button>
            </div>
          </div>
        )}

        {/* Info Section */}
        {!selectedComplaint && !loading && searchMode === 'id' && (
          <div className="info-section">
            <div className="info-box">
              <h3>How to Track Your Complaint</h3>
              <ul>
                <li>Enter the complaint ID you received when filing your complaint</li>
                <li>Click the "Track" button to view status and details</li>
                <li>You can check the current status, assigned officer, and other information</li>
              </ul>
            </div>
            <div className="help-box">
              <p>
                <strong>Need Help?</strong> If you've lost your complaint ID, 
                please contact your local police station with your phone number used during filing.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default TrackComplaintPage;
