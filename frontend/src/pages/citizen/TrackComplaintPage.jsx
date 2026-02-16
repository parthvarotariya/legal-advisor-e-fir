import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { trackComplaint, getUserComplaints } from '../../services/complaintService';
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
  };

  const getStatusColor = (status) => {
    const statusColors = {
      PENDING: '#f59e0b',
      UNDER_INVESTIGATION: '#3b82f6',
      RESOLVED: '#10b981',
      REJECTED: '#ef4444',
      CLOSED: '#6b7280'
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
