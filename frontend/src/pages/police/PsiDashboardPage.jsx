import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import './PoliceDashboardPage.css';

const CRIME_CATEGORIES = [
  'kidnapping',
  'sexual_offence', 
  'assault',
  'women_child_safety',
  'harassment',
  'accident',
  'cybercrime',
  'fraud',
  'theft',
  'trespass',
  'defamation',
  'other'
];

const PsiDashboardPage = () => {
  const navigate = useNavigate();
  const [officerInfo, setOfficerInfo] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('authToken');
      const policeUser = localStorage.getItem('policeUser');
      
      if (!token || !policeUser) {
        navigate('/police/login');
        return;
      }

      const officer = JSON.parse(policeUser);
      
      // Only PSI can access this page
      if (officer.role !== 'INVESTIGATING_OFFICER') {
        navigate('/police/pi-dashboard');
        return;
      }

      setOfficerInfo(officer);
      
      // Extract stationId
      const stationId = officer.stationId || officer.station?.id;
      if (stationId) {
        fetchAssignedComplaints(stationId, officer.policeId);
      } else {
        setError('No station assigned to this officer');
        setLoading(false);
      }
    };

    checkAuth();
  }, [navigate]);

  const fetchAssignedComplaints = async (stationId, officerId) => {
    try {
      setLoading(true);
      const response = await api.get(`/complaints/station/${stationId}`);
      
      // Filter to show only complaints assigned to this officer
      const assignedComplaints = response.data.filter(
        complaint => complaint.assignedOfficerId === officerId
      );
      
      setComplaints(assignedComplaints);
      setError('');
    } catch (err) {
      console.error('Error fetching complaints:', err);
      setError('Failed to load complaints');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('policeUser');
    navigate('/police/login');
  };

  const handleUpdateStatus = async (complaintId, newStatus, actualCategory = null) => {
    try {
      const payload = { status: newStatus };
      if (actualCategory) {
        payload.actualCategory = actualCategory;
      }

      await api.put(`/complaints/${complaintId}/status`, payload);
      
      // Update local state
      setComplaints(prevComplaints =>
        prevComplaints.map(c =>
          c.id === complaintId 
            ? { ...c, status: newStatus, ...(actualCategory && { actualCategory }) }
            : c
        )
      );

      if (selectedComplaint?.id === complaintId) {
        setSelectedComplaint(prev => ({
          ...prev,
          status: newStatus,
          ...(actualCategory && { actualCategory })
        }));
      }

      alert('Status updated successfully');
    } catch (err) {
      console.error('Error updating status:', err);
      alert('Failed to update status');
    }
  };

  const handleApprovePredicted = async (complaintId, predictedCategory) => {
    if (!predictedCategory) {
      alert('No predicted category available');
      return;
    }
    await handleUpdateStatus(complaintId, null, predictedCategory);
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusBadge = (status) => {
    const statusMap = {
      'PENDING': 'status-pending',
      'READ': 'status-read',
      'UNDER_REVIEW': 'status-review',
      'CLOSED': 'status-closed'
    };
    return statusMap[status] || '';
  };

  if (loading) {
    return (
      <div className="dashboard-container">
        <div className="loading">Loading your assigned complaints...</div>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div className="header-content">
          <div className="header-left">
            <h1>PSI Dashboard</h1>
            {officerInfo && (
              <div className="officer-info">
                <span className="officer-name">👮 {officerInfo.name}</span>
                <span className="officer-badge">Badge: {officerInfo.badgeNumber}</span>
                <span className="officer-role">Role: Investigating Officer</span>
              </div>
            )}
          </div>
          <button className="logout-btn" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </header>

      <main className="dashboard-main">
        {error && <div className="error-message">{error}</div>}

        <div className="dashboard-stats">
          <div className="stat-card">
            <h3>My Assigned Complaints</h3>
            <p className="stat-number">{complaints.length}</p>
          </div>
          <div className="stat-card">
            <h3>Pending</h3>
            <p className="stat-number">
              {complaints.filter(c => c.status === 'PENDING').length}
            </p>
          </div>
          <div className="stat-card">
            <h3>Under Review</h3>
            <p className="stat-number">
              {complaints.filter(c => c.status === 'UNDER_REVIEW').length}
            </p>
          </div>
          <div className="stat-card">
            <h3>Closed</h3>
            <p className="stat-number">
              {complaints.filter(c => c.status === 'CLOSED').length}
            </p>
          </div>
        </div>

        <div className="complaints-section">
          <h2>My Assigned Cases</h2>
          {complaints.length === 0 ? (
            <div className="no-complaints">
              <p>No complaints assigned to you yet.</p>
            </div>
          ) : (
            <div className="table-container">
              <table className="complaints-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Description</th>
                    <th>Status</th>
                    <th>Filed On</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {complaints.map(complaint => (
                    <tr key={complaint.id}>
                      <td><span className="complaint-id">#{complaint.id}</span></td>
                      <td>
                        <span className="category-badge">
                          {complaint.actualCategory || complaint.predictedCategory || 'Uncategorized'}
                        </span>
                      </td>
                      <td className="description-cell">
                        {complaint.description?.substring(0, 60)}
                        {complaint.description?.length > 60 ? '...' : ''}
                      </td>
                      <td>
                        <span className={`status-badge ${getStatusBadge(complaint.status)}`}>
                          {complaint.status}
                        </span>
                      </td>
                      <td>{formatDate(complaint.createdAt)}</td>
                      <td>
                        <button 
                          className="view-btn"
                          onClick={() => setSelectedComplaint(complaint)}
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>

      {/* Complaint Detail Modal */}
      {selectedComplaint && (
        <div className="modal-overlay" onClick={() => setSelectedComplaint(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Complaint Details #{selectedComplaint.id}</h2>
              <button className="close-btn" onClick={() => setSelectedComplaint(null)}>
                ×
              </button>
            </div>
            
            <div className="modal-body">
              <div className="detail-row">
                <label>Status:</label>
                <span className={`status-badge ${getStatusBadge(selectedComplaint.status)}`}>
                  {selectedComplaint.status}
                </span>
              </div>
              
              <div className="detail-row">
                <label>Predicted Category:</label>
                <span className="category-badge">
                  {selectedComplaint.predictedCategory || 'N/A'}
                </span>
              </div>

              {selectedComplaint.actualCategory && (
                <div className="detail-row">
                  <label>Approved Category:</label>
                  <span className="category-badge" style={{backgroundColor: '#27ae60', color: 'white'}}>
                    ✓ {selectedComplaint.actualCategory}
                  </span>
                </div>
              )}

              <div className="detail-row">
                <label>Description:</label>
                <p className="detail-description">{selectedComplaint.description}</p>
              </div>

              <div className="detail-row">
                <label>Police Station:</label>
                <span>{selectedComplaint.policeStationName}</span>
              </div>

              <div className="detail-row">
                <label>Filed On:</label>
                <span>{formatDate(selectedComplaint.createdAt)}</span>
              </div>

              {/* Category Approval Section */}
              {!selectedComplaint.actualCategory && (
                <div className="detail-row category-approval-section">
                  <label>Approve Category:</label>
                  <div style={{display: 'flex', gap: '10px', flex: 1, alignItems: 'center'}}>
                    <button
                      onClick={() => handleApprovePredicted(selectedComplaint.id, selectedComplaint.predictedCategory)}
                      className="approve-predicted-btn"
                      disabled={!selectedComplaint.predictedCategory}
                    >
                      ✓ Approve Predicted
                    </button>
                    <span style={{color: '#7f8c8d'}}>or</span>
                    <select 
                      id="categoryDropdown"
                      defaultValue=""
                      style={{
                        padding: '8px',
                        border: '1px solid #ddd',
                        borderRadius: '4px',
                        flex: 1
                      }}
                    >
                      <option value="">-- Select Category --</option>
                      {CRIME_CATEGORIES.map(cat => (
                        <option key={cat} value={cat}>{cat}</option>
                      ))}
                    </select>
                    <button
                      onClick={() => {
                        const dropdown = document.getElementById('categoryDropdown');
                        const category = dropdown.value;
                        if (category) {
                          handleUpdateStatus(selectedComplaint.id, null, category);
                        } else {
                          alert('Please select a category');
                        }
                      }}
                      className="select-category-btn"
                    >
                      Select
                    </button>
                  </div>
                </div>
              )}

              {/* Status Update Section */}
              <div className="detail-row">
                <label>Update Status:</label>
                <div style={{display: 'flex', gap: '10px'}}>
                  <button
                    onClick={() => handleUpdateStatus(selectedComplaint.id, 'UNDER_REVIEW')}
                    className="status-update-btn review"
                    disabled={selectedComplaint.status === 'UNDER_REVIEW'}
                  >
                    Mark Under Review
                  </button>
                  <button
                    onClick={() => handleUpdateStatus(selectedComplaint.id, 'CLOSED')}
                    className="status-update-btn closed"
                    disabled={selectedComplaint.status === 'CLOSED'}
                  >
                    Mark Closed
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PsiDashboardPage;
