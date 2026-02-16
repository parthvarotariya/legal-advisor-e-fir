import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import './PoliceDashboardPage.css';

const PoliceDashboardPage = () => {
  const navigate = useNavigate();
  const [policeUser, setPoliceUser] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selectedComplaint, setSelectedComplaint] = useState(null);
  const [filterStatus, setFilterStatus] = useState('ALL');
  const [subordinates, setSubordinates] = useState([]);
  const [selectedOfficer, setSelectedOfficer] = useState('ALL');

  const isPI = policeUser?.role === 'STATION_ADMIN';

  // Debug: Log when complaints state changes
  useEffect(() => {
    console.log('Complaints state updated:', complaints.length, 'complaints');
  }, [complaints]);

  useEffect(() => {
    const storedUser = localStorage.getItem('policeUser');
    if (!storedUser) {
      navigate('/police/login');
      return;
    }
    const user = JSON.parse(storedUser);
    
    // Only PI can access this page
    if (user.role !== 'STATION_ADMIN') {
      navigate('/police/psi-dashboard');
      return;
    }
    
    console.log('Police user data:', user);
    // stationId might be stored as stationId or station.id depending on API response
    const stationId = user.stationId || user.station?.id;
    console.log('Station ID:', stationId);
    setPoliceUser(user);
    
    if (stationId) {
      loadComplaints(stationId);
      
      // If PI, load subordinate PSIs
      if (user.role === 'STATION_ADMIN') {
        loadSubordinates(stationId);
      }
    } else {
      console.error('No station ID found for police user');
      setLoading(false);
      setError('No station assigned to this officer');
    }
  }, [navigate]);

  const loadSubordinates = async (stationId) => {
    try {
      const response = await api.get(`/police/station/${stationId}`);
      // Filter to get only INVESTIGATING_OFFICERs (PSIs)
      const psis = response.data.filter(p => p.role === 'INVESTIGATING_OFFICER');
      setSubordinates(psis);
    } catch (err) {
      console.error('Error loading subordinates:', err);
    }
  };

  const loadComplaints = async (stationId) => {
    try {
      setLoading(true);
      setError('');
      console.log('Fetching complaints for station ID:', stationId);
      // Try to fetch complaints from backend
      const response = await api.get(`/complaints/station/${stationId}`);
      console.log('Response type:', typeof response.data);
      
      let complaintsData = response.data;
      
      // If already an array, use it
      if (Array.isArray(complaintsData)) {
        console.log('Data is already an array with', complaintsData.length, 'items');
        setComplaints(complaintsData);
      } 
      // If it's a string, try to extract complaint objects manually
      else if (typeof complaintsData === 'string') {
        console.log('Data is a string, attempting to extract complaints...');
        try {
          // Try to extract individual complaint objects from the corrupted JSON
          // Match complaint objects at the start: {"id":X,"description":"...","actualCategory":...,"predictedCategory":"...","createdAt":"...","status":"..."
          const complaintRegex = /\{"id":(\d+),"description":"([^"]*)","actualCategory":(null|"[^"]*"),"predictedCategory":"([^"]*)","createdAt":"([^"]*)","status":"([^"]*)"/g;
          const extractedComplaints = [];
          let match;
          
          while ((match = complaintRegex.exec(complaintsData)) !== null) {
            extractedComplaints.push({
              id: parseInt(match[1]),
              description: match[2],
              actualCategory: match[3] === 'null' ? null : match[3].replace(/"/g, ''),
              predictedCategory: match[4],
              createdAt: match[5],
              status: match[6]
            });
          }
          
          console.log('Extracted', extractedComplaints.length, 'complaints from string');
          setComplaints(extractedComplaints);
        } catch (e) {
          console.error('Failed to extract complaints:', e);
          setComplaints([]);
        }
      } else {
        console.log('Unknown data type, setting empty array');
        setComplaints([]);
      }
    } catch (err) {
      console.error('Error loading complaints:', err);
      setComplaints([]);
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('policeUser');
    navigate('/police/login');
  };

  const handleViewComplaint = (complaint) => {
    setSelectedComplaint(complaint);
  };

  const handleCloseModal = () => {
    setSelectedComplaint(null);
  };

  const handleUpdateStatus = async (complaintId, newStatus, actualCategory = null) => {
    try {
      const updateData = { status: newStatus };
      if (actualCategory) {
        updateData.actualCategory = actualCategory;
      }
      await api.put(`/complaints/${complaintId}/status`, updateData);
      
      // Update the selected complaint state instead of reloading all
      if (actualCategory && selectedComplaint) {
        setSelectedComplaint({
          ...selectedComplaint,
          actualCategory: actualCategory,
          status: newStatus
        });
      }
      
      // Update complaints list without reloading
      setComplaints(prevComplaints => 
        prevComplaints.map(c => 
          c.id === complaintId 
            ? { ...c, status: newStatus, actualCategory: actualCategory || c.actualCategory }
            : c
        )
      );
      
      if (newStatus === 'CLOSED') {
        setSelectedComplaint(null);
      }
    } catch (err) {
      console.error('Error updating status:', err);
      alert('Failed to update status');
    }
  };

  const handleAssignOfficer = async (complaintId, officerId) => {
    try {
      const currentStatus = selectedComplaint ? selectedComplaint.status : 'PENDING';
      await api.put(`/complaints/${complaintId}/status`, { 
        status: currentStatus,
        officerId: officerId 
      });
      
      // Find officer details
      const officer = subordinates.find(s => s.policeId === officerId);
      
      // Update complaints list
      setComplaints(prevComplaints => 
        prevComplaints.map(c => 
          c.id === complaintId 
            ? { 
                ...c, 
                assignedOfficerId: officerId,
                assignedOfficerName: officer?.name,
                assignedOfficerBadge: officer?.badgeNumber
              }
            : c
        )
      );
      
      // Update selected complaint if open
      if (selectedComplaint && selectedComplaint.id === complaintId) {
        setSelectedComplaint({
          ...selectedComplaint,
          assignedOfficerId: officerId,
          assignedOfficerName: officer?.name,
          assignedOfficerBadge: officer?.badgeNumber
        });
      }
      
      alert('Officer assigned successfully');
    } catch (err) {
      console.error('Error assigning officer:', err);
      alert('Failed to assign officer');
    }
  };

  const filteredComplaints = complaints.filter(c => {
    let matches = true;
    if (filterStatus !== 'ALL') {
      matches = c.status === filterStatus;
    }
    return matches;
  });

  const getStatusBadge = (status) => {
    const statusClasses = {
      'PENDING': 'status-pending',
      'READ': 'status-read',
      'UNDER_REVIEW': 'status-investigating',
      'CLOSED': 'status-closed'
    };
    return statusClasses[status] || 'status-pending';
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  if (loading) {
    return (
      <div className="police-dashboard loading">
        <div className="spinner"></div>
        <p>Loading dashboard...</p>
      </div>
    );
  }

  return (
    <div className="police-dashboard pi-view">
      {/* Navbar */}
      <nav className="police-navbar">
        <div className="nav-brand">
          <span className="brand-icon">⭐</span>
          <h1>PI Dashboard</h1>
          <span className="brand-name">PI (Station Admin) Portal</span>
        </div>
        <div className="nav-user">
          <span className="user-info">
            👮 {policeUser?.name} | Badge: {policeUser?.badgeNumber}
          </span>
          <span className="role-badge">PI</span>
          <button onClick={handleLogout} className="btn-logout">Logout</button>
        </div>
      </nav>

      {/* Main Content */}
      <div className="dashboard-container">
        {/* PI Section - Subordinates Overview */}
        {isPI && (
          <div className="pi-section">
            <div className="section-header">
              <h2>👥 Subordinate Officers (PSIs)</h2>
            </div>
            <div className="subordinates-grid">
              {subordinates.length === 0 ? (
                <div className="no-subordinates">
                  <p>No subordinate officers found</p>
                </div>
              ) : (
                subordinates.map(officer => (
                  <div key={officer.policeId} className="officer-card">
                    <div className="officer-avatar">👮</div>
                    <div className="officer-info">
                      <h4>{officer.name}</h4>
                      <p>Badge: {officer.badgeNumber}</p>
                      <p>Email: {officer.email}</p>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        )}

        {/* Stats Cards */}
        <div className="stats-row">
          <div className="stat-card">
            <div className="stat-icon">📋</div>
            <div className="stat-info">
              <h3>{complaints.length}</h3>
              <p>Total Complaints</p>
            </div>
          </div>
          <div className="stat-card pending">
            <div className="stat-icon">⏳</div>
            <div className="stat-info">
              <h3>{complaints.filter(c => c.status === 'PENDING').length}</h3>
              <p>Pending</p>
            </div>
          </div>
          <div className="stat-card read">
            <div className="stat-icon">👁️</div>
            <div className="stat-info">
              <h3>{complaints.filter(c => c.status === 'READ').length}</h3>
              <p>Under Review</p>
            </div>
          </div>
          {isPI && (
            <div className="stat-card officers">
              <div className="stat-icon">👥</div>
              <div className="stat-info">
                <h3>{subordinates.length}</h3>
                <p>PSI Officers</p>
              </div>
            </div>
          )}
        </div>

        {/* Station Info */}
        <div className="station-info-card">
          <h3>🏛️ {policeUser?.stationName}</h3>
          <p>Station Code: {policeUser?.stationCode} | Role: Police Inspector (PI)</p>
        </div>

        {/* Complaints Section */}
        <div className="complaints-section">
          <div className="section-header">
            <h2>📁 Assigned Complaints</h2>
            <div className="filter-controls">
              <select 
                value={filterStatus} 
                onChange={(e) => setFilterStatus(e.target.value)}
                className="filter-select"
              >
                <option value="ALL">All Status</option>
                <option value="PENDING">Pending</option>
                <option value="READ">Read</option>
              </select>
            </div>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          {filteredComplaints.length === 0 ? (
            <div className="no-complaints">
              <span className="empty-icon">📭</span>
              <p>No complaints found</p>
            </div>
          ) : (
            <div className="complaints-table-container">
              <table className="complaints-table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Description</th>
                    <th>Status</th>
                    {isPI && <th>Assigned Officer</th>}
                    <th>Filed On</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredComplaints.map(complaint => (
                    <tr key={complaint.id}>
                      <td><span className="complaint-id">#{complaint.id}</span></td>
                      <td>
                        <span className="category-badge">
                          {complaint.predictedCategory || 'Uncategorized'}
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
                      {isPI && (
                        <td>
                          {complaint.assignedOfficerName ? (
                            <span style={{fontSize: '0.85rem', color: '#2c3e50'}}>
                              👤 {complaint.assignedOfficerName}
                              <br />
                              <small style={{color: '#7f8c8d'}}>({complaint.assignedOfficerBadge})</small>
                            </span>
                          ) : (
                            <span style={{color: '#95a5a6', fontSize: '0.85rem'}}>Not assigned</span>
                          )}
                        </td>
                      )}
                      <td>{formatDate(complaint.createdAt)}</td>
                      <td>
                        <button 
                          className="btn-view"
                          onClick={() => handleViewComplaint(complaint)}
                        >
                          👁️ View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Complaint Detail Modal */}
      {selectedComplaint && (
        <div className="modal-overlay" onClick={handleCloseModal}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h2>Complaint #{selectedComplaint.id}</h2>
              <button className="btn-close" onClick={handleCloseModal}>×</button>
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
                <span>{selectedComplaint.predictedCategory || 'N/A'}</span>
                {selectedComplaint.predictedCategory && !selectedComplaint.actualCategory && (
                  <button 
                    className="btn-action"
                    style={{backgroundColor: '#27ae60', marginLeft: '10px', padding: '6px 12px', fontSize: '0.85rem'}}
                    onClick={() => {
                      handleUpdateStatus(selectedComplaint.id, selectedComplaint.status, selectedComplaint.predictedCategory);
                    }}
                  >
                    ✓ Approve Predicted
                  </button>
                )}
              </div>
              <div className="detail-row">
                <label>Actual Category:</label>
                <span>{selectedComplaint.actualCategory || 'Not yet classified'}</span>
              </div>
              <div className="detail-row">
                <label>Update Actual Category:</label>
                <select 
                  id="actualCategoryDropdown"
                  defaultValue={selectedComplaint.actualCategory || ''}
                  style={{
                    padding: '8px',
                    border: '1px solid #ddd',
                    borderRadius: '4px',
                    flex: 1
                  }}
                >
                  <option value="">-- Select Category --</option>
                  <option value="Kidnapping / Abduction / Missing Person (BNS 140–151)">Kidnapping / Abduction / Missing Person</option>
                  <option value="Sexual Offences (BNS 63–70)">Sexual Offences</option>
                  <option value="Assault / Hurt / Violence (BNS 115–140)">Assault / Hurt / Violence</option>
                  <option value="Women & Child Safety (BNS 86 + POCSO)">Women & Child Safety</option>
                  <option value="Harassment / Threats / Stalking (BNS 351–353)">Harassment / Threats / Stalking</option>
                  <option value="Accident / Hit & Run (BNS 106, 112, 279)">Accident / Hit & Run</option>
                  <option value="Cybercrime (IT Act + BNS mapping)">Cybercrime</option>
                  <option value="Fraud / Cheating / Financial Crimes (BNS 318–324)">Fraud / Cheating / Financial Crimes</option>
                  <option value="Theft & Robbery (BNS 303–309)">Theft & Robbery</option>
                  <option value="Trespass / Housebreaking / Property Disputes (BNS 332–335)">Trespass / Housebreaking / Property Disputes</option>
                  <option value="Defamation / Public Order Offences (BNS 356–357, 147–150)">Defamation / Public Order Offences</option>
                  <option value="Other / Cannot Classify">Other / Cannot Classify</option>
                </select>
                <button 
                  className="btn-action"
                  style={{backgroundColor: '#9b59b6', marginLeft: '10px'}}
                  onClick={() => {
                    const dropdown = document.getElementById('actualCategoryDropdown');
                    const category = dropdown.value;
                    if (category) {
                      handleUpdateStatus(selectedComplaint.id, selectedComplaint.status, category);
                    } else {
                      alert('Please select a category');
                    }
                  }}
                >
                  Update Category
                </button>
              </div>
              <div className="detail-row">
                <label>Filed On:</label>
                <span>{formatDate(selectedComplaint.createdAt)}</span>
              </div>
              {isPI && selectedComplaint.assignedOfficerName && (
                <div className="detail-row">
                  <label>Currently Assigned To:</label>
                  <span style={{color: '#27ae60', fontWeight: '500'}}>
                    👤 {selectedComplaint.assignedOfficerName} ({selectedComplaint.assignedOfficerBadge})
                  </span>
                </div>
              )}
              {isPI && (
                <div className="detail-row">
                  <label>Assign to Officer:</label>
                  <select 
                    id="assignOfficerDropdown"
                    defaultValue={selectedComplaint.assignedOfficerId || ''}
                    style={{
                      padding: '8px',
                      border: '1px solid #ddd',
                      borderRadius: '4px',
                      flex: 1
                    }}
                  >
                    <option value="">-- Select Officer --</option>
                    {subordinates.map(officer => (
                      <option key={officer.policeId} value={officer.policeId}>
                        {officer.name} - {officer.badgeNumber}
                      </option>
                    ))}
                  </select>
                  <button 
                    className="btn-action"
                    style={{backgroundColor: '#e67e22', marginLeft: '10px'}}
                    onClick={() => {
                      const dropdown = document.getElementById('assignOfficerDropdown');
                      const officerId = parseInt(dropdown.value);
                      if (officerId) {
                        handleAssignOfficer(selectedComplaint.id, officerId);
                      } else {
                        alert('Please select an officer');
                      }
                    }}
                  >
                    Assign Officer
                  </button>
                </div>
              )}
              <div className="detail-row full-width">
                <label>Description:</label>
                <div className="description-box">
                  {selectedComplaint.description}
                </div>
              </div>
            </div>
            <div className="modal-footer">
              <div style={{display: 'flex', gap: '10px', flexWrap: 'wrap', justifyContent: 'flex-end'}}>
                {selectedComplaint.status === 'PENDING' && (
                  <button 
                    className="btn-action btn-mark-read"
                    onClick={() => handleUpdateStatus(selectedComplaint.id, 'READ')}
                  >
                    ✓ Mark as Read
                  </button>
                )}
                {(selectedComplaint.status === 'READ' || selectedComplaint.status === 'PENDING') && (
                  <button 
                    className="btn-action"
                    style={{backgroundColor: '#3498db'}}
                    onClick={() => handleUpdateStatus(selectedComplaint.id, 'UNDER_REVIEW')}
                  >
                    🔍 Under Review
                  </button>
                )}
                {selectedComplaint.status !== 'CLOSED' && (
                  <button 
                    className="btn-action"
                    style={{backgroundColor: '#27ae60'}}
                    onClick={() => {
                      const category = prompt('Enter actual crime category (or leave blank):');
                      handleUpdateStatus(selectedComplaint.id, 'CLOSED', category || null);
                    }}
                  >
                    ✔️ Close Complaint
                  </button>
                )}
                <button className="btn-action btn-secondary" onClick={handleCloseModal}>
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PoliceDashboardPage;
