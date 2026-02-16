import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  registerPolice,
  getAllPolice,
  getAllStations,
  updatePolice,
  getPoliceByStation
} from '../../services/policeService';
import './SuperAdminPage.css';

const SuperAdminPage = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('create');
  const [police, setPolice] = useState([]);
  const [stations, setStations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  const [newOfficer, setNewOfficer] = useState({
    name: '',
    email: '',
    password: '',
    mobileNumber: '',
    badgeNumber: '',
    rank: '',
    role: 'INVESTIGATING_OFFICER'
  });

  useEffect(() => {
         // Check if admin is logged in
    const adminUser = localStorage.getItem('adminUser');
    if (!adminUser) {
      navigate('/admin/login');
      return;
    }
    loadData();
  }, [navigate]);

  const loadData = async () => {
    try {
      const [policeData, stationsData] = await Promise.all([
        getAllPolice(),
        getAllStations()
      ]);
      setPolice(policeData);
      setStations(stationsData);
    } catch (err) {
      setError('Failed to load data');
    }
  };

  const handleRegisterOfficer = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      // Find Headquarters station
      const headquarters = stations.find(s => 
        s.stationName.toLowerCase().includes('headquarters') || 
        s.stationName.toLowerCase().includes('hq')
      );
      
      if (!headquarters) {
        setError('Headquarters station not found. Please create a Headquarters station first.');
        setLoading(false);
        return;
      }

      await registerPolice({
        name: newOfficer.name,
        email: newOfficer.email,
        password: newOfficer.password,
        mobileNumber: newOfficer.mobileNumber,
        badgeNumber: newOfficer.badgeNumber,
        rank: newOfficer.rank,
        role: newOfficer.role,
        stationId: headquarters.stationId
      });

      setSuccess(`Police officer registered successfully and assigned to ${headquarters.stationName}!`);
      setNewOfficer({
        name: '',
        email: '',
        password: '',
        mobileNumber: '',
        badgeNumber: '',
        rank: '',
        role: 'INVESTIGATING_OFFICER'
      });
      loadData();
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAssignStation = async (policeId, stationId) => {
    console.log('Assigning police:', policeId, 'to station:', stationId);
    setError('');
    setSuccess('');
    try {
      const result = await updatePolice(parseInt(policeId), { stationId: parseInt(stationId) });
      console.log('Assignment successful:', result);
      setSuccess('Station assigned successfully!');
      await loadData();
    } catch (err) {
      console.error('Assignment error:', err);
      setError(typeof err === 'string' ? err : 'Failed to assign station');
    }
  };

  const getStationAdmins = () => {
    return police.filter(p => p.role === 'STATION_ADMIN');
  };

  const getInvestigatingOfficers = () => {
    return police.filter(p => p.role === 'INVESTIGATING_OFFICER');
  };

  const getStationName = (stationId) => {
    if (!stationId) return 'Not Assigned';
    const station = stations.find(s => s.stationId === stationId);
    return station ? station.stationName : 'Not Assigned';
  };

  return (
    <div className="super-admin-page">
      {/* Header */}
      <nav className="admin-navbar">
        <div className="nav-brand">
          <span className="brand-icon">👮‍♂️</span>
          <span className="brand-name">Super Admin Portal</span>
        </div>
        <button onClick={() => navigate('/')} className="btn-back">
          ← Back to Home
        </button>
      </nav>

      {/* Main Content */}
      <div className="admin-container">
        <h1 className="admin-title">Police Management System</h1>

        {/* Tabs */}
        <div className="admin-tabs">
          <button
            className={`tab-btn ${activeTab === 'create' ? 'active' : ''}`}
            onClick={() => setActiveTab('create')}
          >
            ➕ Register New Officer
          </button>
          <button
            className={`tab-btn ${activeTab === 'assign-admin' ? 'active' : ''}`}
            onClick={() => setActiveTab('assign-admin')}
          >
            🏛️ Assign Station Admins
          </button>
          <button
            className={`tab-btn ${activeTab === 'assign-officer' ? 'active' : ''}`}
            onClick={() => setActiveTab('assign-officer')}
          >
            🔍 Assign Investigating Officers
          </button>
          <button
            className={`tab-btn ${activeTab === 'view-all' ? 'active' : ''}`}
            onClick={() => setActiveTab('view-all')}
          >
            👥 View All Officers
          </button>
        </div>

        {/* Notifications */}
        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        {/* Tab Content */}
        <div className="tab-content">
          {/* Create New Officer Tab */}
          {activeTab === 'create' && (
            <div className="create-officer-section">
              <h2>Register New Police Officer</h2>
              <form onSubmit={handleRegisterOfficer} className="officer-form">
                <div className="form-row">
                  <div className="form-group">
                    <label>Full Name *</label>
                    <input
                      type="text"
                      value={newOfficer.name}
                      onChange={(e) => setNewOfficer({ ...newOfficer, name: e.target.value })}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label>Email *</label>
                    <input
                      type="email"
                      value={newOfficer.email}
                      onChange={(e) => setNewOfficer({ ...newOfficer, email: e.target.value })}
                      required
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Password *</label>
                    <input
                      type="password"
                      value={newOfficer.password}
                      onChange={(e) => setNewOfficer({ ...newOfficer, password: e.target.value })}
                      required
                      minLength={8}
                    />
                  </div>
                  <div className="form-group">
                    <label>Mobile Number *</label>
                    <input
                      type="tel"
                      value={newOfficer.mobileNumber}
                      onChange={(e) => setNewOfficer({ ...newOfficer, mobileNumber: e.target.value })}
                      required
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Badge Number *</label>
                    <input
                      type="text"
                      value={newOfficer.badgeNumber}
                      onChange={(e) => setNewOfficer({ ...newOfficer, badgeNumber: e.target.value })}
                      required
                    />
                  </div>
                  <div className="form-group">
                    <label>Rank *</label>
                    <input
                      type="text"
                      value={newOfficer.rank}
                      onChange={(e) => setNewOfficer({ ...newOfficer, rank: e.target.value })}
                      placeholder="e.g., Inspector, Sub-Inspector"
                      required
                    />
                  </div>
                </div>

                <div className="form-row">
                  <div className="form-group">
                    <label>Role *</label>
                    <select
                      value={newOfficer.role}
                      onChange={(e) => setNewOfficer({ ...newOfficer, role: e.target.value })}
                      required
                    >
                      <option value="INVESTIGATING_OFFICER">Investigating Officer</option>
                      <option value="STATION_ADMIN">Station Admin</option>
                    </select>
                  </div>
                </div>

                <div className="info-note">
                  <span className="info-icon">ℹ️</span>
                  <span>New officers will be automatically assigned to Headquarters. Reassign them to specific stations using the assignment tabs.</span>
                </div>

                <button type="submit" className="btn-submit" disabled={loading}>
                  {loading ? 'Registering...' : '✓ Register Officer'}
                </button>
              </form>
            </div>
          )}

          {/* Assign Station Admins Tab */}
          {activeTab === 'assign-admin' && (
            <div className="assign-admin-section">
              <h2>Assign Station Admins</h2>
              <p className="info-text">Each station can have only ONE admin officer</p>

              <div className="stations-grid">
                {stations.map(station => {
                  const admin = police.find(p =>
                    p.role === 'STATION_ADMIN' &&
                    p.stationId === station.stationId
                  );

                  // Show all station admins in dropdown for reassignment
                  const availableAdmins = getStationAdmins();

                  return (
                    <div key={station.stationId} className="station-card">
                      <h3>{station.stationName}</h3>
                      <p className="station-info">{station.district}, {station.state}</p>
                      <p className="station-code">Code: {station.stationCode}</p>

                      <div className="admin-assignment">
                        <label>Station Admin:</label>
                        {admin ? (
                          <div className="assigned-admin">
                            <span className="admin-badge">✓ {admin.name}</span>
                            <span className="admin-badge-no">Badge: {admin.badgeNumber}</span>
                          </div>
                        ) : (
                          <div className="no-admin">❌ No Admin Assigned</div>
                        )}

                        <select
                          onChange={(e) => e.target.value && handleAssignStation(e.target.value, station.stationId)}
                          className="assign-select"
                          defaultValue=""
                        >
                          <option value="">-- Assign/Change Admin --</option>
                          {availableAdmins.map(officer => (
                            <option key={officer.policeId} value={officer.policeId}>
                              {officer.name} (Badge: {officer.badgeNumber})
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Assign Investigating Officers Tab */}
          {activeTab === 'assign-officer' && (
            <div className="assign-officer-section">
              <h2>Assign Investigating Officers</h2>
              <p className="info-text">Each station can have multiple investigating officers</p>

              <div className="stations-grid">
                {stations.map(station => {
                  const stationOfficers = police.filter(p =>
                    p.role === 'INVESTIGATING_OFFICER' &&
                    p.stationId === station.stationId
                  );

                  const availableOfficers = getInvestigatingOfficers().filter(p =>
                    !p.policeStation || p.policeStation.stationId === station.stationId
                  );

                  return (
                    <div key={station.stationId} className="station-card">
                      <h3>{station.stationName}</h3>
                      <p className="station-info">{station.district}, {station.state}</p>

                      <div className="officers-list">
                        <label>Assigned Officers ({stationOfficers.length}):</label>
                        {stationOfficers.length > 0 ? (
                          <div className="officer-chips">
                            {stationOfficers.map(officer => (
                              <div key={officer.policeId} className="officer-chip">
                                👮 {officer.name} ({officer.badgeNumber})
                              </div>
                            ))}
                          </div>
                        ) : (
                          <div className="no-officers">No officers assigned</div>
                        )}

                        <select
                          onChange={(e) => e.target.value && handleAssignStation(e.target.value, station.stationId)}
                          className="assign-select"
                          defaultValue=""
                        >
                          <option value="">-- Add Officer --</option>
                          {availableOfficers.map(officer => (
                            <option key={officer.policeId} value={officer.policeId}>
                              {officer.name} (Badge: {officer.badgeNumber})
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* View All Officers Tab */}
          {activeTab === 'view-all' && (
            <div className="view-all-section">
              <h2>All Police Officers</h2>

              <div className="officers-table-container">
                <table className="officers-table">
                  <thead>
                    <tr>
                      <th>Badge</th>
                      <th>Name</th>
                      <th>Rank</th>
                      <th>Role</th>
                      <th>Station</th>
                      <th>Email</th>
                      <th>Mobile</th>
                    </tr>
                  </thead>
                  <tbody>
                    {police.map(officer => (
                      <tr key={officer.policeId}>
                        <td><span className="badge-number">{officer.badgeNumber}</span></td>
                        <td><strong>{officer.name}</strong></td>
                        <td>{officer.rank}</td>
                        <td>
                          <span className={`role-badge ${officer.role.toLowerCase()}`}>
                            {officer.role.replace('_', ' ')}
                          </span>
                        </td>
                        <td>{getStationName(officer.stationId)}</td>
                        <td>{officer.email}</td>
                        <td>{officer.mobileNumber}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SuperAdminPage;
