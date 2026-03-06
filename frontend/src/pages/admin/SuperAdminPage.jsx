import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  registerPolice,
  getAllPolice,
  getAllStations,
  updatePolice,
  getPoliceByStation,
  createPoliceStation
} from '../../services/policeService';
import {
  createSubdivision,
  getAllSubdivisions,
  assignDspOfficer,
  addStationToSubdivision,
  removeStationFromSubdivision
} from '../../services/subdivisionService';
import './SuperAdminPage.css';

const SuperAdminPage = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState('create');
  const [police, setPolice] = useState([]);
  const [stations, setStations] = useState([]);
  const [subdivisions, setSubdivisions] = useState([]);
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

  const [newStation, setNewStation] = useState({
    stationCode: '',
    stationName: '',
    address: '',
    district: '',
    state: ''
  });

  const [newSubdivision, setNewSubdivision] = useState({
    subdivisionCode: '',
    subdivisionName: '',
    district: '',
    state: '',
    dspOfficerId: null
  });

  const [selectedSubdivision, setSelectedSubdivision] = useState(null);
  const [expandedStations, setExpandedStations] = useState({});

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
      const [policeData, stationsData, subdivisionsData] = await Promise.all([
        getAllPolice(),
        getAllStations(),
        getAllSubdivisions()
      ]);
      setPolice(policeData);
      setStations(stationsData);
      setSubdivisions(subdivisionsData);
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

  const handleCreateStation = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      await createPoliceStation(newStation);
      setSuccess('Police station created successfully!');
      setNewStation({
        stationCode: '',
        stationName: '',
        address: '',
        district: '',
        state: ''
      });
      loadData();
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSubdivision = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const subdivisionData = {
        subdivisionCode: newSubdivision.subdivisionCode,
        subdivisionName: newSubdivision.subdivisionName,
        district: newSubdivision.district,
        state: newSubdivision.state,
        dspOfficerId: newSubdivision.dspOfficerId || null
      };
      await createSubdivision(subdivisionData);
      setSuccess('Subdivision created successfully!');
      setNewSubdivision({
        subdivisionCode: '',
        subdivisionName: '',
        district: '',
        state: '',
        dspOfficerId: null
      });
      loadData();
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  };

  const handleAssignDsp = async (subdivisionId, dspOfficerId) => {
    setError('');
    setSuccess('');
    try {
      await assignDspOfficer(subdivisionId, dspOfficerId);
      setSuccess('DSP officer assigned successfully!');
      loadData();
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to assign DSP officer');
    }
  };

  const handleAddStationToSubdivision = async (subdivisionId, stationId) => {
    setError('');
    setSuccess('');
    try {
      await addStationToSubdivision(subdivisionId, stationId);
      setSuccess('Police station added to subdivision successfully!');
      loadData();
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to add station to subdivision');
    }
  };

  const handleRemoveStationFromSubdivision = async (subdivisionId, stationId) => {
    setError('');
    setSuccess('');
    try {
      await removeStationFromSubdivision(subdivisionId, stationId);
      setSuccess('Police station removed from subdivision successfully!');
      loadData();
    } catch (err) {
      setError(typeof err === 'string' ? err : 'Failed to remove station from subdivision');
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

  const getOfficerSubdivision = (officerId) => {
    const subdivision = subdivisions.find(sub => 
      sub.dspOfficer && sub.dspOfficer.policeId === officerId
    );
    return subdivision ? subdivision.subdivisionName : null;
  };

  const getSortedOfficers = () => {
    const roleOrder = {
      'DEPUTY_SUPRINTENDENT': 1,
      'STATION_ADMIN': 2,
      'INVESTIGATING_OFFICER': 3
    };
    
    return [...police].sort((a, b) => {
      const orderA = roleOrder[a.role] || 999;
      const orderB = roleOrder[b.role] || 999;
      return orderA - orderB;
    });
  };

  const getDspOfficers = () => {
    return police.filter(p => 
      p.rank === 'DEPUTY_SUPERINTENDENT' || p.role === 'DEPUTY_SUPRINTENDENT'
    );
  };

  const getUnassignedStations = () => {
    const assignedStationIds = subdivisions.flatMap(sub => 
      stations.filter(st => st.subdivisionId === sub.subdivisionId).map(st => st.stationId)
    );
    return stations.filter(st => !assignedStationIds.includes(st.stationId));
  };

  const toggleStationExpansion = (stationId) => {
    setExpandedStations(prev => ({
      ...prev,
      [stationId]: !prev[stationId]
    }));
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
            className={`tab-btn ${activeTab === 'manage-stations' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage-stations')}
          >
            🏢 Manage Police Stations
          </button>
          <button
            className={`tab-btn ${activeTab === 'manage-subdivisions' ? 'active' : ''}`}
            onClick={() => setActiveTab('manage-subdivisions')}
          >
            🏛️ Manage Subdivisions
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
                      <option value="DEPUTY_SUPRINTENDENT">Deputy Superintendent (DSP)</option>
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

                  const isExpanded = expandedStations[station.stationId];
                  const maxVisible = 7;
                  const visibleOfficers = isExpanded ? stationOfficers : stationOfficers.slice(0, maxVisible);
                  const hasMore = stationOfficers.length > maxVisible;

                  return (
                    <div key={station.stationId} className="station-card">
                      <h3>{station.stationName}</h3>
                      <p className="station-info">{station.district}, {station.state}</p>

                      <div className="officers-list">
                        <label>Assigned Officers ({stationOfficers.length}):</label>
                        {stationOfficers.length > 0 ? (
                          <div className="officer-chips">
                            {visibleOfficers.map(officer => (
                              <div key={officer.policeId} className="officer-chip">
                                👮 {officer.name} ({officer.badgeNumber})
                              </div>
                            ))}
                            {hasMore && (
                              <button 
                                className="see-more-btn" 
                                onClick={() => toggleStationExpansion(station.stationId)}
                              >
                                {isExpanded ? '▲ See Less' : `▼ See More (${stationOfficers.length - maxVisible} more)`}
                              </button>
                            )}
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

          {/* Manage Police Stations Tab */}
          {activeTab === 'manage-stations' && (
            <div className="manage-stations-section">
              <h2>Manage Police Stations</h2>
              
              {/* Create Station Form */}
              <div className="create-station-form">
                <h3>Create New Police Station</h3>
                <form onSubmit={handleCreateStation} className="station-form">
                  <div className="form-row">
                    <div className="form-group">
                      <label>Station Code *</label>
                      <input
                        type="text"
                        placeholder="e.g., HQ001, PS-123, RJK-TLK"
                        value={newStation.stationCode}
                        onChange={(e) => setNewStation({...newStation, stationCode: e.target.value.toUpperCase()})}
                        required
                        pattern="[A-Z0-9-]{4,10}"
                        title="4-10 uppercase letters, numbers, and hyphens only"
                      />
                      <small>4-10 uppercase letters, numbers, and hyphens only</small>
                    </div>
                    <div className="form-group">
                      <label>Station Name *</label>
                      <input
                        type="text"
                        placeholder="e.g., Headquarters Police Station"
                        value={newStation.stationName}
                        onChange={(e) => setNewStation({...newStation, stationName: e.target.value})}
                        required
                        minLength={3}
                        maxLength={100}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Address *</label>
                    <textarea
                      placeholder="Complete station address"
                      value={newStation.address}
                      onChange={(e) => setNewStation({...newStation, address: e.target.value})}
                      required
                      minLength={10}
                      maxLength={200}
                      rows={3}
                    />
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label>District *</label>
                      <input
                        type="text"
                        placeholder="e.g., Mumbai"
                        value={newStation.district}
                        onChange={(e) => setNewStation({...newStation, district: e.target.value})}
                        required
                        minLength={2}
                        maxLength={50}
                      />
                    </div>
                    <div className="form-group">
                      <label>State *</label>
                      <input
                        type="text"
                        placeholder="e.g., Maharashtra"
                        value={newStation.state}
                        onChange={(e) => setNewStation({...newStation, state: e.target.value})}
                        required
                        minLength={2}
                        maxLength={50}
                      />
                    </div>
                  </div>

                  <button type="submit" className="btn-primary" disabled={loading}>
                    {loading ? 'Creating...' : '➕ Create Police Station'}
                  </button>
                </form>
              </div>

              {/* Existing Stations List */}
              <div className="stations-list">
                <h3>All Police Stations ({stations.length})</h3>
                <div className="stations-grid">
                  {stations.map(station => (
                    <div key={station.stationId} className="station-card">
                      <div className="station-header">
                        <span className="station-code">{station.stationCode}</span>
                        <span className="station-id">ID: {station.stationId}</span>
                      </div>
                      <h4>{station.stationName}</h4>
                      <p className="station-address">{station.address}</p>
                      <div className="station-location">
                        <span>📍 {station.district}, {station.state}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Manage Subdivisions Tab */}
          {activeTab === 'manage-subdivisions' && (
            <div className="manage-subdivisions-section">
              <h2>Manage Subdivisions</h2>

              {/* Create Subdivision Form */}
              <div className="create-subdivision-form">
                <h3>Create New Subdivision</h3>
                <form onSubmit={handleCreateSubdivision} className="subdivision-form">
                  <div className="form-row">
                    <div className="form-group">
                      <label>Subdivision Code *</label>
                      <input
                        type="text"
                        placeholder="e.g., SUB001, NORTHDIV, RJK-TLK"
                        value={newSubdivision.subdivisionCode}
                        onChange={(e) => setNewSubdivision({...newSubdivision, subdivisionCode: e.target.value.toUpperCase()})}
                        required
                        pattern="[A-Z0-9-]{4,15}"
                        title="4-15 uppercase letters, numbers, and hyphens only"
                      />
                      <small>4-15 uppercase letters, numbers, and hyphens only</small>
                    </div>
                    <div className="form-group">
                      <label>Subdivision Name *</label>
                      <input
                        type="text"
                        placeholder="e.g., North Subdivision"
                        value={newSubdivision.subdivisionName}
                        onChange={(e) => setNewSubdivision({...newSubdivision, subdivisionName: e.target.value})}
                        required
                        minLength={3}
                        maxLength={100}
                      />
                    </div>
                  </div>

                  <div className="form-row">
                    <div className="form-group">
                      <label>District *</label>
                      <input
                        type="text"
                        placeholder="e.g., Mumbai"
                        value={newSubdivision.district}
                        onChange={(e) => setNewSubdivision({...newSubdivision, district: e.target.value})}
                        required
                        minLength={2}
                        maxLength={50}
                      />
                    </div>
                    <div className="form-group">
                      <label>State *</label>
                      <input
                        type="text"
                        placeholder="e.g., Maharashtra"
                        value={newSubdivision.state}
                        onChange={(e) => setNewSubdivision({...newSubdivision, state: e.target.value})}
                        required
                        minLength={2}
                        maxLength={50}
                      />
                    </div>
                  </div>

                  <div className="form-group">
                    <label>Assign DSP Officer (Optional)</label>
                    <select
                      value={newSubdivision.dspOfficerId || ''}
                      onChange={(e) => setNewSubdivision({...newSubdivision, dspOfficerId: e.target.value ? parseInt(e.target.value) : null})}
                    >
                      <option value="">-- Assign Later --</option>
                      {getDspOfficers().map(dsp => (
                        <option key={dsp.policeId} value={dsp.policeId}>
                          {dsp.name} ({dsp.badgeNumber})
                        </option>
                      ))}
                    </select>
                  </div>

                  <button type="submit" className="btn-primary" disabled={loading}>
                    {loading ? 'Creating...' : '➕ Create Subdivision'}
                  </button>
                </form>
              </div>

              {/* Existing Subdivisions */}
              <div className="subdivisions-list">
                <h3>All Subdivisions ({subdivisions.length})</h3>
                <div className="subdivisions-grid">
                  {subdivisions.map(subdivision => {
                    const subdivisionsStations = stations.filter(st => st.subdivisionId === subdivision.subdivisionId);
                    return (
                      <div key={subdivision.subdivisionId} className="subdivision-card">
                        <div className="subdivision-header">
                          <span className="subdivision-code">{subdivision.subdivisionCode}</span>
                          <span className="subdivision-id">ID: {subdivision.subdivisionId}</span>
                        </div>
                        <h4>{subdivision.subdivisionName}</h4>
                        <div className="subdivision-location">
                          <span>📍 {subdivision.district}, {subdivision.state}</span>
                        </div>

                        {/* DSP Officer Info */}
                        <div className="subdivision-dsp">
                          <strong>DSP Officer:</strong>
                          {subdivision.dspOfficer ? (
                            <div className="dsp-info">
                              <span>{subdivision.dspOfficer.name}</span>
                              <span className="badge-small">{subdivision.dspOfficer.badgeNumber}</span>
                            </div>
                          ) : (
                            <span className="text-muted">Not Assigned</span>
                          )}
                        </div>

                        {/* Station Count */}
                        <div className="subdivision-stats">
                          <span>🏢 {subdivision.stationCount} Police Station(s)</span>
                        </div>

                        {/* Assign DSP Section */}
                        {!subdivision.dspOfficer && (
                          <div className="subdivision-actions">
                            <label>Assign DSP Officer:</label>
                            <select
                              onChange={(e) => {
                                if (e.target.value) {
                                  handleAssignDsp(subdivision.subdivisionId, parseInt(e.target.value));
                                  e.target.value = '';
                                }
                              }}
                            >
                              <option value="">-- Select DSP --</option>
                              {getDspOfficers().map(dsp => (
                                <option key={dsp.policeId} value={dsp.policeId}>
                                  {dsp.name} ({dsp.badgeNumber})
                                </option>
                              ))}
                            </select>
                          </div>
                        )}

                        {/* Manage Stations Section */}
                        <div className="subdivision-stations">
                          <strong>Assigned Stations:</strong>
                          {subdivisionsStations.length > 0 ? (
                            <ul className="station-list-small">
                              {subdivisionsStations.map(st => (
                                <li key={st.stationId}>
                                  {st.stationName}
                                  <button
                                    className="btn-remove-small"
                                    onClick={() => handleRemoveStationFromSubdivision(subdivision.subdivisionId, st.stationId)}
                                    title="Remove station"
                                  >
                                    ✕
                                  </button>
                                </li>
                              ))}
                            </ul>
                          ) : (
                            <span className="text-muted">No stations assigned</span>
                          )}

                          {/* Add Station Dropdown */}
                          <div className="add-station-section">
                            <label>Add Station:</label>
                            <select
                              onChange={(e) => {
                                if (e.target.value) {
                                  handleAddStationToSubdivision(subdivision.subdivisionId, parseInt(e.target.value));
                                  e.target.value = '';
                                }
                              }}
                            >
                              <option value="">-- Select Station --</option>
                              {stations
                                .filter(st => !st.subdivisionId || st.subdivisionId === subdivision.subdivisionId)
                                .map(st => (
                                  <option key={st.stationId} value={st.stationId}>
                                    {st.stationName} ({st.stationCode})
                                  </option>
                                ))}
                            </select>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
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
                      <th>Station/Subdivision</th>
                      <th>Email</th>
                      <th>Mobile</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getSortedOfficers().map(officer => {
                      const subdivision = getOfficerSubdivision(officer.policeId);
                      const isDsp = officer.role === 'DEPUTY_SUPRINTENDENT';
                      
                      return (
                        <tr key={officer.policeId}>
                          <td><span className="badge-number">{officer.badgeNumber}</span></td>
                          <td><strong>{officer.name}</strong></td>
                          <td>{officer.rank}</td>
                          <td>
                            <span className={`role-badge ${officer.role.toLowerCase()}`}>
                              {officer.role.replace(/_/g, ' ')}
                            </span>
                          </td>
                          <td>
                            {isDsp ? (
                              subdivision ? (
                                <span className="subdivision-badge">🏛️ {subdivision}</span>
                              ) : (
                                <span className="text-muted">No Subdivision Assigned</span>
                              )
                            ) : (
                              getStationName(officer.stationId)
                            )}
                          </td>
                          <td>{officer.email}</td>
                          <td>{officer.mobileNumber}</td>
                        </tr>
                      );
                    })}
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
