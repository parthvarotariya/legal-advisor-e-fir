import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import { generateFirPdf } from '../../utils/generateFirPdf';
import './DashboardCommon.css';

const STATUS_LABELS = {
  RECEIVED: 'Received',
  PE_PENDING_DSP_APPROVAL: 'PE – Awaiting Approval',
  PE_ASSIGNED: 'PE – Assigned',
  PE_SUBMITTED: 'PE – Submitted',
  FIR_REGISTERED: 'FIR Registered',
  CLOSED_NO_CRIME: 'Closed (No Crime)'
};

const DSPDashboardPage = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [firs, setFirs] = useState([]);
  const [peReports, setPeReports] = useState([]);
  const [stations, setStations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('approvals');
  const [selected, setSelected] = useState(null);
  const [selectedFir, setSelectedFir] = useState(null);

  useEffect(() => {
    const stored = localStorage.getItem('policeUser');
    if (!stored) { navigate('/police/login'); return; }
    const u = JSON.parse(stored);
    if (u.role !== 'DEPUTY_SUPRINTENDENT') {
      if (u.role === 'STATION_ADMIN') navigate('/police/pi-dashboard');
      else if (u.role === 'INVESTIGATING_OFFICER') navigate('/police/psi-dashboard');
      return;
    }
    setUser(u);
    if (u.subdivisionId) {
      loadAll(u.subdivisionId);
    } else {
      setError('No subdivision assigned to this DSP officer. Contact the Super Admin.');
      setLoading(false);
    }
  }, [navigate]);

  const loadAll = async (subdivisionId) => {
    setLoading(true);
    try {
      const [compRes, firRes, peRes] = await Promise.all([
        api.get(`/complaints/subdivision/${subdivisionId}`),
        api.get(`/fir/subdivision/${subdivisionId}`),
        api.get(`/preliminary-report/subdivision/${subdivisionId}`)
      ]);
      setComplaints(Array.isArray(compRes.data) ? compRes.data : []);
      setFirs(Array.isArray(firRes.data) ? firRes.data : []);
      setPeReports(Array.isArray(peRes.data) ? peRes.data : []);
    } catch (e) {
      console.error(e);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const refresh = useCallback(() => {
    if (user?.subdivisionId) loadAll(user.subdivisionId);
  }, [user]);

  /* ── PE Approval / Denial ── */
  const handleApprovePE = async (complaint) => {
    try {
      await api.put(`/complaints/${complaint.id}/status`, { status: 'PE_ASSIGNED' });
      setComplaints(prev => prev.map(c => c.id === complaint.id ? { ...c, status: 'PE_ASSIGNED' } : c));
      if (selected?.id === complaint.id) setSelected({ ...complaint, status: 'PE_ASSIGNED' });
      alert('PE approved. PI can now assign a PSI.');
    } catch (e) {
      alert('Failed: ' + (e.response?.data?.message || e.message));
    }
  };

  const handleDenyPE = async (complaint) => {
    const reason = prompt('Reason for denying PE request:');
    if (!reason) return;
    try {
      await api.put(`/complaints/${complaint.id}/status`, { status: 'RECEIVED' });
      setComplaints(prev => prev.map(c => c.id === complaint.id ? { ...c, status: 'RECEIVED' } : c));
      if (selected?.id === complaint.id) setSelected({ ...complaint, status: 'RECEIVED' });
      alert('PE request denied. Complaint returned to RECEIVED.');
    } catch (e) {
      alert('Failed: ' + (e.response?.data?.message || e.message));
    }
  };

  /* ── Counts ── */
  const pendingApproval = complaints.filter(c => c.status === 'PE_PENDING_DSP_APPROVAL');
  const peActive = complaints.filter(c => c.status === 'PE_ASSIGNED' || c.status === 'PE_SUBMITTED');
  const firComplaints = complaints.filter(c => c.status === 'FIR_REGISTERED');
  const closedComplaints = complaints.filter(c => c.status === 'CLOSED_NO_CRIME');

  const formatDate = (d) => {
    if (!d) return '—';
    return new Date(d).toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('policeUser');
    navigate('/police/login');
  };

  if (loading) {
    return (
      <div className="dash-page">
        <div className="dash-loading"><div className="dash-spinner" /><p>Loading DSP Dashboard…</p></div>
      </div>
    );
  }

  return (
    <div className="dash-page dsp-theme">
      {/* Navbar */}
      <nav className="dash-nav">
        <div className="dash-nav-left">
          <span className="dash-nav-icon">🛡️</span>
          <div>
            <h1 className="dash-nav-title">DSP Dashboard</h1>
            <span className="dash-nav-sub">Deputy Superintendent of Police</span>
          </div>
        </div>
        <div className="dash-nav-right">
          <span className="dash-nav-user">👮 {user?.name}</span>
          <span className="dash-nav-badge">Badge: {user?.badgeNumber}</span>
          <span className="dash-role-chip dsp">DSP</span>
          <button onClick={handleLogout} className="dash-btn-logout">Logout</button>
        </div>
      </nav>

      <div className="dash-body">
        {error && <div className="dash-alert error">{error}</div>}

        {/* Subdivision Card */}
        <div className="dash-station-card dsp-card">
          <h3>🏛️ {user?.subdivisionName || 'Subdivision'}</h3>
          <p>DSP: {user?.name} &bull; Rank: {user?.rank}</p>
        </div>

        {/* Stats */}
        <div className="dash-stats">
          <div className="dash-stat" data-accent="red">
            <span className="dash-stat-icon">🔔</span>
            <div><h3>{pendingApproval.length}</h3><p>PE Pending Approval</p></div>
          </div>
          <div className="dash-stat" data-accent="blue">
            <span className="dash-stat-icon">📋</span>
            <div><h3>{complaints.length}</h3><p>Total Complaints</p></div>
          </div>
          <div className="dash-stat" data-accent="purple">
            <span className="dash-stat-icon">🔍</span>
            <div><h3>{peActive.length}</h3><p>PE Active</p></div>
          </div>
          <div className="dash-stat" data-accent="green">
            <span className="dash-stat-icon">📄</span>
            <div><h3>{firs.length}</h3><p>Total FIRs</p></div>
          </div>
          <div className="dash-stat" data-accent="teal">
            <span className="dash-stat-icon">📊</span>
            <div><h3>{peReports.length}</h3><p>PE Reports</p></div>
          </div>
          <div className="dash-stat" data-accent="gray">
            <span className="dash-stat-icon">✅</span>
            <div><h3>{closedComplaints.length}</h3><p>Closed</p></div>
          </div>
        </div>

        {/* Tabs */}
        <div className="dash-tabs">
          <button className={`dash-tab ${activeTab === 'approvals' ? 'active' : ''}`} onClick={() => setActiveTab('approvals')}>
            🔔 Pending Approval ({pendingApproval.length})
          </button>
          <button className={`dash-tab ${activeTab === 'complaints' ? 'active' : ''}`} onClick={() => setActiveTab('complaints')}>
            📋 All Complaints ({complaints.length})
          </button>
          <button className={`dash-tab ${activeTab === 'pe' ? 'active' : ''}`} onClick={() => setActiveTab('pe')}>
            📊 PE Reports ({peReports.length})
          </button>
          <button className={`dash-tab ${activeTab === 'firs' ? 'active' : ''}`} onClick={() => setActiveTab('firs')}>
            📄 FIRs ({firs.length})
          </button>
        </div>

        {/* PE Approval Queue */}
        {activeTab === 'approvals' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>🔔 PE Requests — Awaiting Your Approval</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {pendingApproval.length === 0 ? (
              <div className="dash-empty">✅ No PE requests pending your approval.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Category</th>
                      <th>Description</th>
                      <th>Station</th>
                      <th>Filed</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pendingApproval.map(c => (
                      <tr key={c.id}>
                        <td><span className="dash-id">#{c.id}</span></td>
                        <td><span className="dash-chip cat">{c.actualCategory || c.predictedCategory || '—'}</span></td>
                        <td className="dash-desc">{c.description?.substring(0, 50)}{c.description?.length > 50 ? '…' : ''}</td>
                        <td>{c.policeStationName || '—'}</td>
                        <td className="dash-date">{formatDate(c.createdAt)}</td>
                        <td>
                          <div className="dash-action-btns">
                            <button className="dash-btn success sm" onClick={() => handleApprovePE(c)}>✓ Approve</button>
                            <button className="dash-btn danger sm" onClick={() => handleDenyPE(c)}>✗ Deny</button>
                            <button className="dash-btn primary sm" onClick={() => setSelected(c)}>View</button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* All Complaints */}
        {activeTab === 'complaints' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>📋 All Complaints Under Jurisdiction</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {complaints.length === 0 ? (
              <div className="dash-empty">No complaints in your jurisdiction.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Category</th>
                      <th>Description</th>
                      <th>Station</th>
                      <th>Status</th>
                      <th>Assigned To</th>
                      <th>Filed</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {complaints.map(c => (
                      <tr key={c.id}>
                        <td><span className="dash-id">#{c.id}</span></td>
                        <td><span className="dash-chip cat">{c.actualCategory || c.predictedCategory || '—'}</span></td>
                        <td className="dash-desc">{c.description?.substring(0, 45)}{c.description?.length > 45 ? '…' : ''}</td>
                        <td>{c.policeStationName || '—'}</td>
                        <td><span className={`dash-status ${c.status?.toLowerCase().replace(/_/g, '-')}`}>{STATUS_LABELS[c.status] || c.status}</span></td>
                        <td>{c.assignedOfficerName ? <span>👤 {c.assignedOfficerName}</span> : <span style={{ color: '#bbb' }}>—</span>}</td>
                        <td className="dash-date">{formatDate(c.createdAt)}</td>
                        <td><button className="dash-btn primary sm" onClick={() => setSelected(c)}>View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* PE Reports */}
        {activeTab === 'pe' && (
          <div className="dash-card">
            <div className="dash-card-header"><h2>📊 Preliminary Enquiry Reports</h2></div>
            {peReports.length === 0 ? (
              <div className="dash-empty">No PE reports in your jurisdiction.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>Report ID</th>
                      <th>Complaint</th>
                      <th>Category</th>
                      <th>Cognizable?</th>
                      <th>Investigating Officer</th>
                      <th>Station</th>
                      <th>Submitted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {peReports.map(r => (
                      <tr key={r.reportId}>
                        <td><span className="dash-id">#{r.reportId}</span></td>
                        <td>#{r.complaintId}</td>
                        <td>{r.crimeCategory || '—'}</td>
                        <td>
                          <span className={`dash-chip ${r.cognizableOffence ? 'danger' : 'neutral'}`}>
                            {r.cognizableOffence ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td>{r.investigatingOfficerName || '—'}</td>
                        <td>{r.stationName || '—'}</td>
                        <td className="dash-date">{formatDate(r.submittedAt)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* FIRs */}
        {activeTab === 'firs' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>📄 FIRs Under Jurisdiction</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {firs.length === 0 ? (
              <div className="dash-empty">No FIRs in your jurisdiction.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>FIR #</th>
                      <th>Crime Category</th>
                      <th>Informant</th>
                      <th>District</th>
                      <th>Station</th>
                      <th>Status</th>
                      <th>Officer</th>
                      <th>Registered</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {firs.map(f => (
                      <tr key={f.firId}>
                        <td><span className="dash-id">{f.firNumber}</span></td>
                        <td>{f.crimeCategory || '—'}</td>
                        <td>{f.informantName || '—'}</td>
                        <td>{f.district || '—'}</td>
                        <td>{f.policeStationName || '—'}</td>
                        <td><span className={`dash-status ${f.status?.toLowerCase()}`}>{f.status}</span></td>
                        <td>{f.investigatingOfficerName || '—'}</td>
                        <td className="dash-date">{formatDate(f.registeredAt)}</td>
                        <td><button className="dash-btn primary sm" onClick={() => setSelectedFir(f)}>View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* FIR Detail Modal */}
      {selectedFir && (
        <div className="dash-overlay" onClick={() => setSelectedFir(null)}>
          <div className="dash-modal lg" onClick={e => e.stopPropagation()}>
            <div className="dash-modal-header">
              <h2>📄 FIR Details — {selectedFir.firNumber}</h2>
              <button className="dash-modal-close" onClick={() => setSelectedFir(null)}>×</button>
            </div>
            <div className="dash-modal-body">
              <div className="fir-pe-section" style={{ background: '#eff6ff', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4>📋 FIR Information</h4>
                <div className="fir-pe-field"><span className="fir-pe-label">FIR Number</span><span style={{ fontWeight: 600 }}>{selectedFir.firNumber}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">District</span><span>{selectedFir.district || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Status</span><span className={`dash-status ${selectedFir.status?.toLowerCase()}`}>{selectedFir.status}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Registered At</span><span>{formatDate(selectedFir.registeredAt)}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Written By</span><span>{selectedFir.firWrittenBy || '—'}</span></div>
                {selectedFir.complaintId && <div className="fir-pe-field"><span className="fir-pe-label">Complaint ID</span><span>#{selectedFir.complaintId}</span></div>}
              </div>
              <div className="fir-pe-section" style={{ background: '#f0fdf4', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4>👤 Informant Details</h4>
                <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{selectedFir.informantName || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Guardian</span><span>{selectedFir.informantGuardianName || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Contact</span><span>{selectedFir.informantContact || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Email</span><span>{selectedFir.informantEmail || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Address</span><span>{selectedFir.informantAddress || '—'}</span></div>
              </div>
              <div className="fir-pe-section" style={{ background: '#fefce8', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4>📍 Incident Details</h4>
                <div className="fir-pe-field"><span className="fir-pe-label">Location</span><span>{selectedFir.incidentLocation || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Date</span><span>{selectedFir.incidentDate || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">Time</span><span>{selectedFir.incidentTime || '—'}</span></div>
              </div>
              <div className="fir-pe-section" style={{ background: '#fef2f2', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4>⚖️ Crime Details</h4>
                <div className="fir-pe-field"><span className="fir-pe-label">Category</span><span className="dash-chip cat">{selectedFir.crimeCategory || '—'}</span></div>
                <div className="fir-pe-field"><span className="fir-pe-label">IPC / BNS Sections</span><span>{selectedFir.ipcSections || '—'}</span></div>
                {selectedFir.stolenPropertyDetails && <div className="fir-pe-field"><span className="fir-pe-label">Stolen Property</span><span>{selectedFir.stolenPropertyDetails}</span></div>}
                {selectedFir.accusedDetails && <div className="fir-pe-field"><span className="fir-pe-label">Accused</span><span>{selectedFir.accusedDetails}</span></div>}
                {selectedFir.witnessDetails && <div className="fir-pe-field"><span className="fir-pe-label">Witnesses</span><span>{selectedFir.witnessDetails}</span></div>}
              </div>
              {selectedFir.incidentDescription && (
                <div className="fir-pe-section" style={{ background: '#fffbeb', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <h4>📝 Incident Description</h4>
                  <div className="fir-pe-narrative">{selectedFir.incidentDescription}</div>
                </div>
              )}
              <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                <h4>🏛️ Station & Officer</h4>
                <div className="fir-pe-field"><span className="fir-pe-label">Police Station</span><span>{selectedFir.policeStationName || '—'}</span></div>
                {selectedFir.policeStationCode && <div className="fir-pe-field"><span className="fir-pe-label">Station Code</span><span>{selectedFir.policeStationCode}</span></div>}
                <div className="fir-pe-field"><span className="fir-pe-label">Investigating Officer</span><span>{selectedFir.investigatingOfficerName || '—'}</span></div>
                {selectedFir.investigatingOfficerBadgeNumber && <div className="fir-pe-field"><span className="fir-pe-label">Badge #</span><span>{selectedFir.investigatingOfficerBadgeNumber}</span></div>}
                {selectedFir.investigatingOfficerRank && <div className="fir-pe-field"><span className="fir-pe-label">Rank</span><span>{selectedFir.investigatingOfficerRank}</span></div>}
              </div>
              {(selectedFir.isEfir || selectedFir.isZeroFir || selectedFir.isVictimWoman || selectedFir.isDisabledVictim) && (
                <div className="fir-pe-section" style={{ background: '#f5f3ff', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                  <h4>📜 BNSS 2023 Compliance</h4>
                  {selectedFir.isEfir && <div className="fir-pe-field"><span className="fir-pe-label">e-FIR</span><span>✅ Yes</span></div>}
                  {selectedFir.isZeroFir && <div className="fir-pe-field"><span className="fir-pe-label">Zero FIR</span><span>✅ Yes — Dest: {selectedFir.destinationPoliceStation || '—'}</span></div>}
                  {selectedFir.isSignatureObtained != null && <div className="fir-pe-field"><span className="fir-pe-label">Signature</span><span>{selectedFir.isSignatureObtained ? '✅ Obtained' : '❌ Not Yet'}</span></div>}
                  {selectedFir.isVictimWoman && <div className="fir-pe-field"><span className="fir-pe-label">Woman Victim</span><span>✅ Yes</span></div>}
                  {selectedFir.recordedByWomanOfficer != null && <div className="fir-pe-field"><span className="fir-pe-label">Recorded by Woman Officer</span><span>{selectedFir.recordedByWomanOfficer ? '✅ Yes' : '❌ No'}</span></div>}
                  {selectedFir.isDisabledVictim && <div className="fir-pe-field"><span className="fir-pe-label">Disabled Victim</span><span>✅ Yes</span></div>}
                  {selectedFir.interpreterOrEducatorName && <div className="fir-pe-field"><span className="fir-pe-label">Interpreter / Educator</span><span>{selectedFir.interpreterOrEducatorName}</span></div>}
                  {selectedFir.isMagistrateStatementRecorded != null && <div className="fir-pe-field"><span className="fir-pe-label">Magistrate Statement</span><span>{selectedFir.isMagistrateStatementRecorded ? '✅ Recorded' : '❌ Not Recorded'}</span></div>}
                </div>
              )}
            </div>
            <div className="dash-modal-footer">
              <button className="dash-btn primary" onClick={() => generateFirPdf(selectedFir)} style={{ marginRight: 8 }}>
                📄 Download FIR PDF
              </button>
              <button className="dash-btn secondary" onClick={() => setSelectedFir(null)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* Complaint Detail Modal */}
      {selected && (
        <div className="dash-overlay" onClick={() => setSelected(null)}>
          <div className="dash-modal lg" onClick={e => e.stopPropagation()}>
            <div className="dash-modal-header">
              <h2>Complaint #{selected.id}</h2>
              <button className="dash-modal-close" onClick={() => setSelected(null)}>×</button>
            </div>
            <div className="dash-modal-body">
              <div className="dash-detail">
                <label>Status</label>
                <span className={`dash-status ${selected.status?.toLowerCase().replace(/_/g, '-')}`}>
                  {STATUS_LABELS[selected.status] || selected.status}
                </span>
              </div>
              <div className="dash-detail">
                <label>Category</label>
                <span className="dash-chip cat">{selected.actualCategory || selected.predictedCategory || '—'}</span>
              </div>
              <div className="dash-detail">
                <label>Station</label>
                <span>{selected.policeStationName || '—'}</span>
              </div>
              {selected.assignedOfficerName && (
                <div className="dash-detail">
                  <label>Assigned Officer</label>
                  <span>👤 {selected.assignedOfficerName} ({selected.assignedOfficerBadge})</span>
                </div>
              )}
              <div className="dash-detail">
                <label>Filed On</label>
                <span>{formatDate(selected.createdAt)}</span>
              </div>
              
              {/* Complainant Details */}
              <div className="dash-detail" style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #e2e8f0' }}>
                <label>Complainant</label>
                <span>{selected.complainantName || '—'}</span>
              </div>
              {selected.complainantMobile && (
                <div className="dash-detail">
                  <label>Mobile</label>
                  <span>📞 {selected.complainantMobile}</span>
                </div>
              )}
              {selected.complainantEmail && (
                <div className="dash-detail">
                  <label>Email</label>
                  <span>📧 {selected.complainantEmail}</span>
                </div>
              )}
              {selected.complainantAddress && (
                <div className="dash-detail">
                  <label>Address</label>
                  <span>{selected.complainantAddress}</span>
                </div>
              )}
              
              <div className="dash-detail col">
                <label>Description</label>
                <div className="dash-desc-box">{selected.description}</div>
              </div>

              {/* Approval Actions */}
              {selected.status === 'PE_PENDING_DSP_APPROVAL' && (
                <div className="dash-action-section">
                  <h4>PE Approval Decision</h4>
                  <div className="dash-action-row">
                    <button className="dash-btn success" onClick={() => handleApprovePE(selected)}>
                      ✓ Approve PE Request
                    </button>
                    <button className="dash-btn danger" onClick={() => handleDenyPE(selected)}>
                      ✗ Deny PE Request
                    </button>
                  </div>
                </div>
              )}

              {selected.status !== 'PE_PENDING_DSP_APPROVAL' && (
                <div className="dash-info-box">
                  This complaint is currently: <b>{STATUS_LABELS[selected.status] || selected.status}</b>. 
                  {selected.status === 'PE_ASSIGNED' && ' PE investigation is in progress.'}
                  {selected.status === 'PE_SUBMITTED' && ' PE report has been submitted. PI will review.'}
                  {selected.status === 'FIR_REGISTERED' && ' FIR has been registered.'}
                  {selected.status === 'CLOSED_NO_CRIME' && ' Case closed — no crime found.'}
                </div>
              )}
            </div>
            <div className="dash-modal-footer">
              <button className="dash-btn secondary" onClick={() => setSelected(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DSPDashboardPage;
