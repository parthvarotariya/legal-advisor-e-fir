import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import { createChargeSheet, submitChargeSheetToPI, getChargeSheetsByOfficer, updateChargeSheet } from '../../services/chargeSheetService';
import './DashboardCommon.css';

const CATEGORY_LABELS = {
  kidnapping: 'Kidnapping / Abduction / Missing Person (BNS 140–151)',
  sexual_offence: 'Sexual Offences (BNS 63–70)',
  assault: 'Assault / Hurt / Violence (BNS 115–140)',
  women_child_safety: 'Women & Child Safety (BNS 86 + POCSO)',
  harassment: 'Harassment / Threats / Stalking (BNS 351–353)',
  accident: 'Accident / Hit & Run (BNS 106, 112, 279)',
  cybercrime: 'Cybercrime (IT Act + BNS mapping)',
  fraud: 'Fraud / Cheating / Financial Crimes (BNS 318–324)',
  theft: 'Theft & Robbery (BNS 303–309)',
  trespass: 'Trespass / Housebreaking / Property Disputes (BNS 332–335)',
  defamation: 'Defamation / Public Order Offences (BNS 356–357, 147–150)',
  other: 'Other / Cannot Classify'
};

const STATUS_LABELS = {
  RECEIVED: 'Received',
  PE_PENDING_DSP_APPROVAL: 'PE – Awaiting DSP',
  PE_ASSIGNED: 'PE – Assigned',
  PE_SUBMITTED: 'PE – Submitted',
  FIR_REGISTERED: 'FIR Registered',
  CLOSED_NO_CRIME: 'Closed (No Crime)'
};

const PsiDashboardPage = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [pReports, setPReports] = useState([]);
  const [myFirs, setMyFirs] = useState([]);
  const [myChargeSheets, setMyChargeSheets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('assigned');
  const [selected, setSelected] = useState(null);
  const [modalView, setModalView] = useState('details');

  /* ── Charge sheet state ── */
  const [selectedFir, setSelectedFir] = useState(null);
  const [csModalOpen, setCsModalOpen] = useState(false);
  const [csEditId, setCsEditId] = useState(null);
  const [csSubmitting, setCsSubmitting] = useState(false);
  const emptyAccusedRow = { name: '', fatherName: '', dob: '', nationality: 'Indian', religion: '', caste: '', scStStatus: false, occupation: '', arrestDate: '', bailDate: '', suretyDetails: '', dateForwardedToCourt: '' };
  const emptyNotChargedRow = { name: '', fatherName: '', reasonForNotProsecuting: '' };
  const emptyAbscondingRow = { name: '', fatherName: '', lastKnownAddress: '', warrantIssued: false };
  const emptyPropertyRow = { description: '', estimatedValue: '', muddamalNumber: '', psPropertyRegNo: '' };
  const emptyWitnessRow = { serialNo: 1, name: '', fatherName: '', address: '', age: '', evidenceType: 'EYE_WITNESS' };

  const defaultCsForm = {
    reportType: 'CHARGE_SHEET',
    actsAndSections: '',
    briefFacts: '',
    accusedChargeSheeted: [{ ...emptyAccusedRow }],
    accusedNotChargeSheeted: [],
    accusedAbsconding: [],
    seizedProperty: [],
    chainOfCustody: '',
    laboratoryResult: '',
    witnessList: [{ ...emptyWitnessRow }],
    complainantNotified: false,
  };
  const [csForm, setCsForm] = useState({ ...defaultCsForm });

  /* ── PE form state ── */
  const [peForm, setPeForm] = useState({
    investigationNarrative: '',
    cognizableOffence: true,
    informantName: '',
    informantAddress: '',
    informantContact: '',
    informantEmail: '',
    incidentLocation: '',
    incidentDate: '',
    incidentTime: '',
    crimeCategory: '',
    ipcSections: '',
    stolenPropertyDetails: '',
    draftAccusedDetails: '',
    draftWitnessDetails: '',
    witnessStatement: ''
  });

  useEffect(() => {
    const stored = localStorage.getItem('policeUser');
    if (!stored) { navigate('/police/login'); return; }
    const u = JSON.parse(stored);
    if (u.role !== 'INVESTIGATING_OFFICER') {
      if (u.role === 'STATION_ADMIN') navigate('/police/pi-dashboard');
      else if (u.role === 'DEPUTY_SUPRINTENDENT') navigate('/police/dsp-dashboard');
      return;
    }
    setUser(u);
    loadAll(u);
  }, [navigate]);

  const loadAll = async (officer) => {
    setLoading(true);
    try {
      const officerId = officer.policeId;
      const [compRes, peRes, firRes, csRes] = await Promise.all([
        api.get(`/complaints/officer/${officerId}`),
        api.get(`/preliminary-report/officer/${officerId}`),
        api.get(`/fir/officer/${officerId}`),
        getChargeSheetsByOfficer(officerId).catch(() => [])
      ]);
      setComplaints(Array.isArray(compRes.data) ? compRes.data : []);
      setPReports(Array.isArray(peRes.data) ? peRes.data : []);
      setMyFirs(Array.isArray(firRes.data) ? firRes.data : []);
      setMyChargeSheets(Array.isArray(csRes) ? csRes : []);
    } catch (e) {
      console.error(e);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const refresh = useCallback(() => {
    if (user) loadAll(user);
  }, [user]);

  /* ── PE Submission ── */
  const handleSubmitPE = async (complaint) => {
    if (peForm.investigationNarrative.length < 20) {
      alert('Investigation narrative must be at least 20 characters.');
      return;
    }
    try {
      const payload = {
        ...peForm,
        cognizableOffence: peForm.cognizableOffence,
        complaintId: complaint.id,
        investigatingOfficerId: user.policeId,
        stationId: user.stationId || user.station?.id,
        crimeCategory: peForm.crimeCategory || complaint.actualCategory || complaint.predictedCategory || '',
        incidentDate: peForm.incidentDate || null,
        incidentTime: peForm.incidentTime || null
      };
      await api.post('/preliminary-report/create', payload);
      await api.put(`/complaints/${complaint.id}/status`, { status: 'PE_SUBMITTED' });
      setComplaints(prev => prev.map(c => c.id === complaint.id ? { ...c, status: 'PE_SUBMITTED' } : c));
      if (selected?.id === complaint.id) setSelected({ ...selected, status: 'PE_SUBMITTED' });
      alert('PE Report submitted successfully!');
      setModalView('details');
      resetPeForm();
      refresh();
    } catch (e) {
      console.error(e);
      alert('Failed to submit PE: ' + (e.response?.data?.message || e.message));
    }
  };

  const resetPeForm = () => {
    setPeForm({
      investigationNarrative: '', cognizableOffence: true, informantName: '', informantAddress: '',
      informantContact: '', informantEmail: '', incidentLocation: '', incidentDate: '',
      incidentTime: '', crimeCategory: '', ipcSections: '', stolenPropertyDetails: '',
      draftAccusedDetails: '', draftWitnessDetails: '', witnessStatement: ''
    });
  };

  /* ── Charge Sheet helpers ── */
  const generateCsNumber = () => {
    const yr = new Date().getFullYear();
    const ts = Date.now().toString(36).toUpperCase();
    const rand = Math.floor(Math.random() * 10000).toString().padStart(4, '0');
    return `CS-${user?.stationCode || 'STN'}-${yr}-${ts}${rand}`;
  };

  const openChargeSheetForm = (fir) => {
    // Check if there's already a charge sheet for this FIR (DRAFT or RETURNED)
    const existing = myChargeSheets.find(cs => cs.firId === fir.firId && (cs.status === 'DRAFT' || cs.status === 'RETURNED_FOR_REVISION'));
    if (existing) {
      setCsEditId(existing.chargeSheetId);
      setCsForm({
        reportType: existing.reportType || 'CHARGE_SHEET',
        actsAndSections: existing.actsAndSections || fir.ipcSections || '',
        briefFacts: existing.briefFacts || '',
        accusedChargeSheeted: safeParseJson(existing.accusedChargeSheetedJson, [{ ...emptyAccusedRow }]),
        accusedNotChargeSheeted: safeParseJson(existing.accusedNotChargeSheetedJson, []),
        accusedAbsconding: safeParseJson(existing.accusedAbscondingJson, []),
        seizedProperty: safeParseJson(existing.seizedPropertyJson, []),
        chainOfCustody: existing.chainOfCustody || '',
        laboratoryResult: existing.laboratoryResult || '',
        witnessList: safeParseJson(existing.witnessListJson, [{ ...emptyWitnessRow }]),
        complainantNotified: existing.complainantNotified || false,
      });
    } else {
      setCsEditId(null);
      setCsForm({
        ...defaultCsForm,
        actsAndSections: fir.ipcSections || '',
      });
    }
    setSelectedFir(fir);
    setCsModalOpen(true);
  };

  const safeParseJson = (jsonStr, fallback) => {
    if (!jsonStr) return fallback;
    try { return JSON.parse(jsonStr); } catch { return fallback; }
  };

  const handleCsFieldChange = (field, value) => {
    setCsForm(prev => ({ ...prev, [field]: value }));
  };

  const addArrayRow = (field, template) => {
    setCsForm(prev => ({ ...prev, [field]: [...prev[field], { ...template }] }));
  };

  const removeArrayRow = (field, index) => {
    setCsForm(prev => ({ ...prev, [field]: prev[field].filter((_, i) => i !== index) }));
  };

  const updateArrayRow = (field, index, key, value) => {
    setCsForm(prev => ({
      ...prev,
      [field]: prev[field].map((row, i) => i === index ? { ...row, [key]: value } : row)
    }));
  };

  const handleSaveChargeSheet = async (submitToPI = false) => {
    if (!csForm.briefFacts || csForm.briefFacts.length < 20) {
      alert('Brief facts must be at least 20 characters.'); return;
    }
    if (!csForm.actsAndSections) {
      alert('Acts & Sections are required.'); return;
    }

    setCsSubmitting(true);
    let payload = null;
    try {
      const stationId = user?.stationId || user?.station?.id;
      payload = {
        chargeSheetNumber: csEditId ? (myChargeSheets.find(c => c.chargeSheetId === csEditId)?.chargeSheetNumber || generateCsNumber()) : generateCsNumber(),
        district: user?.district || selectedFir?.district || '',
        firId: selectedFir.firId,
        policeStationId: selectedFir.policeStationId || stationId,
        reportType: csForm.reportType,
        actsAndSections: csForm.actsAndSections,
        briefFacts: csForm.briefFacts,
        accusedChargeSheetedJson: JSON.stringify(csForm.accusedChargeSheeted.filter(a => a.name)),
        accusedNotChargeSheetedJson: JSON.stringify(csForm.accusedNotChargeSheeted.filter(a => a.name)),
        accusedAbscondingJson: JSON.stringify(csForm.accusedAbsconding.filter(a => a.name)),
        seizedPropertyJson: JSON.stringify(csForm.seizedProperty.filter(p => p.description)),
        chainOfCustody: csForm.chainOfCustody,
        laboratoryResult: csForm.laboratoryResult,
        witnessListJson: JSON.stringify(csForm.witnessList.filter(w => w.name)),
        complainantNotified: csForm.complainantNotified,
        investigatingOfficerId: user.policeId,
      };

      let saved;
      if (csEditId) {
        saved = await updateChargeSheet(csEditId, payload);
      } else {
        // Check for existing draft one more time (in case list was stale)
        const freshCs = await getChargeSheetsByOfficer(user.policeId).catch(() => []);
        const existingDraft = freshCs.find(cs => cs.firId === selectedFir.firId && (cs.status === 'DRAFT' || cs.status === 'RETURNED_FOR_REVISION'));
        if (existingDraft) {
          // Reuse existing draft instead of creating duplicate
          payload.chargeSheetNumber = existingDraft.chargeSheetNumber;
          saved = await updateChargeSheet(existingDraft.chargeSheetId, payload);
          setCsEditId(existingDraft.chargeSheetId);
        } else {
          saved = await createChargeSheet(payload);
        }
      }

      if (submitToPI && saved?.chargeSheetId) {
        await submitChargeSheetToPI(saved.chargeSheetId);
        alert('Charge sheet submitted to PI for approval!');
      } else {
        alert('Charge sheet saved as draft.');
      }

      setCsModalOpen(false);
      setSelectedFir(null);
      setCsEditId(null);
      setCsForm({ ...defaultCsForm });
      refresh();
    } catch (e) {
      console.error('Charge sheet save error:', e);
      console.error('Response data:', e.response?.data);
      console.error('Payload sent:', JSON.stringify(payload, null, 2));
      const errMsg = e.response?.data?.fieldErrors
        ? Object.entries(e.response.data.fieldErrors).map(([k, v]) => `${k}: ${v}`).join(', ')
        : (e.response?.data?.message || e.message);
      alert('Failed to save charge sheet: ' + errMsg);
      refresh(); // Refresh list so stale drafts become visible
    } finally {
      setCsSubmitting(false);
    }
  };

  const handleResubmitChargeSheet = async (cs) => {
    setCsSubmitting(true);
    try {
      await submitChargeSheetToPI(cs.chargeSheetId);
      alert('Charge sheet re-submitted to PI!');
      refresh();
    } catch (e) {
      alert('Failed: ' + (e.response?.data?.message || e.message));
    } finally {
      setCsSubmitting(false);
    }
  };

  const CS_STATUS_LABELS = {
    DRAFT: 'Draft',
    SUBMITTED_TO_PI: 'Submitted to PI',
    RETURNED_FOR_REVISION: 'Returned – Revision Needed',
    APPROVED_BY_PI: 'Approved by PI',
    DISPATCHED_TO_COURT: 'Dispatched to Court',
  };

  const peAssigned = complaints.filter(c => c.status === 'PE_ASSIGNED');
  const peSubmitted = complaints.filter(c => c.status === 'PE_SUBMITTED');

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
        <div className="dash-loading"><div className="dash-spinner" /><p>Loading PSI Dashboard…</p></div>
      </div>
    );
  }

  return (
    <div className="dash-page psi-theme">
      {/* Navbar */}
      <nav className="dash-nav">
        <div className="dash-nav-left">
          <span className="dash-nav-icon">🔍</span>
          <div>
            <h1 className="dash-nav-title">PSI Dashboard</h1>
            <span className="dash-nav-sub">Police Sub-Inspector – Investigating Officer</span>
          </div>
        </div>
        <div className="dash-nav-right">
          <span className="dash-nav-user">👮 {user?.name}</span>
          <span className="dash-nav-badge">Badge: {user?.badgeNumber}</span>
          <span className="dash-role-chip psi">PSI</span>
          <button onClick={handleLogout} className="dash-btn-logout">Logout</button>
        </div>
      </nav>

      <div className="dash-body">
        {error && <div className="dash-alert error">{error}</div>}

        <div className="dash-station-card">
          <h3>🏛️ {user?.stationName || 'Station'}</h3>
          <p>Code: {user?.stationCode} &bull; Role: PSI (Investigating Officer)</p>
        </div>

        {/* Stats */}
        <div className="dash-stats">
          <div className="dash-stat" data-accent="blue">
            <span className="dash-stat-icon">📋</span>
            <div><h3>{complaints.length}</h3><p>My Cases</p></div>
          </div>
          <div className="dash-stat" data-accent="orange">
            <span className="dash-stat-icon">🔍</span>
            <div><h3>{peAssigned.length}</h3><p>PE – To Investigate</p></div>
          </div>
          <div className="dash-stat" data-accent="green">
            <span className="dash-stat-icon">📄</span>
            <div><h3>{peSubmitted.length}</h3><p>PE – Submitted</p></div>
          </div>
          <div className="dash-stat" data-accent="purple">
            <span className="dash-stat-icon">📜</span>
            <div><h3>{myFirs.length}</h3><p>FIR Investigations</p></div>
          </div>
          <div className="dash-stat" data-accent="red">
            <span className="dash-stat-icon">📑</span>
            <div><h3>{myChargeSheets.length}</h3><p>Charge Sheets</p></div>
          </div>
        </div>

        {/* Tabs */}
        <div className="dash-tabs">
          <button className={`dash-tab ${activeTab === 'assigned' ? 'active' : ''}`} onClick={() => setActiveTab('assigned')}>
            🔍 PE Investigations ({peAssigned.length})
          </button>
          <button className={`dash-tab ${activeTab === 'submitted' ? 'active' : ''}`} onClick={() => setActiveTab('submitted')}>
            📄 PE Submitted ({peSubmitted.length})
          </button>
          <button className={`dash-tab ${activeTab === 'firs' ? 'active' : ''}`} onClick={() => setActiveTab('firs')}>
            📜 My FIRs ({myFirs.length})
          </button>
          <button className={`dash-tab ${activeTab === 'chargeSheets' ? 'active' : ''}`} onClick={() => setActiveTab('chargeSheets')}>
            📑 Charge Sheets ({myChargeSheets.length})
          </button>
          <button className={`dash-tab ${activeTab === 'all' ? 'active' : ''}`} onClick={() => setActiveTab('all')}>
            📋 All Cases ({complaints.length})
          </button>
        </div>

        {/* PE Assigned */}
        {activeTab === 'assigned' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>🔍 PE Investigations — Action Needed</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {peAssigned.length === 0 ? (
              <div className="dash-empty">No PE investigations pending.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>ID</th><th>Category</th><th>Description</th><th>Filed</th><th>Action</th></tr></thead>
                  <tbody>
                    {peAssigned.map(c => (
                      <tr key={c.id}>
                        <td><span className="dash-id">#{c.id}</span></td>
                        <td><span className="dash-chip cat">{c.actualCategory || c.predictedCategory || '—'}</span></td>
                        <td className="dash-desc">{c.description?.substring(0, 60)}{c.description?.length > 60 ? '…' : ''}</td>
                        <td className="dash-date">{formatDate(c.createdAt)}</td>
                        <td><button className="dash-btn primary sm" onClick={() => { 
                          setSelected(c); 
                          setModalView('pe'); 
                          setPeForm({
                            investigationNarrative: '', 
                            cognizableOffence: true, 
                            informantName: c.complainantName || '', 
                            informantAddress: c.complainantAddress || '',
                            informantContact: c.complainantMobile || '', 
                            informantEmail: c.complainantEmail || '', 
                            incidentLocation: '', 
                            incidentDate: '',
                            incidentTime: '', 
                            crimeCategory: c.actualCategory || c.predictedCategory || '', 
                            ipcSections: '', 
                            stolenPropertyDetails: '',
                            draftAccusedDetails: '', 
                            draftWitnessDetails: '', 
                            witnessStatement: ''
                          });
                        }}>📝 Write PE</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* PE Submitted */}
        {activeTab === 'submitted' && (
          <div className="dash-card">
            <div className="dash-card-header"><h2>📄 Submitted PE Reports</h2></div>
            {peSubmitted.length === 0 ? (
              <div className="dash-empty">No PE reports submitted yet.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>Complaint ID</th><th>Category</th><th>Status</th><th>Filed</th><th>Action</th></tr></thead>
                  <tbody>
                    {peSubmitted.map(c => (
                      <tr key={c.id}>
                        <td><span className="dash-id">#{c.id}</span></td>
                        <td><span className="dash-chip cat">{c.actualCategory || c.predictedCategory || '—'}</span></td>
                        <td><span className="dash-status pe-submitted">PE Submitted</span></td>
                        <td className="dash-date">{formatDate(c.createdAt)}</td>
                        <td><button className="dash-btn primary sm" onClick={() => { setSelected(c); setModalView('details'); }}>View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* My FIRs */}
        {activeTab === 'firs' && (
          <div className="dash-card">
            <div className="dash-card-header"><h2>📜 My FIR Investigations</h2></div>
            {myFirs.length === 0 ? (
              <div className="dash-empty">No FIR investigations assigned.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>FIR #</th><th>Crime Category</th><th>Informant</th><th>Status</th><th>Registered</th><th>Action</th></tr></thead>
                  <tbody>
                    {myFirs.map(f => {
                      const existingCs = myChargeSheets.find(cs => cs.firId === f.firId);
                      return (
                        <tr key={f.firId}>
                          <td><span className="dash-id">{f.firNumber}</span></td>
                          <td>{f.crimeCategory}</td>
                          <td>{f.informantName}</td>
                          <td><span className={`dash-status ${f.status?.toLowerCase().replace(/_/g, '-')}`}>{f.status?.replace(/_/g, ' ')}</span></td>
                          <td className="dash-date">{formatDate(f.registeredAt)}</td>
                          <td>
                            {!existingCs && (
                              <button className="dash-btn primary sm" onClick={() => openChargeSheetForm(f)}>📑 File Report</button>
                            )}
                            {existingCs && (existingCs.status === 'DRAFT' || existingCs.status === 'RETURNED_FOR_REVISION') && (
                              <button className="dash-btn warning sm" onClick={() => openChargeSheetForm(f)}>✏️ Edit Report</button>
                            )}
                            {existingCs && existingCs.status === 'SUBMITTED_TO_PI' && (
                              <span className="dash-status pe-submitted">Awaiting PI</span>
                            )}
                            {existingCs && existingCs.status === 'APPROVED_BY_PI' && (
                              <span className="dash-status" style={{ background: '#d4edda', color: '#155724' }}>Approved</span>
                            )}
                            {existingCs && existingCs.status === 'DISPATCHED_TO_COURT' && (
                              <span className="dash-status" style={{ background: '#cce5ff', color: '#004085' }}>Filed in Court</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Charge Sheets Tab */}
        {activeTab === 'chargeSheets' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>📑 My Charge Sheets / Final Reports</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {myChargeSheets.length === 0 ? (
              <div className="dash-empty">No charge sheets filed yet. Go to "My FIRs" tab to file one.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>CS #</th><th>FIR #</th><th>Type</th><th>Status</th><th>Submitted</th><th>Action</th></tr></thead>
                  <tbody>
                    {myChargeSheets.map(cs => (
                      <tr key={cs.chargeSheetId}>
                        <td><span className="dash-id">{cs.chargeSheetNumber}</span></td>
                        <td>{cs.firNumber}</td>
                        <td><span className="dash-chip cat">{cs.reportType?.replace(/_/g, ' ')}</span></td>
                        <td><span className={`dash-status ${cs.status?.toLowerCase().replace(/_/g, '-')}`}>{CS_STATUS_LABELS[cs.status] || cs.status}</span></td>
                        <td className="dash-date">{cs.submittedAt ? formatDate(cs.submittedAt) : '-'}</td>
                        <td>
                          {(cs.status === 'DRAFT' || cs.status === 'RETURNED_FOR_REVISION') && (
                            <>
                              <button className="dash-btn primary sm" onClick={() => {
                                const fir = myFirs.find(f => f.firId === cs.firId);
                                if (fir) openChargeSheetForm(fir);
                              }}>✏️ Edit</button>
                              {cs.status === 'RETURNED_FOR_REVISION' && cs.piSuggestions && (
                                <div style={{ marginTop: 4, padding: '4px 8px', background: '#fff3cd', borderRadius: 4, fontSize: 12, color: '#856404' }}>
                                  <strong>PI Notes:</strong> {cs.piSuggestions}
                                </div>
                              )}
                            </>
                          )}
                          {cs.status === 'SUBMITTED_TO_PI' && <span style={{ color: '#6c757d', fontSize: 13 }}>Pending Review</span>}
                          {cs.status === 'APPROVED_BY_PI' && <span style={{ color: '#28a745', fontSize: 13 }}>✅ Approved</span>}
                          {cs.status === 'DISPATCHED_TO_COURT' && <span style={{ color: '#007bff', fontSize: 13 }}>📤 In Court</span>}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* All Cases */}
        {activeTab === 'all' && (
          <div className="dash-card">
            <div className="dash-card-header"><h2>📋 All My Assigned Cases</h2></div>
            {complaints.length === 0 ? (
              <div className="dash-empty">No cases assigned to you yet.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>ID</th><th>Category</th><th>Description</th><th>Status</th><th>Filed</th><th>Action</th></tr></thead>
                  <tbody>
                    {complaints.map(c => (
                      <tr key={c.id}>
                        <td><span className="dash-id">#{c.id}</span></td>
                        <td><span className="dash-chip cat">{c.actualCategory || c.predictedCategory || '—'}</span></td>
                        <td className="dash-desc">{c.description?.substring(0, 55)}{c.description?.length > 55 ? '…' : ''}</td>
                        <td><span className={`dash-status ${c.status?.toLowerCase().replace(/_/g, '-')}`}>{STATUS_LABELS[c.status] || c.status}</span></td>
                        <td className="dash-date">{formatDate(c.createdAt)}</td>
                        <td><button className="dash-btn primary sm" onClick={() => { setSelected(c); setModalView('details'); }}>View</button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Modal */}
      {selected && (
        <div className="dash-overlay" onClick={() => setSelected(null)}>
          <div className="dash-modal lg" onClick={e => e.stopPropagation()}>
            <div className="dash-modal-header">
              <h2>Complaint #{selected.id}</h2>
              <button className="dash-modal-close" onClick={() => setSelected(null)}>×</button>
            </div>

            <div className="dash-modal-tabs">
              <button className={modalView === 'details' ? 'active' : ''} onClick={() => setModalView('details')}>Details</button>
              {selected.status === 'PE_ASSIGNED' && (
                <button className={modalView === 'pe' ? 'active' : ''} onClick={() => setModalView('pe')}>Write PE Report</button>
              )}
            </div>

            <div className="dash-modal-body">
              {modalView === 'details' && (
                <>
                  <div className="dash-detail"><label>Status</label><span className={`dash-status ${selected.status?.toLowerCase().replace(/_/g, '-')}`}>{STATUS_LABELS[selected.status] || selected.status}</span></div>
                  <div className="dash-detail"><label>Category</label><span className="dash-chip cat">{selected.actualCategory || selected.predictedCategory || '—'}</span></div>
                  <div className="dash-detail"><label>Police Station</label><span>{selected.policeStationName || '—'}</span></div>
                  <div className="dash-detail"><label>Filed On</label><span>{formatDate(selected.createdAt)}</span></div>
                  
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
                  
                  <div className="dash-detail col"><label>Description</label><div className="dash-desc-box">{selected.description}</div></div>
                </>
              )}

              {modalView === 'pe' && selected.status === 'PE_ASSIGNED' && (
                <>
                  {/* Complaint Summary at Top */}
                  <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px', padding: '14px 18px', marginBottom: '18px' }}>
                    <h4 style={{ margin: '0 0 10px', fontSize: '14px', color: '#1e293b' }}>📋 Complaint Details</h4>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '13px' }}>
                      <div><strong>ID:</strong> #{selected.id}</div>
                      <div><strong>Category:</strong> {selected.actualCategory || selected.predictedCategory || '—'}</div>
                      <div><strong>Complainant:</strong> {selected.complainantName || '—'}</div>
                      <div><strong>Mobile:</strong> {selected.complainantMobile || '—'}</div>
                      <div style={{ gridColumn: '1 / -1' }}><strong>Description:</strong> {selected.description}</div>
                    </div>
                  </div>
                  
                  <div className="dash-pe-form">
                  <h3>Preliminary Enquiry Report</h3>
                  <p className="dash-form-hint">Fill in the investigation findings. All fields marked * are required.</p>

                  <div className="dash-form-group">
                    <label>Investigation Narrative *</label>
                    <textarea value={peForm.investigationNarrative} onChange={e => setPeForm({ ...peForm, investigationNarrative: e.target.value })} placeholder="Describe findings (min 20 chars)..." rows={5} />
                  </div>

                  <div className="dash-form-row">
                    <div className="dash-form-group">
                      <label>Cognizable Offence? *</label>
                      <select value={peForm.cognizableOffence} onChange={e => setPeForm({ ...peForm, cognizableOffence: e.target.value === 'true' })}>
                        <option value="true">Yes — Cognizable</option>
                        <option value="false">No — Non-Cognizable</option>
                      </select>
                    </div>
                    <div className="dash-form-group">
                      <label>Crime Category *</label>
                      <select value={peForm.crimeCategory} onChange={e => setPeForm({ ...peForm, crimeCategory: e.target.value })}>
                        <option value="">— Select —</option>
                        {Object.entries(CATEGORY_LABELS).map(([k, v]) => (<option key={k} value={v}>{v}</option>))}
                      </select>
                    </div>
                  </div>

                  <div className="dash-form-row">
                    <div className="dash-form-group"><label>Informant Name *</label><input value={peForm.informantName} onChange={e => setPeForm({ ...peForm, informantName: e.target.value })} placeholder="Full name" /></div>
                    <div className="dash-form-group"><label>Informant Contact</label><input value={peForm.informantContact} onChange={e => setPeForm({ ...peForm, informantContact: e.target.value })} placeholder="10-digit mobile" /></div>
                  </div>

                  <div className="dash-form-group"><label>Informant Address *</label><input value={peForm.informantAddress} onChange={e => setPeForm({ ...peForm, informantAddress: e.target.value })} placeholder="Full address" /></div>
                  <div className="dash-form-group"><label>Informant Email</label><input type="email" value={peForm.informantEmail} onChange={e => setPeForm({ ...peForm, informantEmail: e.target.value })} placeholder="email@example.com" /></div>

                  <div className="dash-form-row">
                    <div className="dash-form-group"><label>Incident Location *</label><input value={peForm.incidentLocation} onChange={e => setPeForm({ ...peForm, incidentLocation: e.target.value })} placeholder="Location" /></div>
                    <div className="dash-form-group"><label>Incident Date</label><input type="date" value={peForm.incidentDate} onChange={e => setPeForm({ ...peForm, incidentDate: e.target.value })} /></div>
                    <div className="dash-form-group"><label>Incident Time</label><input type="time" value={peForm.incidentTime} onChange={e => setPeForm({ ...peForm, incidentTime: e.target.value })} /></div>
                  </div>

                  <div className="dash-form-group"><label>IPC / BNS Sections</label><input value={peForm.ipcSections} onChange={e => setPeForm({ ...peForm, ipcSections: e.target.value })} placeholder="e.g. BNS 303, 304" /></div>
                  <div className="dash-form-group"><label>Stolen Property Details</label><textarea value={peForm.stolenPropertyDetails} onChange={e => setPeForm({ ...peForm, stolenPropertyDetails: e.target.value })} rows={2} placeholder="If applicable" /></div>
                  <div className="dash-form-group"><label>Draft Accused Details</label><textarea value={peForm.draftAccusedDetails} onChange={e => setPeForm({ ...peForm, draftAccusedDetails: e.target.value })} rows={2} placeholder="Names, descriptions..." /></div>
                  <div className="dash-form-group"><label>Witness Details</label><textarea value={peForm.draftWitnessDetails} onChange={e => setPeForm({ ...peForm, draftWitnessDetails: e.target.value })} rows={2} placeholder="Names, contact info..." /></div>
                  <div className="dash-form-group"><label>Witness Statement</label><textarea value={peForm.witnessStatement} onChange={e => setPeForm({ ...peForm, witnessStatement: e.target.value })} rows={3} placeholder="Recorded witness statements..." /></div>

                  <div className="dash-action-row" style={{ marginTop: 16 }}>
                    <button className="dash-btn success" onClick={() => handleSubmitPE(selected)}>📤 Submit PE Report</button>
                    <button className="dash-btn secondary" onClick={() => setModalView('details')}>Cancel</button>
                  </div>
                </div>
                </>
              )}
            </div>

            <div className="dash-modal-footer">
              <button className="dash-btn secondary" onClick={() => setSelected(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
      {/* Charge Sheet Modal */}
      {csModalOpen && selectedFir && (
        <div className="dash-overlay" onClick={() => { setCsModalOpen(false); setSelectedFir(null); }}>
          <div className="dash-modal lg" style={{ maxWidth: 900, maxHeight: '92vh', overflow: 'auto' }} onClick={e => e.stopPropagation()}>
            <div className="dash-modal-header">
              <h2>📑 {csEditId ? 'Edit' : 'New'} Final Report — FIR {selectedFir.firNumber}</h2>
              <button className="dash-modal-close" onClick={() => { setCsModalOpen(false); setSelectedFir(null); }}>×</button>
            </div>
            <div className="dash-modal-body">
              {/* PI Suggestions Banner */}
              {(() => {
                const existing = myChargeSheets.find(cs => cs.firId === selectedFir.firId && cs.status === 'RETURNED_FOR_REVISION');
                if (existing?.piSuggestions) return (
                  <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 8, padding: '12px 16px', marginBottom: 16 }}>
                    <strong>⚠️ PI Suggestions (Revision #{existing.revisionCount}):</strong>
                    <p style={{ margin: '6px 0 0', whiteSpace: 'pre-wrap' }}>{existing.piSuggestions}</p>
                  </div>
                );
                return null;
              })()}

              {/* FIR Summary */}
              <div style={{ background: '#f0f4ff', border: '1px solid #c3d4ff', borderRadius: 8, padding: 14, marginBottom: 16 }}>
                <h4 style={{ margin: '0 0 8px', fontSize: 14 }}>📋 FIR Details</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, fontSize: 13 }}>
                  <div><strong>FIR #:</strong> {selectedFir.firNumber}</div>
                  <div><strong>Crime:</strong> {selectedFir.crimeCategory}</div>
                  <div><strong>Informant:</strong> {selectedFir.informantName}</div>
                  <div><strong>Station:</strong> {selectedFir.policeStationName}</div>
                  <div style={{ gridColumn: '1/-1' }}><strong>Description:</strong> {selectedFir.incidentDescription?.substring(0, 200)}{selectedFir.incidentDescription?.length > 200 ? '...' : ''}</div>
                </div>
              </div>

              {/* 1. Report Type */}
              <div className="dash-form-group">
                <label><strong>1. Report Type *</strong></label>
                <select value={csForm.reportType} onChange={e => handleCsFieldChange('reportType', e.target.value)}>
                  <option value="CHARGE_SHEET">Charge Sheet</option>
                  <option value="CLOSURE_UNTRACED">Closure – Untraced</option>
                  <option value="CLOSURE_UNOCCURRED">Closure – Unoccurred</option>
                  <option value="CLOSURE_ABATED_DEATH">Closure – Abated (Death of Accused)</option>
                  <option value="CLOSURE_INSUFFICIENT_EVIDENCE">Closure – Insufficient Evidence</option>
                </select>
              </div>

              {/* 2. Acts & Sections */}
              <div className="dash-form-group">
                <label><strong>2. Acts & Sections (BNS / IPC / Special Acts) *</strong></label>
                <input value={csForm.actsAndSections} onChange={e => handleCsFieldChange('actsAndSections', e.target.value)} placeholder="e.g. BNS 303, 304, Arms Act Sec 25" />
              </div>

              {/* 3. Brief Facts */}
              <div className="dash-form-group">
                <label><strong>3. Brief Facts of the Case *</strong></label>
                <textarea value={csForm.briefFacts} onChange={e => handleCsFieldChange('briefFacts', e.target.value)} rows={6} placeholder="Chronological narrative of investigation findings and motive..." />
              </div>

              {/* 4. Accused Charge-sheeted */}
              {csForm.reportType === 'CHARGE_SHEET' && (
                <>
                  <div style={{ borderTop: '2px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label><strong>4. Accused — Charge-sheeted</strong></label>
                      <button className="dash-btn primary sm" onClick={() => addArrayRow('accusedChargeSheeted', emptyAccusedRow)}>+ Add Accused</button>
                    </div>
                    {csForm.accusedChargeSheeted.map((a, i) => (
                      <div key={i} style={{ background: '#fafafa', border: '1px solid #e2e8f0', borderRadius: 8, padding: 12, marginTop: 8, position: 'relative' }}>
                        <button onClick={() => removeArrayRow('accusedChargeSheeted', i)} style={{ position: 'absolute', top: 4, right: 8, background: 'none', border: 'none', color: '#e74c3c', cursor: 'pointer', fontSize: 16 }}>×</button>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div className="dash-form-group"><label>Name *</label><input value={a.name} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'name', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Father/Husband Name</label><input value={a.fatherName} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'fatherName', e.target.value)} /></div>
                          <div className="dash-form-group"><label>DOB / Birth Year</label><input value={a.dob} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'dob', e.target.value)} placeholder="DD/MM/YYYY" /></div>
                          <div className="dash-form-group"><label>Nationality</label><input value={a.nationality} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'nationality', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Religion</label><input value={a.religion} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'religion', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Caste</label><input value={a.caste} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'caste', e.target.value)} /></div>
                          <div className="dash-form-group"><label>SC/ST Status</label>
                            <select value={a.scStStatus ? 'true' : 'false'} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'scStStatus', e.target.value === 'true')}>
                              <option value="false">No</option><option value="true">Yes</option>
                            </select>
                          </div>
                          <div className="dash-form-group"><label>Occupation</label><input value={a.occupation} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'occupation', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Arrest Date</label><input type="date" value={a.arrestDate} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'arrestDate', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Bail Date</label><input type="date" value={a.bailDate} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'bailDate', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Surety Details</label><input value={a.suretyDetails} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'suretyDetails', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Date Forwarded to Court</label><input type="date" value={a.dateForwardedToCourt} onChange={e => updateArrayRow('accusedChargeSheeted', i, 'dateForwardedToCourt', e.target.value)} /></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Accused Not Charge-sheeted */}
                  <div style={{ borderTop: '1px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label><strong>Accused — Not Charge-sheeted</strong></label>
                      <button className="dash-btn secondary sm" onClick={() => addArrayRow('accusedNotChargeSheeted', emptyNotChargedRow)}>+ Add</button>
                    </div>
                    {csForm.accusedNotChargeSheeted.map((a, i) => (
                      <div key={i} style={{ background: '#fafafa', border: '1px solid #e2e8f0', borderRadius: 6, padding: 10, marginTop: 6, position: 'relative' }}>
                        <button onClick={() => removeArrayRow('accusedNotChargeSheeted', i)} style={{ position: 'absolute', top: 4, right: 8, background: 'none', border: 'none', color: '#e74c3c', cursor: 'pointer' }}>×</button>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div className="dash-form-group"><label>Name</label><input value={a.name} onChange={e => updateArrayRow('accusedNotChargeSheeted', i, 'name', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Father Name</label><input value={a.fatherName} onChange={e => updateArrayRow('accusedNotChargeSheeted', i, 'fatherName', e.target.value)} /></div>
                        </div>
                        <div className="dash-form-group"><label>Reason for Not Prosecuting</label><input value={a.reasonForNotProsecuting} onChange={e => updateArrayRow('accusedNotChargeSheeted', i, 'reasonForNotProsecuting', e.target.value)} /></div>
                      </div>
                    ))}
                  </div>

                  {/* Accused Absconding */}
                  <div style={{ borderTop: '1px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <label><strong>Accused — Absconding</strong> <span style={{ color: '#e74c3c', fontSize: 12 }}>(Red Flag)</span></label>
                      <button className="dash-btn secondary sm" onClick={() => addArrayRow('accusedAbsconding', emptyAbscondingRow)}>+ Add</button>
                    </div>
                    {csForm.accusedAbsconding.map((a, i) => (
                      <div key={i} style={{ background: '#fff5f5', border: '1px solid #fc8181', borderRadius: 6, padding: 10, marginTop: 6, position: 'relative' }}>
                        <button onClick={() => removeArrayRow('accusedAbsconding', i)} style={{ position: 'absolute', top: 4, right: 8, background: 'none', border: 'none', color: '#e74c3c', cursor: 'pointer' }}>×</button>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                          <div className="dash-form-group"><label>Name</label><input value={a.name} onChange={e => updateArrayRow('accusedAbsconding', i, 'name', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Father Name</label><input value={a.fatherName} onChange={e => updateArrayRow('accusedAbsconding', i, 'fatherName', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Last Known Address</label><input value={a.lastKnownAddress} onChange={e => updateArrayRow('accusedAbsconding', i, 'lastKnownAddress', e.target.value)} /></div>
                          <div className="dash-form-group"><label>Warrant Issued?</label>
                            <select value={a.warrantIssued ? 'true' : 'false'} onChange={e => updateArrayRow('accusedAbsconding', i, 'warrantIssued', e.target.value === 'true')}>
                              <option value="false">No</option><option value="true">Yes</option>
                            </select>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* 5. Evidence & Recovery */}
              <div style={{ borderTop: '2px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label><strong>5. Seized Property / Articles (Panchnama)</strong></label>
                  <button className="dash-btn secondary sm" onClick={() => addArrayRow('seizedProperty', emptyPropertyRow)}>+ Add Item</button>
                </div>
                {csForm.seizedProperty.map((p, i) => (
                  <div key={i} style={{ background: '#fafafa', border: '1px solid #e2e8f0', borderRadius: 6, padding: 10, marginTop: 6, position: 'relative' }}>
                    <button onClick={() => removeArrayRow('seizedProperty', i)} style={{ position: 'absolute', top: 4, right: 8, background: 'none', border: 'none', color: '#e74c3c', cursor: 'pointer' }}>×</button>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div className="dash-form-group"><label>Description</label><input value={p.description} onChange={e => updateArrayRow('seizedProperty', i, 'description', e.target.value)} /></div>
                      <div className="dash-form-group"><label>Estimated Value (Rs.)</label><input value={p.estimatedValue} onChange={e => updateArrayRow('seizedProperty', i, 'estimatedValue', e.target.value)} /></div>
                      <div className="dash-form-group"><label>Muddamal Number</label><input value={p.muddamalNumber} onChange={e => updateArrayRow('seizedProperty', i, 'muddamalNumber', e.target.value)} /></div>
                      <div className="dash-form-group"><label>PS Property Reg No.</label><input value={p.psPropertyRegNo} onChange={e => updateArrayRow('seizedProperty', i, 'psPropertyRegNo', e.target.value)} /></div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="dash-form-group" style={{ marginTop: 12 }}>
                <label><strong>Chain of Custody (Sec 193 BNSS — Electronic Evidence)</strong></label>
                <textarea value={csForm.chainOfCustody} onChange={e => handleCsFieldChange('chainOfCustody', e.target.value)} rows={3} placeholder="Describe handling of digital/electronic evidence from seizure to lab..." />
              </div>

              <div className="dash-form-group">
                <label><strong>FSL / Laboratory Result</strong></label>
                <textarea value={csForm.laboratoryResult} onChange={e => handleCsFieldChange('laboratoryResult', e.target.value)} rows={3} placeholder="Summary of forensic laboratory findings..." />
              </div>

              {/* 6. Witnesses */}
              <div style={{ borderTop: '2px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <label><strong>6. Witness List</strong></label>
                  <button className="dash-btn secondary sm" onClick={() => addArrayRow('witnessList', { ...emptyWitnessRow, serialNo: csForm.witnessList.length + 1 })}>+ Add Witness</button>
                </div>
                {csForm.witnessList.map((w, i) => (
                  <div key={i} style={{ background: '#fafafa', border: '1px solid #e2e8f0', borderRadius: 6, padding: 10, marginTop: 6, position: 'relative' }}>
                    <button onClick={() => removeArrayRow('witnessList', i)} style={{ position: 'absolute', top: 4, right: 8, background: 'none', border: 'none', color: '#e74c3c', cursor: 'pointer' }}>×</button>
                    <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr 1fr auto 1fr', gap: 8 }}>
                      <div className="dash-form-group"><label>S.No</label><input value={w.serialNo} onChange={e => updateArrayRow('witnessList', i, 'serialNo', e.target.value)} style={{ width: 50 }} /></div>
                      <div className="dash-form-group"><label>Name *</label><input value={w.name} onChange={e => updateArrayRow('witnessList', i, 'name', e.target.value)} /></div>
                      <div className="dash-form-group"><label>Father Name</label><input value={w.fatherName} onChange={e => updateArrayRow('witnessList', i, 'fatherName', e.target.value)} /></div>
                      <div className="dash-form-group"><label>Address</label><input value={w.address} onChange={e => updateArrayRow('witnessList', i, 'address', e.target.value)} /></div>
                      <div className="dash-form-group"><label>Age</label><input value={w.age} onChange={e => updateArrayRow('witnessList', i, 'age', e.target.value)} style={{ width: 50 }} /></div>
                      <div className="dash-form-group"><label>Evidence Type</label>
                        <select value={w.evidenceType} onChange={e => updateArrayRow('witnessList', i, 'evidenceType', e.target.value)}>
                          <option value="EYE_WITNESS">Eye Witness</option>
                          <option value="SEIZURE_WITNESS">Seizure Witness</option>
                          <option value="MEDICAL_WITNESS">Medical Witness</option>
                          <option value="EXPERT">Expert Witness</option>
                          <option value="PANCH_WITNESS">Panch Witness</option>
                          <option value="CIRCUMSTANTIAL">Circumstantial</option>
                          <option value="OTHER">Other</option>
                        </select>
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* 7. Verification */}
              <div style={{ borderTop: '2px solid #e2e8f0', marginTop: 16, paddingTop: 12 }}>
                <label><strong>7. Verification & Dispatch</strong></label>
                <div className="dash-form-group" style={{ marginTop: 8 }}>
                  <label style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <input type="checkbox" checked={csForm.complainantNotified} onChange={e => handleCsFieldChange('complainantNotified', e.target.checked)} />
                    Complainant / Informant has been notified of this final report (Refer Notice)
                  </label>
                </div>
                <div style={{ background: '#f8fafc', padding: 10, borderRadius: 6, fontSize: 13, marginTop: 8 }}>
                  <div><strong>IO:</strong> {user?.name} (Badge: {user?.badgeNumber}, Rank: {user?.rank || 'PSI'})</div>
                  <div><strong>Station:</strong> {user?.stationName} ({user?.stationCode})</div>
                </div>
              </div>

              {/* Actions */}
              <div className="dash-action-row" style={{ marginTop: 20, gap: 10 }}>
                <button className="dash-btn secondary" disabled={csSubmitting} onClick={() => handleSaveChargeSheet(false)}>
                  {csSubmitting ? 'Saving...' : '💾 Save as Draft'}
                </button>
                <button className="dash-btn success" disabled={csSubmitting} onClick={() => handleSaveChargeSheet(true)}>
                  {csSubmitting ? 'Submitting...' : '📤 Submit to PI for Approval'}
                </button>
                <button className="dash-btn secondary" onClick={() => { setCsModalOpen(false); setSelectedFir(null); }}>Cancel</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default PsiDashboardPage;
