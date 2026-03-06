import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../../services/api';
import { generateFirPdf } from '../../utils/generateFirPdf';
import { getPendingChargeSheetsByStation, getChargeSheetsByStation, approveChargeSheet, returnChargeSheet, dispatchChargeSheet } from '../../services/chargeSheetService';
import './DashboardCommon.css';

/* ── Crime Category Routing Rules ── */
const DIRECT_FIR_CATEGORIES = [
  'kidnapping', 'sexual_offence', 'assault', 'accident', 'theft'
];
const PE_RECOMMENDED_CATEGORIES = ['fraud', 'trespass'];
const CONDITIONAL_CATEGORIES = ['women_child_safety', 'harassment', 'cybercrime'];
const NON_COGNIZABLE_CATEGORIES = ['defamation'];

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

const ALL_CATEGORIES = Object.keys(CATEGORY_LABELS);

function getRouteAdvice(category) {
  if (!category) return { type: 'unknown', label: 'Classify first', color: '#95a5a6' };
  const cat = category.toLowerCase().replace(/\s+/g, '_');
  if (DIRECT_FIR_CATEGORIES.includes(cat))
    return { type: 'direct_fir', label: 'Direct FIR (Mandatory)', color: '#e74c3c' };
  if (PE_RECOMMENDED_CATEGORIES.includes(cat))
    return { type: 'pe_recommended', label: 'PE Recommended', color: '#f39c12' };
  if (CONDITIONAL_CATEGORIES.includes(cat))
    return { type: 'conditional', label: 'Conditional – Review needed', color: '#e67e22' };
  if (NON_COGNIZABLE_CATEGORIES.includes(cat))
    return { type: 'non_cognizable', label: 'Non-Cognizable (NCR)', color: '#7f8c8d' };
  return { type: 'unknown', label: 'Review required', color: '#3498db' };
}

const STATUS_LABELS = {
  RECEIVED: 'Received',
  PE_PENDING_DSP_APPROVAL: 'PE – Awaiting DSP',
  PE_ASSIGNED: 'PE – Assigned to PSI',
  PE_SUBMITTED: 'PE – Report Submitted',
  FIR_REGISTERED: 'FIR Registered',
  CLOSED_NO_CRIME: 'Closed (No Crime)'
};

/* Helper: generate FIR number */
const generateFirNumber = (stationCode) => {
  const yr = new Date().getFullYear();
  const rand = Math.floor(1000 + Math.random() * 9000);
  return `FIR-${stationCode || 'STN'}-${yr}-${rand}`;
};

/* Default FIR form state */
const emptyFirForm = {
  firNumber: '',
  district: '',
  informantName: '',
  informantGuardianName: '',
  informantAddress: '',
  informantContact: '',
  informantEmail: '',
  incidentLocation: '',
  incidentDate: '',
  incidentTime: '',
  incidentDescription: '',
  crimeCategory: '',
  ipcSections: '',
  stolenPropertyDetails: '',
  accusedDetails: '',
  witnessDetails: '',
  firWrittenBy: '',
  investigatingOfficerId: '',
  isEfir: true,
  isZeroFir: false,
  isVictimWoman: false,
  isDisabledVictim: false,
  recordedByWomanOfficer: false,
  isMagistrateStatementRecorded: false,
};

const PIDashboardPage = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(null);
  const [complaints, setComplaints] = useState([]);
  const [firs, setFirs] = useState([]);
  const [subordinates, setSubordinates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('complaints');
  const [filterStatus, setFilterStatus] = useState('ALL');

  /* Modal state */
  const [selected, setSelected] = useState(null);
  const [modalTab, setModalTab] = useState('details');
  const [categoryOverride, setCategoryOverride] = useState('');
  const [assignOfficerId, setAssignOfficerId] = useState('');

  /* PE Report state */
  const [peReport, setPeReport] = useState(null);
  const [peLoading, setPeLoading] = useState(false);

  /* FIR detail modal state */
  const [selectedFir, setSelectedFir] = useState(null);

  /* FIR form state */
  const [showFirView, setShowFirView] = useState(false); // split-view mode
  const [firForm, setFirForm] = useState({ ...emptyFirForm });
  const [firSubmitting, setFirSubmitting] = useState(false);
  const [firMode, setFirMode] = useState('direct'); // 'direct' or 'from_pe'

  /* ── Charge Sheet review state ── */
  const [chargeSheets, setChargeSheets] = useState([]);
  const [selectedCs, setSelectedCs] = useState(null);
  const [csReviewOpen, setCsReviewOpen] = useState(false);
  const [returnSuggestions, setReturnSuggestions] = useState('');
  const [csActionLoading, setCsActionLoading] = useState(false);

  /* ── Auth & data loading ── */
  useEffect(() => {
    const stored = localStorage.getItem('policeUser');
    if (!stored) { navigate('/police/login'); return; }
    const u = JSON.parse(stored);
    if (u.role !== 'STATION_ADMIN') {
      if (u.role === 'INVESTIGATING_OFFICER') navigate('/police/psi-dashboard');
      else if (u.role === 'DEPUTY_SUPRINTENDENT') navigate('/police/dsp-dashboard');
      return;
    }
    setUser(u);
    const stationId = u.stationId || u.station?.id;
    if (stationId) {
      loadAll(stationId);
    } else {
      setError('No station assigned');
      setLoading(false);
    }
  }, [navigate]);

  const loadAll = async (stationId) => {
    setLoading(true);
    try {
      const [compRes, firRes, subRes, csRes] = await Promise.all([
        api.get(`/complaints/station/${stationId}`),
        api.get(`/fir/station/${stationId}`),
        api.get(`/police/station/${stationId}`),
        getChargeSheetsByStation(stationId).catch(() => [])
      ]);
      setComplaints(Array.isArray(compRes.data) ? compRes.data : []);
      setFirs(Array.isArray(firRes.data) ? firRes.data : []);
      setSubordinates(
        (Array.isArray(subRes.data) ? subRes.data : [])
          .filter(p => p.role === 'INVESTIGATING_OFFICER')
      );
      setChargeSheets(Array.isArray(csRes) ? csRes : []);
    } catch (e) {
      console.error(e);
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const refresh = useCallback(() => {
    const stationId = user?.stationId || user?.station?.id;
    if (stationId) loadAll(stationId);
  }, [user]);

  /* ── Fetch PE report for a complaint ── */
  const fetchPeReport = async (complaintId) => {
    setPeLoading(true);
    setPeReport(null);
    try {
      const res = await api.get(`/preliminary-report/complaint/${complaintId}`);
      setPeReport(res.data);
      return res.data;
    } catch (e) {
      console.error('Failed to fetch PE report:', e);
      setPeReport(null);
      return null;
    } finally {
      setPeLoading(false);
    }
  };

  /* ── When a complaint is selected, fetch PE if applicable ── */
  const handleSelectComplaint = async (complaint, tab = 'details') => {
    setSelected(complaint);
    setModalTab(tab);
    setShowFirView(false);
    setPeReport(null);
    // Fetch PE report if complaint has PE status
    if (['PE_SUBMITTED', 'PE_ASSIGNED', 'FIR_REGISTERED'].includes(complaint.status)) {
      fetchPeReport(complaint.id);
    }
  };

  /* ── Actions ── */
  const updateComplaint = async (id, payload) => {
    try {
      const res = await api.put(`/complaints/${id}/status`, payload);
      setComplaints(prev => prev.map(c => c.id === id ? res.data : c));
      if (selected?.id === id) setSelected(res.data);
      return res.data;
    } catch (e) {
      alert('Update failed: ' + (e.response?.data?.message || e.message));
    }
  };

  const handleApproveCategory = async (complaint) => {
    await updateComplaint(complaint.id, {
      status: complaint.status,
      actualCategory: complaint.predictedCategory
    });
  };

  const handleOverrideCategory = async (complaint) => {
    if (!categoryOverride) { alert('Select a category'); return; }
    await updateComplaint(complaint.id, {
      status: complaint.status,
      actualCategory: CATEGORY_LABELS[categoryOverride] || categoryOverride
    });
    setCategoryOverride('');
  };

  const handleRequestPE = async (complaint) => {
    if (!window.confirm('Send this complaint to DSP for PE approval?')) return;
    const result = await updateComplaint(complaint.id, { status: 'PE_PENDING_DSP_APPROVAL' });
    if (result) {
      alert('PE request sent to DSP successfully!');
      setSelected(null);
    }
  };

  const handleAssignPSI = async (complaint) => {
    if (!assignOfficerId) { alert('Select a PSI officer'); return; }
    await updateComplaint(complaint.id, {
      status: 'PE_ASSIGNED',
      officerId: parseInt(assignOfficerId)
    });
    setAssignOfficerId('');
  };

  const handleClose = async (complaint) => {
    if (!window.confirm('Close this complaint as No Crime Found?')) return;
    const result = await updateComplaint(complaint.id, { status: 'CLOSED_NO_CRIME' });
    if (result) {
      alert('Complaint closed successfully!');
      setSelected(null);
      setShowFirView(false);
    }
  };

  /* ── Open FIR form (Direct — from RECEIVED complaint) ── */
  const openDirectFirForm = (complaint) => {
    const stationId = user?.stationId || user?.station?.id;
    setFirMode('direct');
    setFirForm({
      ...emptyFirForm,
      firNumber: generateFirNumber(user?.stationCode),
      district: user?.district || '',
      informantName: complaint.complainantName || '',
      informantGuardianName: '',
      informantAddress: complaint.complainantAddress || '',
      informantContact: complaint.complainantMobile || '',
      informantEmail: complaint.complainantEmail || '',
      incidentDescription: complaint.description || '',
      crimeCategory: complaint.actualCategory || complaint.predictedCategory || '',
      firWrittenBy: user?.name || '',
      isEfir: true,
    });
    setShowFirView(true);
  };

  /* ── Open FIR form (From PE report) ── */
  const openPeFirForm = (complaint, report) => {
    if (!report || !report.reportId) {
      alert('PE Report is not available yet. Please wait for it to load or refresh the page.');
      return;
    }
    // Truncate investigation narrative to 2000 chars for FIR incident description
    const narrative = report?.investigationNarrative || complaint.description || '';
    const truncatedNarrative = narrative.length > 2000 ? narrative.substring(0, 2000) : narrative;
    
    setFirMode('from_pe');
    setFirForm({
      ...emptyFirForm,
      firNumber: generateFirNumber(user?.stationCode),
      district: user?.district || '',
      informantName: report?.informantName || complaint.complainantName || '',
      informantGuardianName: '',
      informantAddress: report?.informantAddress || complaint.complainantAddress || '',
      informantContact: report?.informantContact || complaint.complainantMobile || '',
      informantEmail: report?.informantEmail || complaint.complainantEmail || '',
      incidentLocation: report?.incidentLocation || '',
      incidentDate: report?.incidentDate || '',
      incidentTime: report?.incidentTime || '',
      incidentDescription: truncatedNarrative,
      crimeCategory: report?.crimeCategory || complaint.actualCategory || complaint.predictedCategory || '',
      ipcSections: report?.ipcSections || '',
      stolenPropertyDetails: report?.stolenPropertyDetails || '',
      accusedDetails: report?.draftAccusedDetails || '',
      witnessDetails: report?.draftWitnessDetails || '',
      firWrittenBy: user?.name || '',
      investigatingOfficerId: report?.investigatingOfficerId || '',
      isEfir: true,
    });
    setShowFirView(true);
  };

  /* ── Submit FIR ── */
  const handleSubmitFir = async () => {
    // Validation
    if (!firForm.firNumber || !firForm.informantName || !firForm.incidentDescription || !firForm.crimeCategory) {
      alert('Please fill in required fields: FIR Number, Informant Name, Incident Description, Crime Category');
      return;
    }
    if (!firForm.informantContact || !/^[6-9]\d{9}$/.test(firForm.informantContact)) {
      alert('Please enter a valid 10-digit Indian mobile number for informant contact');
      return;
    }
    if (!firForm.incidentDate) {
      alert('Please enter the incident date');
      return;
    }
    if (!firForm.firWrittenBy) {
      alert('Please enter the name of the officer writing this FIR');
      return;
    }
    if (!firForm.district) {
      alert('Please enter the district');
      return;
    }

    // Truncate incident description if too long
    const truncatedDescription = firForm.incidentDescription.length > 2000 
      ? firForm.incidentDescription.substring(0, 2000) 
      : firForm.incidentDescription;

    setFirSubmitting(true);
    const stationId = user?.stationId || user?.station?.id;

    try {
      let firRes;

      if (firMode === 'from_pe' && peReport?.reportId) {
        // Register FIR from PE report
        const payload = {
          reportId: peReport.reportId,
          firNumber: firForm.firNumber,
          district: firForm.district,
          incidentDescription: truncatedDescription,
          status: 'REGISTERED',
          firWrittenBy: firForm.firWrittenBy,
          informantSignaturePath: '',
          isEfir: firForm.isEfir || false,
          isZeroFir: firForm.isZeroFir || false,
          isVictimWoman: firForm.isVictimWoman || false,
          isDisabledVictim: firForm.isDisabledVictim || false,
          recordedByWomanOfficer: firForm.recordedByWomanOfficer || false,
          isMagistrateStatementRecorded: firForm.isMagistrateStatementRecorded || false,
        };
        
        console.log('Submitting FIR from PE report. Payload:', payload);
        console.log('PE Report ID:', peReport.reportId);
        console.log('Full PE Report:', peReport);
        
        firRes = await api.post('/fir/register-from-report', payload);
      } else {
        // Register direct FIR
        console.log('Submitting direct FIR');
        firRes = await api.post('/fir/register', {
          firNumber: firForm.firNumber,
          district: firForm.district,
          informantName: firForm.informantName,
          informantGuardianName: firForm.informantGuardianName || 'N/A',
          informantAddress: firForm.informantAddress || 'N/A',
          informantContact: firForm.informantContact,
          informantEmail: firForm.informantEmail || 'na@example.com',
          incidentLocation: firForm.incidentLocation || 'To be determined',
          incidentDate: firForm.incidentDate,
          incidentTime: firForm.incidentTime || '00:00',
          incidentDescription: truncatedDescription,
          crimeCategory: firForm.crimeCategory,
          ipcSections: firForm.ipcSections || '',
          stolenPropertyDetails: firForm.stolenPropertyDetails || '',
          accusedDetails: firForm.accusedDetails || '',
          witnessDetails: firForm.witnessDetails || '',
          status: 'REGISTERED',
          policeStationId: stationId,
          investigatingOfficerId: firForm.investigatingOfficerId ? parseInt(firForm.investigatingOfficerId) : null,
          complaintId: selected?.id,
          firWrittenBy: firForm.firWrittenBy,
          informantSignaturePath: '',
          isEfir: firForm.isEfir || false,
          isZeroFir: firForm.isZeroFir || false,
          isVictimWoman: firForm.isVictimWoman || false,
          isDisabledVictim: firForm.isDisabledVictim || false,
          recordedByWomanOfficer: firForm.recordedByWomanOfficer || false,
          isMagistrateStatementRecorded: firForm.isMagistrateStatementRecorded || false,
        });
      }

      // Update complaint status to FIR_REGISTERED
      await updateComplaint(selected.id, {
        status: 'FIR_REGISTERED',
        actualCategory: firForm.crimeCategory || selected.actualCategory || selected.predictedCategory
      });

      alert(`FIR ${firForm.firNumber} registered successfully!`);
      setShowFirView(false);
      setSelected(null);
      refresh();
    } catch (e) {
      console.error('FIR registration failed:', e);
      console.error('Error response data:', e.response?.data);
      console.error('Error response data (stringified):', JSON.stringify(e.response?.data, null, 2));
      console.error('Error response status:', e.response?.status);
      console.error('Error response headers:', e.response?.headers);
      
      let errorMsg = 'Unknown error';
      if (e.response?.data) {
        if (typeof e.response.data === 'string') {
          errorMsg = e.response.data;
        } else if (e.response.data.message) {
          errorMsg = e.response.data.message;
        } else if (e.response.data.errors) {
          // Validation errors
          errorMsg = Object.entries(e.response.data.errors).map(([field, msg]) => `${field}: ${msg}`).join('\n');
        } else {
          errorMsg = JSON.stringify(e.response.data);
        }
      } else {
        errorMsg = e.message;
      }
      
      alert('FIR registration failed:\n\n' + errorMsg);
    } finally {
      setFirSubmitting(false);
    }
  };

  const handleFirFieldChange = (field, value) => {
    setFirForm(prev => ({ ...prev, [field]: value }));
  };

  /* ── Charge Sheet Review Actions ── */
  const CS_STATUS_LABELS = {
    DRAFT: 'Draft',
    SUBMITTED_TO_PI: 'Pending Your Review',
    RETURNED_FOR_REVISION: 'Returned to IO',
    APPROVED_BY_PI: 'Approved',
    DISPATCHED_TO_COURT: 'Dispatched to Court',
  };

  const safeParseJson = (jsonStr, fallback = []) => {
    if (!jsonStr) return fallback;
    try { return JSON.parse(jsonStr); } catch { return fallback; }
  };

  const openCsReview = (cs) => {
    setSelectedCs(cs);
    setReturnSuggestions('');
    setCsReviewOpen(true);
  };

  const handleApproveCs = async () => {
    if (!selectedCs) return;
    setCsActionLoading(true);
    try {
      await approveChargeSheet(selectedCs.chargeSheetId, user.policeId);
      alert('Charge sheet approved successfully!');
      setCsReviewOpen(false);
      setSelectedCs(null);
      refresh();
    } catch (e) {
      alert('Failed to approve: ' + (e.response?.data?.message || e.message));
    } finally {
      setCsActionLoading(false);
    }
  };

  const handleReturnCs = async () => {
    if (!selectedCs) return;
    if (!returnSuggestions.trim()) { alert('Please enter suggestions for the IO.'); return; }
    setCsActionLoading(true);
    try {
      await returnChargeSheet(selectedCs.chargeSheetId, user.policeId, returnSuggestions);
      alert('Charge sheet returned to IO with suggestions.');
      setCsReviewOpen(false);
      setSelectedCs(null);
      setReturnSuggestions('');
      refresh();
    } catch (e) {
      alert('Failed to return: ' + (e.response?.data?.message || e.message));
    } finally {
      setCsActionLoading(false);
    }
  };

  const handleDispatchCs = async (cs) => {
    if (!window.confirm('Dispatch this charge sheet to court? This action cannot be undone.')) return;
    setCsActionLoading(true);
    try {
      await dispatchChargeSheet(cs.chargeSheetId);
      alert('Charge sheet dispatched to court!');
      refresh();
    } catch (e) {
      alert('Failed: ' + (e.response?.data?.message || e.message));
    } finally {
      setCsActionLoading(false);
    }
  };

  const pendingChargeSheets = chargeSheets.filter(cs => cs.status === 'SUBMITTED_TO_PI');

  /* ── Filtering ── */
  const filtered = complaints.filter(c =>
    filterStatus === 'ALL' || c.status === filterStatus
  );

  const counts = {
    total: complaints.length,
    received: complaints.filter(c => c.status === 'RECEIVED').length,
    pePending: complaints.filter(c => c.status === 'PE_PENDING_DSP_APPROVAL').length,
    peAssigned: complaints.filter(c => c.status === 'PE_ASSIGNED').length,
    peSubmitted: complaints.filter(c => c.status === 'PE_SUBMITTED').length,
    firDone: complaints.filter(c => c.status === 'FIR_REGISTERED').length,
    closed: complaints.filter(c => c.status === 'CLOSED_NO_CRIME').length,
    firs: firs.length,
    officers: subordinates.length
  };

  const formatDate = (d) => {
    if (!d) return '—';
    return new Date(d).toLocaleString('en-IN', {
      day: '2-digit', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit'
    });
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    localStorage.removeItem('policeUser');
    navigate('/police/login');
  };

  /* ── PE Report Panel (used in split-view left side) ── */
  const renderPePanel = () => {
    if (peLoading) {
      return (
        <div className="fir-split-panel fir-split-left">
          <div className="fir-panel-header"><h3>📋 Preliminary Enquiry Report</h3></div>
          <div className="fir-panel-body"><div className="dash-loading" style={{ minHeight: 200 }}><div className="dash-spinner" /><p>Loading PE report…</p></div></div>
        </div>
      );
    }
    if (!peReport) {
      return (
        <div className="fir-split-panel fir-split-left">
          <div className="fir-panel-header"><h3>📋 Complaint Details</h3></div>
          <div className="fir-panel-body">
            <div className="fir-pe-section">
              <h4>Complainant Information</h4>
              <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{selected?.complainantName || '—'}</span></div>
              <div className="fir-pe-field"><span className="fir-pe-label">Mobile</span><span>{selected?.complainantMobile || '—'}</span></div>
              <div className="fir-pe-field"><span className="fir-pe-label">Email</span><span>{selected?.complainantEmail || '—'}</span></div>
              <div className="fir-pe-field"><span className="fir-pe-label">Address</span><span>{selected?.complainantAddress || '—'}</span></div>
            </div>
            <div className="fir-pe-section">
              <h4>Complaint</h4>
              <div className="fir-pe-field"><span className="fir-pe-label">Category</span><span className="dash-chip cat">{selected?.actualCategory || selected?.predictedCategory || '—'}</span></div>
              <div className="fir-pe-field"><span className="fir-pe-label">Filed On</span><span>{formatDate(selected?.createdAt)}</span></div>
              <div className="fir-pe-narrative">{selected?.description}</div>
            </div>
          </div>
        </div>
      );
    }
    return (
      <div className="fir-split-panel fir-split-left">
        <div className="fir-panel-header">
          <h3>📋 PE Report #{peReport.reportId}</h3>
          <span className={`dash-chip ${peReport.cognizableOffence ? 'danger' : 'neutral'}`}>
            {peReport.cognizableOffence ? 'Cognizable' : 'Non-Cognizable'}
          </span>
        </div>
        <div className="fir-panel-body">
          {/* Investigating Officer */}
          <div className="fir-pe-section">
            <h4>👮 Investigating Officer</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{peReport.investigatingOfficerName || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Badge</span><span>{peReport.investigatingOfficerBadgeNumber || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Rank</span><span>{peReport.investigatingOfficerRank || '—'}</span></div>
          </div>

          {/* Informant from PE */}
          <div className="fir-pe-section">
            <h4>👤 Informant Details</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{peReport.informantName || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Contact</span><span>{peReport.informantContact || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Email</span><span>{peReport.informantEmail || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Address</span><span>{peReport.informantAddress || '—'}</span></div>
          </div>

          {/* Incident from PE */}
          <div className="fir-pe-section">
            <h4>📍 Incident Details</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Location</span><span>{peReport.incidentLocation || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Date</span><span>{peReport.incidentDate || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Time</span><span>{peReport.incidentTime || '—'}</span></div>
          </div>

          {/* Crime from PE */}
          <div className="fir-pe-section">
            <h4>⚖️ Crime Classification</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Category</span><span className="dash-chip cat">{peReport.crimeCategory || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">IPC/BNS Sections</span><span>{peReport.ipcSections || '—'}</span></div>
            {peReport.stolenPropertyDetails && (
              <div className="fir-pe-field"><span className="fir-pe-label">Stolen Property</span><span>{peReport.stolenPropertyDetails}</span></div>
            )}
          </div>

          {/* Accused & Witnesses */}
          {(peReport.draftAccusedDetails || peReport.draftWitnessDetails) && (
            <div className="fir-pe-section">
              <h4>🔍 Persons of Interest</h4>
              {peReport.draftAccusedDetails && (
                <div className="fir-pe-field col"><span className="fir-pe-label">Accused Details</span><span>{peReport.draftAccusedDetails}</span></div>
              )}
              {peReport.draftWitnessDetails && (
                <div className="fir-pe-field col"><span className="fir-pe-label">Witness Details</span><span>{peReport.draftWitnessDetails}</span></div>
              )}
              {peReport.witnessStatement && (
                <div className="fir-pe-field col"><span className="fir-pe-label">Witness Statement</span><span>{peReport.witnessStatement}</span></div>
              )}
            </div>
          )}

          {/* PE Protocol */}
          <div className="fir-pe-section">
            <h4>📑 PE Protocol</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">PE Category</span><span>{peReport.peCategory || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Start Date</span><span>{peReport.peStartDate || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Deadline</span><span>{peReport.peDeadline || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Submitted</span><span>{formatDate(peReport.submittedAt)}</span></div>
          </div>

          {/* Narrative */}
          <div className="fir-pe-section">
            <h4>📝 Investigation Narrative</h4>
            <div className="fir-pe-narrative">{peReport.investigationNarrative || 'No narrative provided.'}</div>
          </div>
        </div>
      </div>
    );
  };

  /* ── FIR Form Panel (right side) ── */
  const renderFirForm = () => (
    <div className="fir-split-panel fir-split-right">
      <div className="fir-panel-header">
        <h3>📄 Register FIR</h3>
        <span className="dash-chip" style={{ background: '#dcfce7', color: '#166534' }}>
          {firMode === 'from_pe' ? 'From PE Report' : 'Direct FIR'}
        </span>
      </div>
      <div className="fir-panel-body">
        <div className="dash-pe-form">
          {/* FIR Basic Info */}
          <h4>FIR Information</h4>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>FIR Number *</label>
              <input type="text" value={firForm.firNumber} onChange={e => handleFirFieldChange('firNumber', e.target.value)} />
            </div>
            <div className="dash-form-group">
              <label>District *</label>
              <input type="text" value={firForm.district} onChange={e => handleFirFieldChange('district', e.target.value)} placeholder="e.g. Pune" />
            </div>
          </div>

          {/* Informant Details */}
          <h4 style={{ marginTop: 8 }}>👤 Informant Details</h4>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>Informant Name *</label>
              <input type="text" value={firForm.informantName} onChange={e => handleFirFieldChange('informantName', e.target.value)} />
            </div>
            <div className="dash-form-group">
              <label>Father/Guardian Name</label>
              <input type="text" value={firForm.informantGuardianName} onChange={e => handleFirFieldChange('informantGuardianName', e.target.value)} />
            </div>
          </div>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>Contact *</label>
              <input type="text" value={firForm.informantContact} onChange={e => handleFirFieldChange('informantContact', e.target.value)} placeholder="10-digit mobile" maxLength={10} />
            </div>
            <div className="dash-form-group">
              <label>Email</label>
              <input type="email" value={firForm.informantEmail} onChange={e => handleFirFieldChange('informantEmail', e.target.value)} />
            </div>
          </div>
          <div className="dash-form-group">
            <label>Address</label>
            <textarea rows={2} value={firForm.informantAddress} onChange={e => handleFirFieldChange('informantAddress', e.target.value)} />
          </div>

          {/* Incident Details */}
          <h4 style={{ marginTop: 8 }}>📍 Incident Details</h4>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>Incident Date *</label>
              <input type="date" value={firForm.incidentDate} onChange={e => handleFirFieldChange('incidentDate', e.target.value)} />
            </div>
            <div className="dash-form-group">
              <label>Incident Time</label>
              <input type="time" value={firForm.incidentTime} onChange={e => handleFirFieldChange('incidentTime', e.target.value)} />
            </div>
          </div>
          <div className="dash-form-group">
            <label>Incident Location</label>
            <input type="text" value={firForm.incidentLocation} onChange={e => handleFirFieldChange('incidentLocation', e.target.value)} placeholder="Where the incident occurred" />
          </div>
          <div className="dash-form-group">
            <label>Incident Description * (max 2000 chars)</label>
            <textarea rows={4} value={firForm.incidentDescription} onChange={e => handleFirFieldChange('incidentDescription', e.target.value)} placeholder="Detailed description of the incident" />
            <div style={{ fontSize: 11, color: firForm.incidentDescription.length > 2000 ? '#dc2626' : '#64748b', textAlign: 'right', marginTop: 2 }}>
              {firForm.incidentDescription.length} / 2000 chars {firForm.incidentDescription.length > 2000 && '(will be truncated)'}
            </div>
          </div>

          {/* Crime Details */}
          <h4 style={{ marginTop: 8 }}>⚖️ Crime Classification</h4>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>Crime Category *</label>
              <input type="text" value={firForm.crimeCategory} onChange={e => handleFirFieldChange('crimeCategory', e.target.value)} />
            </div>
            <div className="dash-form-group">
              <label>IPC / BNS Sections</label>
              <input type="text" value={firForm.ipcSections} onChange={e => handleFirFieldChange('ipcSections', e.target.value)} placeholder="e.g. BNS 303, 305" />
            </div>
          </div>
          <div className="dash-form-group">
            <label>Stolen Property Details</label>
            <textarea rows={2} value={firForm.stolenPropertyDetails} onChange={e => handleFirFieldChange('stolenPropertyDetails', e.target.value)} />
          </div>
          <div className="dash-form-group">
            <label>Accused Details</label>
            <textarea rows={2} value={firForm.accusedDetails} onChange={e => handleFirFieldChange('accusedDetails', e.target.value)} placeholder="Known accused persons and descriptions" />
          </div>
          <div className="dash-form-group">
            <label>Witness Details</label>
            <textarea rows={2} value={firForm.witnessDetails} onChange={e => handleFirFieldChange('witnessDetails', e.target.value)} placeholder="Witness names and statements" />
          </div>

          {/* Officer Assignment */}
          <h4 style={{ marginTop: 8 }}>👮 FIR Registration</h4>
          <div className="dash-form-row">
            <div className="dash-form-group">
              <label>FIR Written By *</label>
              <input type="text" value={firForm.firWrittenBy} onChange={e => handleFirFieldChange('firWrittenBy', e.target.value)} />
            </div>
            <div className="dash-form-group">
              <label>Investigating Officer (PSI)</label>
              <select value={firForm.investigatingOfficerId} onChange={e => handleFirFieldChange('investigatingOfficerId', e.target.value)}>
                <option value="">— Select PSI —</option>
                {subordinates.map(o => (
                  <option key={o.policeId} value={o.policeId}>{o.name} – {o.badgeNumber}</option>
                ))}
              </select>
            </div>
          </div>

          {/* BNSS Compliance */}
          <h4 style={{ marginTop: 8 }}>📋 BNSS 2023 Compliance</h4>
          <div className="fir-checkboxes">
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.isEfir} onChange={e => handleFirFieldChange('isEfir', e.target.checked)} />
              <span>E-FIR</span>
            </label>
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.isZeroFir} onChange={e => handleFirFieldChange('isZeroFir', e.target.checked)} />
              <span>Zero FIR</span>
            </label>
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.isVictimWoman} onChange={e => handleFirFieldChange('isVictimWoman', e.target.checked)} />
              <span>Victim is Woman</span>
            </label>
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.recordedByWomanOfficer} onChange={e => handleFirFieldChange('recordedByWomanOfficer', e.target.checked)} />
              <span>Recorded by Woman Officer</span>
            </label>
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.isDisabledVictim} onChange={e => handleFirFieldChange('isDisabledVictim', e.target.checked)} />
              <span>Disabled Victim</span>
            </label>
            <label className="fir-checkbox-label">
              <input type="checkbox" checked={firForm.isMagistrateStatementRecorded} onChange={e => handleFirFieldChange('isMagistrateStatementRecorded', e.target.checked)} />
              <span>Magistrate Statement Recorded</span>
            </label>
          </div>

          {/* Submit */}
          <div className="fir-submit-row">
            <button className="dash-btn secondary" onClick={() => setShowFirView(false)} disabled={firSubmitting}>
              ← Back
            </button>
            <button className="dash-btn success" onClick={handleSubmitFir} disabled={firSubmitting} style={{ minWidth: 180 }}>
              {firSubmitting ? '⏳ Registering FIR…' : '✔️ Register FIR'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );

  /* ── Render ── */
  if (loading) {
    return (
      <div className="dash-page">
        <div className="dash-loading"><div className="dash-spinner" /><p>Loading PI Dashboard…</p></div>
      </div>
    );
  }

  return (
    <div className="dash-page pi-theme">
      {/* ── Navbar ── */}
      <nav className="dash-nav">
        <div className="dash-nav-left">
          <span className="dash-nav-icon">⭐</span>
          <div>
            <h1 className="dash-nav-title">PI Dashboard</h1>
            <span className="dash-nav-sub">Police Inspector – Station Admin</span>
          </div>
        </div>
        <div className="dash-nav-right">
          <span className="dash-nav-user">👮 {user?.name}</span>
          <span className="dash-nav-badge">Badge: {user?.badgeNumber}</span>
          <span className="dash-role-chip pi">PI</span>
          <button onClick={handleLogout} className="dash-btn-logout">Logout</button>
        </div>
      </nav>

      <div className="dash-body">
        {error && <div className="dash-alert error">{error}</div>}

        {/* ── Station Card ── */}
        <div className="dash-station-card">
          <h3>🏛️ {user?.stationName || 'Station'}</h3>
          <p>Code: {user?.stationCode} &bull; Role: Police Inspector (PI)</p>
        </div>

        {/* ── Stats ── */}
        <div className="dash-stats">
          <div className="dash-stat" data-accent="blue">
            <span className="dash-stat-icon">📋</span>
            <div><h3>{counts.total}</h3><p>Total Complaints</p></div>
          </div>
          <div className="dash-stat" data-accent="orange">
            <span className="dash-stat-icon">🆕</span>
            <div><h3>{counts.received}</h3><p>Fresh (Received)</p></div>
          </div>
          <div className="dash-stat" data-accent="yellow">
            <span className="dash-stat-icon">⏳</span>
            <div><h3>{counts.pePending}</h3><p>PE – Awaiting DSP</p></div>
          </div>
          <div className="dash-stat" data-accent="purple">
            <span className="dash-stat-icon">🔍</span>
            <div><h3>{counts.peAssigned + counts.peSubmitted}</h3><p>PE In Progress</p></div>
          </div>
          <div className="dash-stat" data-accent="green">
            <span className="dash-stat-icon">📄</span>
            <div><h3>{counts.firDone}</h3><p>FIR Registered</p></div>
          </div>
          <div className="dash-stat" data-accent="teal">
            <span className="dash-stat-icon">👥</span>
            <div><h3>{counts.officers}</h3><p>PSI Officers</p></div>
          </div>
        </div>

        {/* ── Tabs ── */}
        <div className="dash-tabs">
          <button className={`dash-tab ${activeTab === 'complaints' ? 'active' : ''}`} onClick={() => setActiveTab('complaints')}>
            📁 Complaints ({counts.total})
          </button>
          <button className={`dash-tab ${activeTab === 'officers' ? 'active' : ''}`} onClick={() => setActiveTab('officers')}>
            👥 PSI Officers ({counts.officers})
          </button>
          <button className={`dash-tab ${activeTab === 'firs' ? 'active' : ''}`} onClick={() => setActiveTab('firs')}>
            📄 FIRs ({counts.firs})
          </button>
          <button className={`dash-tab ${activeTab === 'chargeSheets' ? 'active' : ''}`} onClick={() => setActiveTab('chargeSheets')}
            style={pendingChargeSheets.length > 0 ? { background: '#fff3cd', borderColor: '#ffc107' } : {}}>
            📑 Charge Sheets ({chargeSheets.length}){pendingChargeSheets.length > 0 && ` ⚠️ ${pendingChargeSheets.length} pending`}
          </button>
        </div>

        {/* ── Complaints Tab ── */}
        {activeTab === 'complaints' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>Complaints</h2>
              <div className="dash-controls">
                <select value={filterStatus} onChange={e => setFilterStatus(e.target.value)} className="dash-select">
                  <option value="ALL">All Status</option>
                  <option value="RECEIVED">Received</option>
                  <option value="PE_PENDING_DSP_APPROVAL">PE – Awaiting DSP</option>
                  <option value="PE_ASSIGNED">PE – Assigned</option>
                  <option value="PE_SUBMITTED">PE – Submitted</option>
                  <option value="FIR_REGISTERED">FIR Registered</option>
                  <option value="CLOSED_NO_CRIME">Closed</option>
                </select>
                <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
              </div>
            </div>

            {filtered.length === 0 ? (
              <div className="dash-empty">📭 No complaints match the filter.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>ID</th>
                      <th>Predicted Category</th>
                      <th>Route Advice</th>
                      <th>Actual Category</th>
                      <th>Description</th>
                      <th>Status</th>
                      <th>Assigned To</th>
                      <th>Filed</th>
                      <th>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filtered.map(c => {
                      const advice = getRouteAdvice(c.actualCategory || c.predictedCategory);
                      return (
                        <tr key={c.id}>
                          <td><span className="dash-id">#{c.id}</span></td>
                          <td><span className="dash-chip cat">{c.predictedCategory || '—'}</span></td>
                          <td><span className="dash-chip" style={{ background: advice.color + '22', color: advice.color, border: `1px solid ${advice.color}` }}>{advice.label}</span></td>
                          <td>{c.actualCategory ? <span className="dash-chip confirmed">✓ {c.actualCategory}</span> : <span style={{ color: '#bbb' }}>—</span>}</td>
                          <td className="dash-desc">{c.description?.substring(0, 55)}{c.description?.length > 55 ? '…' : ''}</td>
                          <td><span className={`dash-status ${c.status?.toLowerCase().replace(/_/g, '-')}`}>{STATUS_LABELS[c.status] || c.status}</span></td>
                          <td>{c.assignedOfficerName ? <span>👤 {c.assignedOfficerName}</span> : <span style={{ color: '#bbb' }}>—</span>}</td>
                          <td className="dash-date">{formatDate(c.createdAt)}</td>
                          <td>
                            <button className="dash-btn primary sm" onClick={() => handleSelectComplaint(c, 'details')}>View</button>
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

        {/* ── Officers Tab ── */}
        {activeTab === 'officers' && (
          <div className="dash-card">
            <div className="dash-card-header"><h2>👥 Subordinate PSI Officers</h2></div>
            {subordinates.length === 0 ? (
              <div className="dash-empty">No PSI officers found at this station.</div>
            ) : (
              <div className="dash-officers-grid">
                {subordinates.map(o => (
                  <div key={o.policeId} className="dash-officer-card">
                    <div className="dash-officer-avatar">👮</div>
                    <div>
                      <h4>{o.name}</h4>
                      <p>Badge: {o.badgeNumber}</p>
                      <p>Rank: {o.rank}</p>
                      <p>Email: {o.email}</p>
                      <p className="dash-officer-cases">
                        Assigned: {complaints.filter(c => c.assignedOfficerId === o.policeId).length} cases
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── FIRs Tab ── */}
        {activeTab === 'firs' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>📄 FIRs at This Station</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {firs.length === 0 ? (
              <div className="dash-empty">No FIRs registered yet.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead>
                    <tr>
                      <th>FIR #</th>
                      <th>Crime Category</th>
                      <th>Informant</th>
                      <th>District</th>
                      <th>Status</th>
                      <th>Investigating Officer</th>
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

        {/* ── Charge Sheets Tab ── */}
        {activeTab === 'chargeSheets' && (
          <div className="dash-card">
            <div className="dash-card-header">
              <h2>📑 Charge Sheets / Final Reports</h2>
              <button onClick={refresh} className="dash-btn secondary">🔄 Refresh</button>
            </div>
            {chargeSheets.length === 0 ? (
              <div className="dash-empty">No charge sheets filed yet.</div>
            ) : (
              <div className="dash-table-wrap">
                <table className="dash-table">
                  <thead><tr><th>CS #</th><th>FIR #</th><th>IO</th><th>Type</th><th>Status</th><th>Submitted</th><th>Action</th></tr></thead>
                  <tbody>
                    {chargeSheets.map(cs => (
                      <tr key={cs.chargeSheetId} style={cs.status === 'SUBMITTED_TO_PI' ? { background: '#fffbeb' } : {}}>
                        <td><span className="dash-id">{cs.chargeSheetNumber}</span></td>
                        <td>{cs.firNumber}</td>
                        <td>{cs.investigatingOfficerName}</td>
                        <td><span className="dash-chip cat">{cs.reportType?.replace(/_/g, ' ')}</span></td>
                        <td><span className={`dash-status ${cs.status?.toLowerCase().replace(/_/g, '-')}`}>{CS_STATUS_LABELS[cs.status] || cs.status}</span></td>
                        <td className="dash-date">{cs.submittedAt ? formatDate(cs.submittedAt) : '-'}</td>
                        <td>
                          {cs.status === 'SUBMITTED_TO_PI' && (
                            <button className="dash-btn primary sm" onClick={() => openCsReview(cs)}>📋 Review</button>
                          )}
                          {cs.status === 'APPROVED_BY_PI' && (
                            <button className="dash-btn success sm" disabled={csActionLoading} onClick={() => handleDispatchCs(cs)}>📤 Dispatch to Court</button>
                          )}
                          {cs.status === 'RETURNED_FOR_REVISION' && (
                            <span style={{ color: '#856404', fontSize: 12 }}>Returned to IO</span>
                          )}
                          {cs.status === 'DISPATCHED_TO_COURT' && (
                            <span style={{ color: '#004085', fontSize: 12 }}>✅ In Court</span>
                          )}
                          {cs.status === 'DRAFT' && (
                            <span style={{ color: '#6c757d', fontSize: 12 }}>Draft (IO editing)</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ══════════════════════════════════════════════════════
          Charge Sheet Review Modal
         ══════════════════════════════════════════════════════ */}
      {csReviewOpen && selectedCs && (
        <div className="dash-overlay" onClick={() => { setCsReviewOpen(false); setSelectedCs(null); }}>
          <div className="dash-modal lg" style={{ maxWidth: 850, maxHeight: '92vh', overflow: 'auto' }} onClick={e => e.stopPropagation()}>
            <div className="dash-modal-header">
              <h2>📋 Review Charge Sheet — {selectedCs.chargeSheetNumber}</h2>
              <button className="dash-modal-close" onClick={() => { setCsReviewOpen(false); setSelectedCs(null); }}>×</button>
            </div>
            <div className="dash-modal-body">
              {/* Header Info */}
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 16 }}>
                <div className="dash-detail"><label>FIR Number</label><span>{selectedCs.firNumber}</span></div>
                <div className="dash-detail"><label>Report Type</label><span className="dash-chip cat">{selectedCs.reportType?.replace(/_/g, ' ')}</span></div>
                <div className="dash-detail"><label>IO</label><span>{selectedCs.investigatingOfficerName} (Badge: {selectedCs.investigatingOfficerBadgeNumber})</span></div>
                <div className="dash-detail"><label>Station</label><span>{selectedCs.policeStationName} ({selectedCs.policeStationCode})</span></div>
                <div className="dash-detail"><label>District</label><span>{selectedCs.district}</span></div>
                <div className="dash-detail"><label>Submitted</label><span>{formatDate(selectedCs.submittedAt)}</span></div>
                {selectedCs.revisionCount > 0 && <div className="dash-detail"><label>Revision #</label><span style={{ color: '#e67e22' }}>{selectedCs.revisionCount}</span></div>}
              </div>

              {/* Acts & Sections */}
              <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: 12, marginTop: 8 }}>
                <h4 style={{ margin: '0 0 8px' }}>Acts & Sections</h4>
                <div style={{ background: '#f8fafc', padding: 10, borderRadius: 6, fontSize: 13 }}>{selectedCs.actsAndSections || '-'}</div>
              </div>

              {/* Brief Facts */}
              <div style={{ marginTop: 12 }}>
                <h4 style={{ margin: '0 0 8px' }}>Brief Facts of the Case</h4>
                <div style={{ background: '#f8fafc', padding: 10, borderRadius: 6, fontSize: 13, whiteSpace: 'pre-wrap', maxHeight: 200, overflowY: 'auto' }}>{selectedCs.briefFacts || '-'}</div>
              </div>

              {/* Accused Charge-sheeted */}
              {(() => {
                const accused = safeParseJson(selectedCs.accusedChargeSheetedJson);
                if (accused.length === 0) return null;
                return (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ margin: '0 0 8px' }}>Accused — Charge-sheeted ({accused.length})</h4>
                    <div className="dash-table-wrap">
                      <table className="dash-table" style={{ fontSize: 12 }}>
                        <thead><tr><th>Name</th><th>Father Name</th><th>DOB</th><th>Occupation</th><th>Arrest Date</th><th>Bail Date</th></tr></thead>
                        <tbody>
                          {accused.map((a, i) => (
                            <tr key={i}><td>{a.name}</td><td>{a.fatherName}</td><td>{a.dob}</td><td>{a.occupation}</td><td>{a.arrestDate || '-'}</td><td>{a.bailDate || '-'}</td></tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()}

              {/* Accused Not Charge-sheeted */}
              {(() => {
                const notCharged = safeParseJson(selectedCs.accusedNotChargeSheetedJson);
                if (notCharged.length === 0) return null;
                return (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ margin: '0 0 8px' }}>Accused — Not Charge-sheeted ({notCharged.length})</h4>
                    {notCharged.map((a, i) => (
                      <div key={i} style={{ background: '#f8fafc', padding: 8, borderRadius: 6, marginTop: 4, fontSize: 13 }}>
                        <strong>{a.name}</strong> (S/o {a.fatherName}) — <em>{a.reasonForNotProsecuting}</em>
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* Accused Absconding */}
              {(() => {
                const absconding = safeParseJson(selectedCs.accusedAbscondingJson);
                if (absconding.length === 0) return null;
                return (
                  <div style={{ marginTop: 12 }}>
                    <h4 style={{ margin: '0 0 8px', color: '#e74c3c' }}>🔴 Accused — Absconding ({absconding.length})</h4>
                    {absconding.map((a, i) => (
                      <div key={i} style={{ background: '#fff5f5', border: '1px solid #fc8181', padding: 8, borderRadius: 6, marginTop: 4, fontSize: 13 }}>
                        <strong>{a.name}</strong> (S/o {a.fatherName}) — Last: {a.lastKnownAddress || '-'} {a.warrantIssued ? '⚠️ Warrant Issued' : ''}
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* Evidence */}
              {(() => {
                const property = safeParseJson(selectedCs.seizedPropertyJson);
                if (property.length === 0 && !selectedCs.chainOfCustody && !selectedCs.laboratoryResult) return null;
                return (
                  <div style={{ marginTop: 12, borderTop: '1px solid #e2e8f0', paddingTop: 12 }}>
                    <h4 style={{ margin: '0 0 8px' }}>Evidence & Recovery</h4>
                    {property.length > 0 && (
                      <div className="dash-table-wrap">
                        <table className="dash-table" style={{ fontSize: 12 }}>
                          <thead><tr><th>Description</th><th>Value (Rs.)</th><th>Muddamal #</th><th>PS Reg No.</th></tr></thead>
                          <tbody>
                            {property.map((p, i) => (
                              <tr key={i}><td>{p.description}</td><td>{p.estimatedValue}</td><td>{p.muddamalNumber}</td><td>{p.psPropertyRegNo}</td></tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                    {selectedCs.chainOfCustody && (
                      <div style={{ marginTop: 8 }}><strong>Chain of Custody:</strong><div style={{ background: '#f8fafc', padding: 8, borderRadius: 6, marginTop: 4, fontSize: 13, whiteSpace: 'pre-wrap' }}>{selectedCs.chainOfCustody}</div></div>
                    )}
                    {selectedCs.laboratoryResult && (
                      <div style={{ marginTop: 8 }}><strong>Lab Result:</strong><div style={{ background: '#f8fafc', padding: 8, borderRadius: 6, marginTop: 4, fontSize: 13, whiteSpace: 'pre-wrap' }}>{selectedCs.laboratoryResult}</div></div>
                    )}
                  </div>
                );
              })()}

              {/* Witnesses */}
              {(() => {
                const witnesses = safeParseJson(selectedCs.witnessListJson);
                if (witnesses.length === 0) return null;
                return (
                  <div style={{ marginTop: 12, borderTop: '1px solid #e2e8f0', paddingTop: 12 }}>
                    <h4 style={{ margin: '0 0 8px' }}>Witnesses ({witnesses.length})</h4>
                    <div className="dash-table-wrap">
                      <table className="dash-table" style={{ fontSize: 12 }}>
                        <thead><tr><th>#</th><th>Name</th><th>Father Name</th><th>Address</th><th>Age</th><th>Evidence Type</th></tr></thead>
                        <tbody>
                          {witnesses.map((w, i) => (
                            <tr key={i}><td>{w.serialNo}</td><td>{w.name}</td><td>{w.fatherName}</td><td>{w.address}</td><td>{w.age}</td><td>{w.evidenceType?.replace(/_/g, ' ')}</td></tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()}

              {/* Notifications & Verification */}
              <div style={{ marginTop: 12, borderTop: '1px solid #e2e8f0', paddingTop: 12 }}>
                <div className="dash-detail"><label>Complainant Notified?</label><span>{selectedCs.complainantNotified ? '✅ Yes' : '❌ No'}</span></div>
              </div>

              {/* PI Action: Approve or Return */}
              {selectedCs.status === 'SUBMITTED_TO_PI' && (
                <div style={{ marginTop: 20, borderTop: '2px solid #3b82f6', paddingTop: 16 }}>
                  <h3 style={{ margin: '0 0 12px', color: '#1e40af' }}>PI Decision</h3>

                  <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
                    <button className="dash-btn success" disabled={csActionLoading} onClick={handleApproveCs} style={{ flex: 1, padding: '12px 0', fontSize: 15 }}>
                      {csActionLoading ? 'Processing...' : '✅ Approve Charge Sheet'}
                    </button>
                  </div>

                  <div style={{ background: '#fff3cd', border: '1px solid #ffc107', borderRadius: 8, padding: 14 }}>
                    <label style={{ fontWeight: 600, marginBottom: 8, display: 'block' }}>Return for Revision — Suggestions to IO:</label>
                    <textarea
                      value={returnSuggestions}
                      onChange={e => setReturnSuggestions(e.target.value)}
                      placeholder="Explain what needs to be corrected or added..."
                      rows={3}
                      style={{ width: '100%', padding: 10, borderRadius: 6, border: '1px solid #ddd' }}
                    />
                    <button className="dash-btn warning" disabled={csActionLoading} onClick={handleReturnCs} style={{ marginTop: 8 }}>
                      {csActionLoading ? 'Processing...' : '↩️ Return to IO for Revision'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div className="dash-modal-footer">
              <button className="dash-btn secondary" onClick={() => { setCsReviewOpen(false); setSelectedCs(null); }}>Close</button>
            </div>
          </div>
        </div>
      )}

      {/* ══════════════════════════════════════════════════════
          FIR Detail Modal
         ══════════════════════════════════════════════════════ */}
      {selectedFir && (
        <FirDetailModal
          fir={selectedFir}
          onClose={() => setSelectedFir(null)}
          formatDate={formatDate}
          subordinates={subordinates}
          onAssignOfficer={async (firId, policeId) => {
            try {
              const res = await api.put(`/fir/${firId}/assign-officer?policeId=${policeId}`);
              setFirs(prev => prev.map(f => f.firId === firId ? { ...f, ...res.data } : f));
              setSelectedFir(res.data);
              alert('Investigating Officer assigned successfully!');
            } catch (e) {
              alert('Failed to assign officer: ' + (e.response?.data?.message || e.message));
            }
          }}
        />
      )}

      {/* ══════════════════════════════════════════════════════
          Complaint Detail / FIR Split-View Modal
         ══════════════════════════════════════════════════════ */}
      {selected && (
        <div className="dash-overlay" onClick={() => { setSelected(null); setShowFirView(false); }}>
          <div className={`dash-modal ${showFirView ? 'fir-split-modal' : 'lg'}`} onClick={e => e.stopPropagation()}>
            {/* ── Split View: PE on left, FIR form on right ── */}
            {showFirView ? (
              <>
                <div className="dash-modal-header">
                  <h2>Register FIR — Complaint #{selected.id}</h2>
                  <button className="dash-modal-close" onClick={() => { setSelected(null); setShowFirView(false); }}>×</button>
                </div>
                <div className="fir-split-container">
                  {renderPePanel()}
                  {renderFirForm()}
                </div>
              </>
            ) : (
              <>
                {/* ── Normal Detail Modal ── */}
                <div className="dash-modal-header">
                  <h2>Complaint #{selected.id}</h2>
                  <button className="dash-modal-close" onClick={() => setSelected(null)}>×</button>
                </div>

                {/* Modal Tabs */}
                <div className="dash-modal-tabs">
                  <button className={`dash-tab ${modalTab === 'details' ? 'active' : ''}`} onClick={() => setModalTab('details')}>Details</button>
                  {['PE_SUBMITTED', 'PE_ASSIGNED', 'FIR_REGISTERED'].includes(selected.status) && (
                    <button className={`dash-tab ${modalTab === 'pe_report' ? 'active' : ''}`} onClick={() => setModalTab('pe_report')}>
                      📋 PE Report
                    </button>
                  )}
                  <button className={`dash-tab ${modalTab === 'classify' ? 'active' : ''}`} onClick={() => setModalTab('classify')}>Classify</button>
                  <button className={`dash-tab ${modalTab === 'action' ? 'active' : ''}`} onClick={() => setModalTab('action')}>Action</button>
                </div>

                <div className="dash-modal-body">
                  {/* Details Tab */}
                  {modalTab === 'details' && (
                    <>
                      <div className="dash-detail">
                        <label>Status</label>
                        <span className={`dash-status ${selected.status?.toLowerCase().replace(/_/g, '-')}`}>
                          {STATUS_LABELS[selected.status] || selected.status}
                        </span>
                      </div>
                      <div className="dash-detail">
                        <label>Predicted Category</label>
                        <span className="dash-chip cat">{selected.predictedCategory || '—'}</span>
                      </div>
                      {selected.actualCategory && (
                        <div className="dash-detail">
                          <label>Actual Category</label>
                          <span className="dash-chip confirmed">✓ {selected.actualCategory}</span>
                        </div>
                      )}
                      <div className="dash-detail">
                        <label>Route Advice</label>
                        {(() => {
                          const adv = getRouteAdvice(selected.actualCategory || selected.predictedCategory);
                          return <span style={{ color: adv.color, fontWeight: 600 }}>{adv.label}</span>;
                        })()}
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
                    </>
                  )}

                  {/* PE Report Tab */}
                  {modalTab === 'pe_report' && (
                    <>
                      {peLoading && (
                        <div className="dash-loading" style={{ minHeight: 200 }}>
                          <div className="dash-spinner" /><p>Loading PE report…</p>
                        </div>
                      )}
                      {!peLoading && !peReport && (
                        <div className="dash-empty">📭 No PE report found for this complaint.</div>
                      )}
                      {!peLoading && peReport && (
                        <>
                          <div className="dash-info-box" style={{ marginBottom: 16 }}>
                            📋 PE Report #{peReport.reportId} — Submitted {formatDate(peReport.submittedAt)}
                            &nbsp;&nbsp;
                            <span className={`dash-chip ${peReport.cognizableOffence ? 'danger' : 'neutral'}`}>
                              {peReport.cognizableOffence ? 'Cognizable Offence' : 'Non-Cognizable'}
                            </span>
                          </div>

                          <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                            <h4>👮 Investigating Officer</h4>
                            <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{peReport.investigatingOfficerName}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Badge</span><span>{peReport.investigatingOfficerBadgeNumber}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Rank</span><span>{peReport.investigatingOfficerRank}</span></div>
                          </div>

                          <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                            <h4>👤 Informant</h4>
                            <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{peReport.informantName || '—'}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Contact</span><span>{peReport.informantContact || '—'}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Email</span><span>{peReport.informantEmail || '—'}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Address</span><span>{peReport.informantAddress || '—'}</span></div>
                          </div>

                          <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                            <h4>📍 Incident</h4>
                            <div className="fir-pe-field"><span className="fir-pe-label">Location</span><span>{peReport.incidentLocation || '—'}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Date</span><span>{peReport.incidentDate || '—'}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Time</span><span>{peReport.incidentTime || '—'}</span></div>
                          </div>

                          <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                            <h4>⚖️ Crime Details</h4>
                            <div className="fir-pe-field"><span className="fir-pe-label">Category</span><span className="dash-chip cat">{peReport.crimeCategory}</span></div>
                            <div className="fir-pe-field"><span className="fir-pe-label">Sections</span><span>{peReport.ipcSections || '—'}</span></div>
                            {peReport.stolenPropertyDetails && <div className="fir-pe-field"><span className="fir-pe-label">Stolen Property</span><span>{peReport.stolenPropertyDetails}</span></div>}
                            {peReport.draftAccusedDetails && <div className="fir-pe-field"><span className="fir-pe-label">Accused</span><span>{peReport.draftAccusedDetails}</span></div>}
                            {peReport.draftWitnessDetails && <div className="fir-pe-field"><span className="fir-pe-label">Witnesses</span><span>{peReport.draftWitnessDetails}</span></div>}
                          </div>

                          <div className="fir-pe-section" style={{ background: '#fffbeb', borderRadius: 8, padding: 16, marginBottom: 12 }}>
                            <h4>📝 Investigation Narrative</h4>
                            <div className="fir-pe-narrative">{peReport.investigationNarrative}</div>
                          </div>
                        </>
                      )}
                    </>
                  )}

                  {/* Classify Tab */}
                  {modalTab === 'classify' && (
                    <>
                      <div className="dash-detail">
                        <label>ML Predicted</label>
                        <span className="dash-chip cat">{selected.predictedCategory || '—'}</span>
                      </div>
                      <div className="dash-detail">
                        <label>Current Actual</label>
                        <span>{selected.actualCategory || 'Not yet classified'}</span>
                      </div>

                      {!selected.actualCategory && selected.predictedCategory && (
                        <div className="dash-action-row">
                          <button className="dash-btn success" onClick={() => handleApproveCategory(selected)}>
                            ✓ Approve Predicted Category
                          </button>
                        </div>
                      )}

                      <div className="dash-action-row">
                        <select value={categoryOverride} onChange={e => setCategoryOverride(e.target.value)} className="dash-select full">
                          <option value="">— Override Category —</option>
                          {ALL_CATEGORIES.map(cat => (
                            <option key={cat} value={cat}>{CATEGORY_LABELS[cat]}</option>
                          ))}
                        </select>
                        <button className="dash-btn primary" onClick={() => handleOverrideCategory(selected)} disabled={!categoryOverride}>
                          Set Category
                        </button>
                      </div>

                      {(selected.actualCategory || selected.predictedCategory) && (
                        <div className="dash-route-box">
                          {(() => {
                            const adv = getRouteAdvice(selected.actualCategory || selected.predictedCategory);
                            return (
                              <>
                                <h4 style={{ color: adv.color }}>Route: {adv.label}</h4>
                                {adv.type === 'direct_fir' && <p>This category requires immediate FIR registration. No PE needed.</p>}
                                {adv.type === 'pe_recommended' && <p>Preliminary Enquiry is recommended. Request DSP approval to proceed.</p>}
                                {adv.type === 'conditional' && <p>Review the specifics: may need Direct FIR (e.g. POCSO) or PE (e.g. BNS 86 / family matters).</p>}
                                {adv.type === 'non_cognizable' && <p>Non-cognizable offence. File an NCR report instead. Can be closed.</p>}
                              </>
                            );
                          })()}
                        </div>
                      )}
                    </>
                  )}

                  {/* Action Tab */}
                  {modalTab === 'action' && (
                    <>
                      <div className="dash-detail">
                        <label>Current Status</label>
                        <span className={`dash-status ${selected.status?.toLowerCase().replace(/_/g, '-')}`}>
                          {STATUS_LABELS[selected.status] || selected.status}
                        </span>
                      </div>

                      {/* RECEIVED → Direct FIR or Request PE */}
                      {selected.status === 'RECEIVED' && (
                        <div className="dash-action-section">
                          <h4>Choose Path</h4>
                          <div className="dash-action-row">
                            <button className="dash-btn danger" onClick={() => openDirectFirForm(selected)}>
                              🚨 Register Direct FIR
                            </button>
                            <button className="dash-btn warning" onClick={() => handleRequestPE(selected)}>
                              📝 Request PE (Send to DSP)
                            </button>
                            <button className="dash-btn secondary" onClick={() => handleClose(selected)}>
                              ❌ Close (No Crime)
                            </button>
                          </div>
                        </div>
                      )}

                      {/* PE_PENDING_DSP_APPROVAL → Waiting */}
                      {selected.status === 'PE_PENDING_DSP_APPROVAL' && (
                        <div className="dash-action-section">
                          <div className="dash-info-box">
                            ⏳ Waiting for DSP approval. The DSP will approve or deny the PE request from their dashboard.
                          </div>
                        </div>
                      )}

                      {/* PE_ASSIGNED — Assign PSI if not yet done, or change */}
                      {selected.status === 'PE_ASSIGNED' && (
                        <div className="dash-action-section">
                          <h4>Assign PSI for PE Investigation</h4>
                          <div className="dash-action-row">
                            <select value={assignOfficerId} onChange={e => setAssignOfficerId(e.target.value)} className="dash-select full">
                              <option value="">— Select PSI —</option>
                              {subordinates.map(o => (
                                <option key={o.policeId} value={o.policeId}>
                                  {o.name} – {o.badgeNumber} ({complaints.filter(c => c.assignedOfficerId === o.policeId).length} cases)
                                </option>
                              ))}
                            </select>
                            <button className="dash-btn primary" onClick={() => handleAssignPSI(selected)} disabled={!assignOfficerId}>
                              Assign PSI
                            </button>
                          </div>
                        </div>
                      )}

                      {/* PE_SUBMITTED — Review PE and register FIR or close */}
                      {selected.status === 'PE_SUBMITTED' && (
                        <div className="dash-action-section">
                          <h4>PE Report Received — Take Decision</h4>
                          {peLoading && (
                            <div className="dash-info-box" style={{ marginBottom: 12 }}>
                              ⏳ Loading PE report...
                            </div>
                          )}
                          {!peLoading && peReport && (
                            <div className="dash-info-box" style={{ marginBottom: 12 }}>
                              📋 PE Report available — <b>{peReport.cognizableOffence ? 'Cognizable Offence Found' : 'Non-Cognizable'}</b>
                              {peReport.investigatingOfficerName && <> by {peReport.investigatingOfficerName}</>}
                            </div>
                          )}
                          {!peLoading && !peReport && (
                            <div className="dash-info-box" style={{ marginBottom: 12, background: '#fef2f2', borderColor: '#fecaca', color: '#991b1b' }}>
                              ⚠️ PE Report not found. Please refresh the page.
                            </div>
                          )}
                          <div className="dash-action-row">
                            <button 
                              className="dash-btn success" 
                              onClick={() => openPeFirForm(selected, peReport)} 
                              style={{ minWidth: 200 }}
                              disabled={peLoading || !peReport}
                            >
                              ✔️ Register FIR (Open Form)
                            </button>
                            <button className="dash-btn secondary" onClick={() => handleClose(selected)}>
                              ❌ Close (No Crime Found)
                            </button>
                          </div>
                          <p className="dash-form-hint" style={{ marginTop: 8 }}>
                            {peReport ? 'Click "Register FIR" to open the split-view with PE report on the left and FIR form on the right.' : 'Waiting for PE report to load...'}
                          </p>
                        </div>
                      )}

                      {/* Terminal states */}
                      {(selected.status === 'FIR_REGISTERED' || selected.status === 'CLOSED_NO_CRIME') && (
                        <div className="dash-info-box">
                          ✅ This complaint has been finalized. Status: <b>{STATUS_LABELS[selected.status]}</b>
                        </div>
                      )}
                    </>
                  )}
                </div>

                <div className="dash-modal-footer">
                  <button className="dash-btn secondary" onClick={() => setSelected(null)}>Close</button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

/* ══════════ FIR Detail Modal (shared) ══════════ */
const FirDetailModal = ({ fir, onClose, formatDate: fmt, subordinates, onAssignOfficer }) => {
  const [assignOfficer, setAssignOfficer] = useState(fir?.investigatingOfficerId?.toString() || '');
  const [assigning, setAssigning] = useState(false);
  if (!fir) return null;
  const formatDateFn = fmt || ((d) => d ? new Date(d).toLocaleString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }) : '—');

  const handleAssign = async () => {
    if (!assignOfficer) { alert('Please select an officer'); return; }
    setAssigning(true);
    try {
      await onAssignOfficer(fir.firId, parseInt(assignOfficer));
    } finally {
      setAssigning(false);
    }
  };
  return (
    <div className="dash-overlay" onClick={onClose}>
      <div className="dash-modal lg" onClick={e => e.stopPropagation()}>
        <div className="dash-modal-header">
          <h2>📄 FIR Details — {fir.firNumber}</h2>
          <button className="dash-modal-close" onClick={onClose}>×</button>
        </div>
        <div className="dash-modal-body">
          {/* Basic FIR Info */}
          <div className="fir-pe-section" style={{ background: '#eff6ff', borderRadius: 8, padding: 16, marginBottom: 12 }}>
            <h4>📋 FIR Information</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">FIR Number</span><span style={{ fontWeight: 600 }}>{fir.firNumber}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">District</span><span>{fir.district || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Status</span><span className={`dash-status ${fir.status?.toLowerCase()}`}>{fir.status}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Registered At</span><span>{formatDateFn(fir.registeredAt)}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Written By</span><span>{fir.firWrittenBy || '—'}</span></div>
            {fir.complaintId && <div className="fir-pe-field"><span className="fir-pe-label">Complaint ID</span><span>#{fir.complaintId}</span></div>}
          </div>

          {/* Informant */}
          <div className="fir-pe-section" style={{ background: '#f0fdf4', borderRadius: 8, padding: 16, marginBottom: 12 }}>
            <h4>👤 Informant Details</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Name</span><span>{fir.informantName || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Guardian</span><span>{fir.informantGuardianName || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Contact</span><span>{fir.informantContact || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Email</span><span>{fir.informantEmail || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Address</span><span>{fir.informantAddress || '—'}</span></div>
          </div>

          {/* Incident */}
          <div className="fir-pe-section" style={{ background: '#fefce8', borderRadius: 8, padding: 16, marginBottom: 12 }}>
            <h4>📍 Incident Details</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Location</span><span>{fir.incidentLocation || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Date</span><span>{fir.incidentDate || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">Time</span><span>{fir.incidentTime || '—'}</span></div>
          </div>

          {/* Crime */}
          <div className="fir-pe-section" style={{ background: '#fef2f2', borderRadius: 8, padding: 16, marginBottom: 12 }}>
            <h4>⚖️ Crime Details</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Category</span><span className="dash-chip cat">{fir.crimeCategory || '—'}</span></div>
            <div className="fir-pe-field"><span className="fir-pe-label">IPC / BNS Sections</span><span>{fir.ipcSections || '—'}</span></div>
            {fir.stolenPropertyDetails && <div className="fir-pe-field"><span className="fir-pe-label">Stolen Property</span><span>{fir.stolenPropertyDetails}</span></div>}
            {fir.accusedDetails && <div className="fir-pe-field"><span className="fir-pe-label">Accused Details</span><span>{fir.accusedDetails}</span></div>}
            {fir.witnessDetails && <div className="fir-pe-field"><span className="fir-pe-label">Witness Details</span><span>{fir.witnessDetails}</span></div>}
          </div>

          {/* Incident Description */}
          {fir.incidentDescription && (
            <div className="fir-pe-section" style={{ background: '#fffbeb', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <h4>📝 Incident Description</h4>
              <div className="fir-pe-narrative">{fir.incidentDescription}</div>
            </div>
          )}

          {/* Station & Officer */}
          <div className="fir-pe-section" style={{ background: '#f8fafc', borderRadius: 8, padding: 16, marginBottom: 12 }}>
            <h4>🏛️ Station & Officer</h4>
            <div className="fir-pe-field"><span className="fir-pe-label">Police Station</span><span>{fir.policeStationName || '—'}</span></div>
            {fir.policeStationCode && <div className="fir-pe-field"><span className="fir-pe-label">Station Code</span><span>{fir.policeStationCode}</span></div>}
            {fir.policeStationAddress && <div className="fir-pe-field"><span className="fir-pe-label">Station Address</span><span>{fir.policeStationAddress}</span></div>}
            <div className="fir-pe-field"><span className="fir-pe-label">Investigating Officer</span><span>{fir.investigatingOfficerName || <em style={{ color: '#ef4444' }}>Not Assigned</em>}</span></div>
            {fir.investigatingOfficerBadgeNumber && <div className="fir-pe-field"><span className="fir-pe-label">Badge #</span><span>{fir.investigatingOfficerBadgeNumber}</span></div>}
            {fir.investigatingOfficerRank && <div className="fir-pe-field"><span className="fir-pe-label">Rank</span><span>{fir.investigatingOfficerRank}</span></div>}

            {/* IO Assignment / Reassignment */}
            {subordinates && subordinates.length > 0 && onAssignOfficer && (
              <div style={{ marginTop: 12, paddingTop: 12, borderTop: '1px solid #e2e8f0' }}>
                <label style={{ fontWeight: 600, fontSize: '0.85rem', color: '#475569', marginBottom: 6, display: 'block' }}>
                  {fir.investigatingOfficerName ? '🔄 Reassign IO' : '👮 Assign Investigating Officer'}
                </label>
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <select
                    value={assignOfficer}
                    onChange={e => setAssignOfficer(e.target.value)}
                    style={{ flex: 1, padding: '8px 12px', borderRadius: 6, border: '1px solid #cbd5e1', fontSize: '0.9rem' }}
                  >
                    <option value="">— Select IO (PSI) —</option>
                    {subordinates.map(o => (
                      <option key={o.policeId} value={o.policeId}>
                        {o.name} – {o.badgeNumber} {o.policeId === fir.investigatingOfficerId ? '(current)' : ''}
                      </option>
                    ))}
                  </select>
                  <button
                    onClick={handleAssign}
                    disabled={assigning || !assignOfficer}
                    className="dash-btn primary sm"
                    style={{ whiteSpace: 'nowrap' }}
                  >
                    {assigning ? '⏳…' : fir.investigatingOfficerName ? '🔄 Reassign' : '✔️ Assign'}
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* BNSS 2023 Compliance */}
          {(fir.isEfir || fir.isZeroFir || fir.isVictimWoman || fir.isDisabledVictim) && (
            <div className="fir-pe-section" style={{ background: '#f5f3ff', borderRadius: 8, padding: 16, marginBottom: 12 }}>
              <h4>📜 BNSS 2023 Compliance</h4>
              {fir.isEfir && <div className="fir-pe-field"><span className="fir-pe-label">e-FIR</span><span>✅ Yes</span></div>}
              {fir.isZeroFir && <div className="fir-pe-field"><span className="fir-pe-label">Zero FIR</span><span>✅ Yes — Dest: {fir.destinationPoliceStation || '—'}</span></div>}
              {fir.isSignatureObtained != null && <div className="fir-pe-field"><span className="fir-pe-label">Signature Obtained</span><span>{fir.isSignatureObtained ? '✅ Yes' : '❌ No'}</span></div>}
              {fir.isVictimWoman && <div className="fir-pe-field"><span className="fir-pe-label">Woman Victim</span><span>✅ Yes</span></div>}
              {fir.recordedByWomanOfficer != null && <div className="fir-pe-field"><span className="fir-pe-label">Record by Woman Officer</span><span>{fir.recordedByWomanOfficer ? '✅ Yes' : '❌ No'}</span></div>}
              {fir.isDisabledVictim && <div className="fir-pe-field"><span className="fir-pe-label">Disabled Victim</span><span>✅ Yes</span></div>}
              {fir.interpreterOrEducatorName && <div className="fir-pe-field"><span className="fir-pe-label">Interpreter / Educator</span><span>{fir.interpreterOrEducatorName}</span></div>}
              {fir.isMagistrateStatementRecorded != null && <div className="fir-pe-field"><span className="fir-pe-label">Magistrate Statement</span><span>{fir.isMagistrateStatementRecorded ? '✅ Recorded' : '❌ Not Recorded'}</span></div>}
            </div>
          )}
        </div>
        <div className="dash-modal-footer">
          <button className="dash-btn primary" onClick={() => generateFirPdf(fir)} style={{ marginRight: 8 }}>
            📄 Download FIR PDF
          </button>
          <button className="dash-btn secondary" onClick={onClose}>Close</button>
        </div>
      </div>
    </div>
  );
};

export default PIDashboardPage;
