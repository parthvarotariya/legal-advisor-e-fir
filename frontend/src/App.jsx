import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import HomePage from './pages/citizen/HomePage';
import LoginPage from './pages/citizen/LoginPage';
import RegisterPage from './pages/citizen/RegisterPage';
import LegalAdvicePage from './pages/citizen/LegalAdvicePage';
import FileComplaintPage from './pages/citizen/FileComplaintPage';
import TrackComplaintPage from './pages/citizen/TrackComplaintPage';
import SuperAdminPage from './pages/admin/SuperAdminPage';
import AdminLoginPage from './pages/admin/AdminLoginPage';
import PoliceLoginPage from './pages/police/PoliceLoginPage';
import PoliceDashboardPage from './pages/police/PoliceDashboardPage';
import PIDashboardPage from './pages/police/PIDashboardPage';
import PsiDashboardPage from './pages/police/PsiDashboardPage';
import DSPDashboardPage from './pages/police/DSPDashboardPage';
import './App.css';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/legal-advice" element={<LegalAdvicePage />} />
          <Route path="/file-complaint" element={<FileComplaintPage />} />
          <Route path="/track-complaint" element={<TrackComplaintPage />} />
          <Route path="/admin/login" element={<AdminLoginPage />} />
          <Route path="/super-admin" element={<SuperAdminPage />} />
          <Route path="/police/login" element={<PoliceLoginPage />} />
          <Route path="/police/pi-dashboard" element={<PIDashboardPage />} />
          <Route path="/police/psi-dashboard" element={<PsiDashboardPage />} />
          <Route path="/police/dsp-dashboard" element={<DSPDashboardPage />} />
          <Route path="/police/dashboard" element={<PoliceDashboardPage />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
