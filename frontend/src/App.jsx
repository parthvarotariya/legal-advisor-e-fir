import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import HomePage from './pages/citizen/HomePage';
import LoginPage from './pages/citizen/LoginPage';
import RegisterPage from './pages/citizen/RegisterPage';
import LegalAdvicePage from './pages/citizen/LegalAdvicePage';
import FileComplaintPage from './pages/citizen/FileComplaintPage';
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
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
