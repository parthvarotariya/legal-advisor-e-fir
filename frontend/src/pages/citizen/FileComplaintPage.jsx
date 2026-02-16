import { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import api from '../../services/api';

const FileComplaintPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { user } = useAuth();
  
  const [formData, setFormData] = useState({
    description: location.state?.complaint || '',
    policeStationId: ''
  });
  const [policeStations, setPoliceStations] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState(location.state?.aiAnalysis || null);

  // Fetch police stations from backend
  useEffect(() => {
    const fetchPoliceStations = async () => {
      try {
        const response = await api.get('/police-stations');
        setPoliceStations(response.data);
      } catch (err) {
        console.error('Failed to fetch police stations:', err);
      }
    };
    fetchPoliceStations();
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
    setError('');
  };

  const analyzeWithAI = async () => {
    if (!formData.description || formData.description.length < 50) {
      setError('Please provide a detailed description (at least 50 characters) for AI analysis');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:5000/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ complaint: formData.description }),
      });

      if (response.ok) {
        const data = await response.json();
        setAiAnalysis(data);
      } else {
        setError('AI analysis unavailable. You can still submit your complaint.');
      }
    } catch (err) {
      setError('AI service not available. You can still submit your complaint.');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (formData.description.length < 50) {
      setError('Description must be at least 50 characters');
      return;
    }

    if (!formData.policeStationId) {
      setError('Please select a police station');
      return;
    }

    setLoading(true);

    try {
      const complaintData = {
        description: formData.description,
        predictedCategory: aiAnalysis?.category_full || null,
        userId: user?.id || 1,
        policeStationId: parseInt(formData.policeStationId)
      };

      // TODO: Send to backend when complaint endpoint is available
      // const response = await fetch('http://localhost:8080/api/complaints', {
      //   method: 'POST',
      //   headers: { 'Content-Type': 'application/json' },
      //   body: JSON.stringify(complaintData)
      // });

      // For now, just show success
      setSuccess(true);
      setTimeout(() => {
        navigate('/track-complaint');
      }, 2000);

    } catch (err) {
      setError('Failed to submit complaint. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50">
      {/* Header */}
      <div className="bg-white shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <button 
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-4 py-2 text-indigo-600 hover:text-indigo-800 font-semibold hover:bg-indigo-50 rounded-lg transition-all"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to Home
          </button>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            File Complaint
          </h1>
          <div className="w-32"></div>
        </div>
      </div>

      {success && (
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="bg-green-50 border-2 border-green-500 rounded-2xl p-8 text-center animate-pulse">
            <div className="text-6xl mb-4">✅</div>
            <h2 className="text-3xl font-bold text-green-900 mb-3">Complaint Submitted Successfully!</h2>
            <p className="text-green-700 text-lg">Redirecting to track complaints...</p>
          </div>
        </div>
      )}

      {!success && (
        <div className="max-w-4xl mx-auto px-6 py-8">
          {/* Info Banner */}
          <div className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-2xl p-6 mb-8 shadow-xl">
            <div className="flex items-start gap-4">
              <div className="text-4xl">📋</div>
              <div>
                <h2 className="text-2xl font-bold mb-2">File Your Complaint</h2>
                <p className="text-blue-100">Provide detailed information about the incident. Our AI will help classify your complaint for faster processing.</p>
              </div>
            </div>
          </div>

          {/* Main Form */}
          <div className="bg-white rounded-2xl shadow-2xl p-8">
            {error && (
              <div className="bg-red-50 border-l-4 border-red-500 rounded-lg p-4 mb-6">
                <div className="flex items-center gap-3">
                  <span className="text-2xl">⚠️</span>
                  <span className="text-red-800 font-semibold">{error}</span>
                </div>
              </div>
            )}

            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Description */}
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Incident Description <span className="text-red-500">*</span>
                </label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 resize-none transition-all"
                  placeholder="Describe what happened in detail. Include: What happened? When? Where? Who was involved?"
                  rows={6}
                  required
                />
                <div className="flex items-center justify-between mt-2">
                  <span className="text-sm text-gray-500">{formData.description.length} characters</span>
                  {formData.description.length >= 50 && (
                    <span className="text-sm text-green-600 font-semibold flex items-center gap-1">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      Valid
                    </span>
                  )}
                </div>
                {formData.description.length >= 50 && (
                  <button
                    type="button"
                    onClick={analyzeWithAI}
                    disabled={loading || aiAnalysis}
                    className="mt-3 px-5 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 font-semibold"
                  >
                    <span>🤖</span>
                    {aiAnalysis ? 'AI Analysis Complete' : 'Get AI Classification'}
                  </button>
                )}
              </div>

              {/* AI Analysis Result */}
              {aiAnalysis && (
                <div className="bg-gradient-to-r from-purple-50 to-indigo-50 border-2 border-purple-300 rounded-xl p-6">
                  <div className="flex items-center gap-3 mb-4">
                    <span className="text-3xl">🎯</span>
                    <h3 className="text-xl font-bold text-gray-900">AI Classification</h3>
                  </div>
                  <div className="space-y-3">
                    <div className="bg-white rounded-lg p-4">
                      <p className="text-sm text-gray-600 mb-1">Category</p>
                      <p className="text-lg font-bold text-indigo-900">{aiAnalysis.category_full}</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Police Station Selection */}
              <div>
                <label className="block text-sm font-bold text-gray-700 mb-2">
                  Select Police Station <span className="text-red-500">*</span>
                </label>
                <select
                  name="policeStationId"
                  value={formData.policeStationId}
                  onChange={handleChange}
                  required
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all bg-white"
                >
                  <option value="">-- Choose a police station --</option>
                  {policeStations.map(station => (
                    <option key={station.stationId} value={station.stationId}>
                      {station.stationName} - {station.district}, {station.state}
                    </option>
                  ))}
                </select>
                <p className="text-sm text-gray-500 mt-1">Select the nearest police station to the incident location</p>
              </div>

              {/* Important Note */}
              <div className="bg-yellow-50 border-l-4 border-yellow-400 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <span className="text-2xl">ℹ️</span>
                  <div>
                    <p className="font-bold text-yellow-900 mb-1">Important Information</p>
                    <ul className="text-sm text-yellow-800 space-y-1">
                      <li>• Your complaint will be reviewed by police authorities</li>
                      <li>• An FIR will be registered if necessary</li>
                      <li>• You can track your complaint status anytime</li>
                      <li>• False complaints may result in legal action</li>
                    </ul>
                  </div>
                </div>
              </div>

              {/* Submit Buttons */}
              <div className="flex gap-4 pt-4">
                <button
                  type="submit"
                  disabled={loading}
                  className="flex-1 px-8 py-4 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-xl hover:from-indigo-700 hover:to-purple-700 font-bold disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-xl text-lg flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-6 h-6 border-3 border-white border-t-transparent rounded-full animate-spin"></div>
                      Submitting...
                    </>
                  ) : (
                    <>
                      <span>📤</span>
                      Submit Complaint
                    </>
                  )}
                </button>
                <button
                  type="button"
                  onClick={() => navigate('/')}
                  disabled={loading}
                  className="px-8 py-4 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 font-bold disabled:opacity-50 transition-all border-2 border-gray-300"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>

          {/* Help Section */}
          <div className="mt-8 bg-gradient-to-r from-green-50 to-teal-50 rounded-2xl p-6 border-2 border-green-200">
            <div className="flex items-start gap-4">
              <span className="text-4xl">💡</span>
              <div>
                <h3 className="text-xl font-bold text-gray-900 mb-3">Tips for Filing Complaint</h3>
                <ul className="space-y-2 text-gray-700">
                  <li className="flex items-start gap-2">
                    <span className="text-green-600 font-bold">✓</span>
                    <span>Provide as much detail as possible about the incident</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-600 font-bold">✓</span>
                    <span>Include specific dates, times, and locations</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-600 font-bold">✓</span>
                    <span>Mention any witnesses or evidence you have</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-green-600 font-bold">✓</span>
                    <span>Be truthful and accurate in your description</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileComplaintPage;
