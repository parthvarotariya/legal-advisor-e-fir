import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Grid
} from '@mui/material';
import GavelIcon from '@mui/icons-material/Gavel';
import DescriptionIcon from '@mui/icons-material/Description';
import TrackChangesIcon from '@mui/icons-material/TrackChanges';

const HomePage = () => {
  const navigate = useNavigate();
  const { user, logout, isAuthenticated } = useAuth();

  return (
    <Box sx={{ minHeight: '100vh', bgcolor: '#f5f5f5', pb: 4 }}>
      {/* Header */}
      <Paper elevation={2} sx={{ py: 2, px: 4, borderRadius: 0 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Typography variant="h5" component="h1" fontWeight="bold">
            Legal Advisor e-FIR System
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            {isAuthenticated ? (
              <>
                <Typography variant="body1" color="text.secondary">
                  Welcome, <strong>{user?.name}</strong>
                </Typography>
                <Button variant="outlined" onClick={logout} color="error">
                  Logout
                </Button>
              </>
            ) : (
              <>
                <Button variant="outlined" onClick={() => navigate('/login')}>
                  Login
                </Button>
                <Button variant="contained" onClick={() => navigate('/register')}>
                  Register
                </Button>
              </>
            )}
          </Box>
        </Box>
      </Paper>

      <Container maxWidth="lg" sx={{ mt: 4 }}>
        {/* Hero Section */}
        <Paper 
          elevation={0} 
          sx={{ 
            p: 6, 
            mb: 5, 
            textAlign: 'center',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            color: 'white',
            borderRadius: 3
          }}
        >
          <GavelIcon sx={{ fontSize: 80, mb: 2, opacity: 0.9 }} />
          <Typography variant="h3" gutterBottom fontWeight="bold">
            AI-Assisted Legal Advisor & e-FIR System
          </Typography>
          <Typography variant="h6" sx={{ mb: 3, opacity: 0.95 }}>
            Get legal guidance from our AI assistant and file complaints online
          </Typography>
          <Typography variant="body1" sx={{ opacity: 0.9 }}>
            Compliant with Section 154 of CrPC
          </Typography>
        </Paper>

        {/* Features Grid */}
        <Typography variant="h4" gutterBottom align="center" sx={{ mb: 4, mt: 2, fontWeight: 'bold' }}>
          Our Services
        </Typography>
        
        <Grid container spacing={3} sx={{ mb: 5 }}>
          <Grid item xs={12} sm={6} md={4}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                textAlign: 'center', 
                display: 'flex',
                flexDirection: 'column',
                minHeight: '350px',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': { 
                  transform: 'translateY(-5px)',
                  boxShadow: 6
                }
              }}
            >
              <GavelIcon sx={{ fontSize: 60, color: 'primary.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Get Legal Advice
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3, flexGrow: 1 }}>
                Submit your complaint details and get automatic crime classification with legal guidance.
              </Typography>
              <Button 
                variant="contained" 
                size="large"
                fullWidth
                onClick={() => navigate('/legal-advice')}
                disabled={!isAuthenticated}
              >
                GET ADVICE
              </Button>
              {!isAuthenticated && (
                <Typography variant="caption" color="error" display="block" sx={{ mt: 1 }}>
                  Please login to continue
                </Typography>
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                textAlign: 'center', 
                display: 'flex',
                flexDirection: 'column',
                minHeight: '350px',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': { 
                  transform: 'translateY(-5px)',
                  boxShadow: 6
                }
              }}
            >
              <DescriptionIcon sx={{ fontSize: 60, color: 'success.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom fontWeight="bold">
                File Complaint
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3, flexGrow: 1 }}>
                Submit your complaint online. Police will review and register FIR if applicable.
              </Typography>
              <Button 
                variant="contained" 
                size="large"
                fullWidth
                color="success"
                onClick={() => navigate('/file-complaint')}
                disabled={!isAuthenticated}
              >
                FILE COMPLAINT
              </Button>
              {!isAuthenticated && (
                <Typography variant="caption" color="error" display="block" sx={{ mt: 1 }}>
                  Please login to continue
                </Typography>
              )}
            </Paper>
          </Grid>

          <Grid item xs={12} sm={6} md={4}>
            <Paper 
              elevation={3} 
              sx={{ 
                p: 3, 
                textAlign: 'center', 
                display: 'flex',
                flexDirection: 'column',
                minHeight: '350px',
                transition: 'transform 0.2s, box-shadow 0.2s',
                '&:hover': { 
                  transform: 'translateY(-5px)',
                  boxShadow: 6
                }
              }}
            >
              <TrackChangesIcon sx={{ fontSize: 60, color: 'info.main', mb: 2 }} />
              <Typography variant="h6" gutterBottom fontWeight="bold">
                Track Complaint
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 3, flexGrow: 1 }}>
                Check the status of your complaint using your Complaint ID.
              </Typography>
              <Button 
                variant="contained" 
                size="large"
                fullWidth
                color="info"
                onClick={() => navigate('/track-complaint')}
              >
                TRACK STATUS
              </Button>
            </Paper>
          </Grid>
        </Grid>

        {/* Info Section */}
        <Paper 
          elevation={2} 
          sx={{ 
            p: 4, 
            bgcolor: '#e3f2fd',
            borderLeft: '5px solid #2196f3',
            mb: 4
          }}
        >
          <Typography variant="h5" gutterBottom fontWeight="bold" color="primary">
            Important Information
          </Typography>
          <Typography variant="body1" component="div" sx={{ lineHeight: 2 }}>
            ✓ This system is compliant with Section 154 of the Criminal Procedure Code (CrPC)
            <br />
            ✓ Citizens file complaints, not FIRs - Police officers register the official FIR
            <br />
            ✓ AI provides guidance only - final legal authority rests with police
            <br />
            ✓ You can track your complaint status at any time
          </Typography>
        </Paper>
      </Container>
    </Box>
  );
};

export default HomePage;
