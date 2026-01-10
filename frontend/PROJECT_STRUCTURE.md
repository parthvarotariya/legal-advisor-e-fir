# Legal Advisor e-FIR System - Frontend Architecture

## Project Overview

A React-based frontend for an AI-assisted e-FIR and Legal Advisor system that complies with **Section 154 of CrPC**. Citizens interact with an AI assistant, file complaints, and police officers review and register official FIRs.

---

## Technology Stack

- **Framework**: React 18 with Vite
- **UI Library**: Material-UI (MUI)
- **HTTP Client**: Axios
- **Routing**: React Router DOM
- **State Management**: React Context API
- **Backend**: Spring Boot (http://localhost:8080)

---

## Application Flow

### Citizen Journey (Public Access - No Authentication)

```
1. Landing Page
   ↓
2. AI Legal Assistant (Chat Interface)
   - Citizen describes incident
   - AI provides legal guidance
   - AI suggests crime category
   - AI determines if cognizable offence
   ↓
3. Complaint Form (if citizen proceeds)
   - Informant details
   - Incident details (place, date, time)
   - Description
   - Accused/witness info (optional)
   ↓
4. Submit Complaint
   - Stored as "Complaint" (NOT FIR)
   - Status: "Pending"
   - Citizen receives Complaint ID
   ↓
5. Track Complaint
   - Check status using Complaint ID
   - View progress (Pending → Under Review → FIR Registered)
```

### Police Journey (Authenticated Access)

```
1. Police Login
   ↓
2. Police Dashboard
   - View all pending complaints
   ↓
3. Review Complaint
   - Read complaint details
   - Verify if cognizable
   ↓
4. Register FIR (if approved)
   - Assign IPC sections
   - Generate FIR number
   - Assign investigating officer
   - Change status to "FIR Registered"
   ↓
5. FIR becomes legally valid document
```

---

## Folder Structure Explained

```
frontend/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── common/          # Shared across citizen & police
│   │   ├── citizen/         # Citizen-specific components
│   │   └── police/          # Police-specific components
│   │
│   ├── pages/               # Full page components (routes)
│   │   ├── citizen/         # Citizen-facing pages
│   │   └── police/          # Police-facing pages
│   │
│   ├── services/            # API communication with backend
│   │   ├── api.js           # Base axios configuration
│   │   ├── complaintService.js   # Complaint operations
│   │   └── aiService.js     # AI chat operations (future)
│   │
│   ├── context/             # Global state management
│   │   └── AuthContext.jsx  # Police authentication state (future)
│   │
│   ├── utils/               # Helper functions
│   │   └── validators.js    # Form validation helpers (future)
│   │
│   ├── App.jsx              # Main app component with routing
│   ├── main.jsx             # React entry point
│   └── index.css            # Global styles
│
├── public/                  # Static assets
├── package.json             # Dependencies
└── vite.config.js           # Vite configuration
```

---

## Detailed File Documentation

### 📁 **services/** - Backend Communication Layer

#### **services/api.js**
**Purpose**: Base axios configuration for all API calls

**Key Features**:
- Base URL: `http://localhost:8080/api`
- Request interceptor: Automatically adds JWT token for authenticated requests
- Response interceptor: Handles 401 errors globally (redirects to login)
- 10-second timeout for all requests

**Code Flow**:
```javascript
// 1. Import in other services
import api from './api';

// 2. Make API calls
api.get('/complaints')      // GET request
api.post('/complaints', data)  // POST request
api.put('/complaints/1', data) // PUT request
api.delete('/complaints/1')    // DELETE request
```

**When Token is Added**:
- Police login → Token stored in localStorage
- All subsequent police API calls include: `Authorization: Bearer <token>`
- Citizen calls → No token (public access)

---

#### **services/complaintService.js**
**Purpose**: Handles all complaint-related API operations

**Functions**:

1. **`submitComplaint(complaintData)`**
   - **Used by**: Citizen complaint form
   - **Endpoint**: `POST /api/complaints`
   - **No Auth Required**: Public endpoint
   - **Input**: 
     ```javascript
     {
       informantName: "John Doe",
       informantContact: "9876543210",
       incidentDate: "2026-01-10",
       incidentTime: "14:30",
       incidentPlace: "Market Street",
       description: "Theft of mobile phone",
       accusedName: "Unknown",
       witnessName: ""
     }
     ```
   - **Returns**: Complaint object with generated ID
   - **Error Handling**: Catches and throws backend errors

2. **`trackComplaint(complaintId)`**
   - **Used by**: Track complaint page
   - **Endpoint**: `GET /api/complaints/{id}`
   - **Returns**: Full complaint details including status
   - **Example Usage**:
     ```javascript
     const complaint = await trackComplaint(12345);
     console.log(complaint.status); // "Pending" or "FIR Registered"
     ```

3. **`getComplaintStatus(complaintId)`**
   - **Used by**: Quick status check
   - **Endpoint**: `GET /api/complaints/{id}/status`
   - **Returns**: Status object only
   - **Lighter than trackComplaint** (doesn't fetch full details)

**Data Flow**:
```
Citizen Form → submitComplaint() → Spring Boot Controller
                                 ↓
                            Saved to Database
                                 ↓
                            Returns Complaint ID
                                 ↓
                          Display to Citizen
```

---

### 📁 **components/** - UI Building Blocks

#### **components/common/** (Shared Components)
Will contain:
- `Navbar.jsx` - Navigation bar for both citizen & police
- `Footer.jsx` - Common footer
- `LoadingSpinner.jsx` - Loading indicator
- `ErrorMessage.jsx` - Error display component

**Why Common?**
- Reusable across the entire app
- Consistent UI/UX
- Single point of maintenance

---

#### **components/citizen/** (Citizen-Specific)
Will contain:
- `AIChat.jsx` - Chat interface with AI assistant
- `ComplaintForm.jsx` - Complaint submission form
- `ComplaintTracker.jsx` - Track complaint component
- `ChatMessage.jsx` - Individual chat bubble

**Characteristics**:
- No authentication required
- Public-facing components
- User-friendly language (no legal jargon)

---

#### **components/police/** (Police-Specific)
Will contain (future):
- `ComplaintList.jsx` - List of pending complaints
- `FIRForm.jsx` - FIR registration form
- `IPCSelector.jsx` - IPC section selector
- `InvestigatorAssignment.jsx` - Assign investigating officer

**Characteristics**:
- Requires authentication
- Professional interface
- Legal/administrative features

---

### 📁 **pages/** - Full Page Components

#### **pages/citizen/** (Citizen Pages)
Will contain:
- `HomePage.jsx` - Landing page with options
- `AIAssistantPage.jsx` - AI chat page
- `ComplaintPage.jsx` - Complaint form page
- `TrackComplaintPage.jsx` - Track complaint status

**Routing Structure**:
```
/                     → HomePage
/ai-assistant         → AIAssistantPage
/file-complaint       → ComplaintPage
/track-complaint      → TrackComplaintPage
```

**Page Flow**:
1. **HomePage**: 
   - Welcome message
   - Two buttons: "Talk to AI Assistant" | "Track Complaint"

2. **AIAssistantPage**:
   - Chat interface
   - AI provides guidance
   - Button: "Proceed to File Complaint" (appears after chat)

3. **ComplaintPage**:
   - Multi-step form
   - Preview before submit
   - Success message with Complaint ID

4. **TrackComplaintPage**:
   - Input: Complaint ID
   - Display: Status, timeline, FIR number (if registered)

---

#### **pages/police/** (Police Pages - Future)
Will contain:
- `LoginPage.jsx` - Police authentication
- `DashboardPage.jsx` - Overview of complaints
- `ReviewComplaintPage.jsx` - Review single complaint
- `RegisterFIRPage.jsx` - FIR registration form

---

### 📁 **context/** - State Management

#### **context/AuthContext.jsx** (Future)
**Purpose**: Manage police authentication state globally

**Will Provide**:
- `user` - Current logged-in police officer
- `isAuthenticated` - Boolean
- `login(credentials)` - Login function
- `logout()` - Logout function

**Usage**:
```javascript
// In any component
const { user, isAuthenticated, logout } = useAuth();

if (!isAuthenticated) {
  return <Navigate to="/police/login" />;
}
```

---

### 📁 **utils/** - Helper Functions

#### **utils/validators.js** (Future)
**Purpose**: Form validation helpers

**Will Contain**:
- `validatePhone(phone)` - Indian phone number validation
- `validateEmail(email)` - Email format check
- `validateDate(date)` - Date range validation
- `validateComplaintForm(data)` - Complete form validation

---

## Data Models

### Complaint Object (Frontend → Backend)

```javascript
{
  // Informant Details
  informantName: String (required),
  informantFatherName: String,
  informantAge: Number,
  informantAddress: String (required),
  informantContact: String (required, 10 digits),
  informantEmail: String (optional),
  
  // Incident Details
  incidentDate: String (YYYY-MM-DD, required),
  incidentTime: String (HH:MM, required),
  incidentPlace: String (required),
  policeStationArea: String,
  
  // Description
  description: String (required, min 50 chars),
  
  // Accused (Optional)
  accusedName: String,
  accusedAddress: String,
  accusedDescription: String,
  
  // Witness (Optional)
  witnessName: String,
  witnessContact: String,
  
  // Auto-Generated
  complaintId: Number (backend generated),
  status: "PENDING" (default),
  submittedDate: Timestamp (backend),
  aiSuggestedCategory: String (from AI chat)
}
```

### Complaint Status Lifecycle

```
PENDING
  ↓ (Police reviews)
UNDER_REVIEW
  ↓ (Police approves)
FIR_REGISTERED
  ↓ (Investigation starts)
UNDER_INVESTIGATION
  ↓ (Case resolved)
CLOSED
```

---

## API Endpoints (Expected from Spring Boot)

### Public Endpoints (No Auth)
```
POST   /api/complaints              - Submit complaint
GET    /api/complaints/{id}         - Get complaint by ID
GET    /api/complaints/{id}/status  - Get complaint status
POST   /api/ai/chat                 - Send message to AI
POST   /api/ai/analyze              - Analyze incident for category
```

### Protected Endpoints (Police Only)
```
POST   /api/auth/login              - Police login
GET    /api/complaints              - Get all complaints
PUT    /api/complaints/{id}/review  - Mark under review
POST   /api/fir/register            - Register FIR
PUT    /api/fir/{id}/assign         - Assign investigator
```

---

## Environment Setup

### Prerequisites
```bash
Node.js >= 18
npm >= 9
Spring Boot backend running on http://localhost:8080
```

### Installation
```bash
cd frontend
npm install
npm run dev  # Starts dev server on http://localhost:5173
```

### Build for Production
```bash
npm run build      # Creates dist/ folder
npm run preview    # Preview production build
```

---

## Security Considerations

### Citizen Side (Public)
- ✅ No authentication required
- ✅ Rate limiting on complaint submission (backend)
- ✅ Input validation and sanitization
- ✅ CAPTCHA for complaint form (future)

### Police Side (Protected)
- ✅ JWT-based authentication
- ✅ Token stored in localStorage
- ✅ Token expires after 24 hours
- ✅ Auto-logout on 401 response
- ✅ Role-based access control (backend)

---

## Integration with Backend

### CORS Configuration Required in Spring Boot
```java
@Configuration
public class CorsConfig {
    @Bean
    public WebMvcConfigurer corsConfigurer() {
        return new WebMvcConfigurer() {
            @Override
            public void addCorsMappings(CorsRegistry registry) {
                registry.addMapping("/api/**")
                        .allowedOrigins("http://localhost:5173")
                        .allowedMethods("GET", "POST", "PUT", "DELETE")
                        .allowedHeaders("*")
                        .allowCredentials(true);
            }
        };
    }
}
```

---

## Current Development Status

### ✅ Completed
- [x] Project structure created
- [x] Base API configuration (api.js)
- [x] Complaint service (complaintService.js)
- [x] Folder organization

### 🚧 In Progress
- [ ] AI chat service
- [ ] Citizen pages
- [ ] Complaint form component

### 📋 Pending
- [ ] Police authentication
- [ ] Police dashboard
- [ ] FIR registration
- [ ] UI/UX design with Material-UI

---

## How Components Connect

```
App.jsx (Routing)
    ↓
CitizenHomePage
    ↓
AIAssistantPage (uses AIChat component)
    ↓ (calls aiService.js)
Spring Boot AI Endpoint
    ↓ (AI suggests: proceed with complaint)
ComplaintPage (uses ComplaintForm component)
    ↓ (calls complaintService.submitComplaint)
Spring Boot saves Complaint
    ↓ (returns Complaint ID)
Success Page (Display ID)
    ↓
TrackComplaintPage
    ↓ (calls complaintService.trackComplaint)
Display Status & Details
```

---

## Next Steps

1. Create AI chat service and component
2. Build citizen home page
3. Implement complaint form with validation
4. Add complaint tracking functionality
5. Style with Material-UI
6. Connect to actual Spring Boot backend
7. Test complete citizen flow
8. Move to police side development

---

## Development Guidelines

### Component Naming
- Page components: `XxxPage.jsx`
- Regular components: `XxxComponent.jsx` or `Xxx.jsx`
- Services: `xxxService.js` (camelCase)

### API Call Pattern
```javascript
// Always use try-catch
const handleSubmit = async () => {
  try {
    setLoading(true);
    const result = await submitComplaint(data);
    setSuccess(true);
  } catch (error) {
    setError(error.message);
  } finally {
    setLoading(false);
  }
};
```

### State Management
- Local state: `useState` for component-specific data
- Global state: Context API for auth, theme
- Server state: Direct API calls (no Redux needed for this project)

---

## Contact & Support

For questions about this architecture, refer to:
- **Legal Framework**: CrPC Section 154
- **Backend API**: Spring Boot documentation
- **Frontend**: React + Vite official docs
- **UI Components**: Material-UI documentation

---

**Last Updated**: January 10, 2026  
**Version**: 1.0  
**Author**: Development Team
