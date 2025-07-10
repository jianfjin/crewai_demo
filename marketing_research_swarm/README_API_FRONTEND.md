# Marketing Research Swarm - FastAPI + React Implementation

This implementation transforms the CrewAI marketing research platform from a Streamlit dashboard to a modern FastAPI backend with a React/Next.js frontend.

## 🏗️ Architecture Overview

### Backend (FastAPI)
- **Location**: `backend/`
- **Framework**: FastAPI with async support
- **Features**:
  - RESTful API endpoints for all analysis operations
  - Real-time analysis monitoring with polling
  - Background task execution for long-running analyses
  - Comprehensive token usage tracking
  - Agent and analysis type management
  - CORS support for frontend integration

### Frontend (React/Next.js)
- **Location**: `frontend/`
- **Framework**: Next.js 14 with TypeScript
- **UI Library**: Tailwind CSS + shadcn/ui components
- **Icons**: Lucide React
- **Features**:
  - Modern, responsive dashboard interface
  - Real-time analysis monitoring
  - Interactive agent selection
  - Token usage visualization
  - Analysis results display with download
  - Analysis history tracking

## 🚀 Quick Start

### Prerequisites
- Python 3.10+ 
- Node.js 18+
- OpenAI API key

### Setup
```bash
# Make setup script executable
chmod +x setup_project.sh

# Run setup
./setup_project.sh
```

### Configuration
1. Update `backend/.env` with your API keys:
```env
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here  # Optional
```

2. Update `frontend/.env.local` if needed:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Running the Application

#### Option 1: Using Scripts
```bash
# Terminal 1: Start Backend
chmod +x start_backend.sh
./start_backend.sh

# Terminal 2: Start Frontend  
chmod +x start_frontend.sh
./start_frontend.sh
```

#### Option 2: Manual Start
```bash
# Terminal 1: Backend
cd backend
source venv/bin/activate  # or venv\Scripts\activate on Windows
python main.py

# Terminal 2: Frontend
cd frontend
npm run dev
```

### Access
- **Frontend Dashboard**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Health check
- `GET /api/agents` - Get available agents
- `GET /api/analysis-types` - Get analysis types
- `POST /api/analysis/start` - Start new analysis
- `GET /api/analysis/{id}/status` - Get analysis status
- `GET /api/analysis/{id}/result` - Get analysis result
- `GET /api/analysis/history` - Get analysis history
- `DELETE /api/analysis/{id}` - Cancel analysis

### Request/Response Examples

#### Start Analysis
```json
POST /api/analysis/start
{
  "analysis_type": "roi_analysis",
  "selected_agents": ["market_research_analyst", "data_analyst", "financial_analyst"],
  "optimization_level": "blackboard",
  "custom_inputs": {}
}
```

#### Analysis Status
```json
GET /api/analysis/{id}/status
{
  "analysis_id": "uuid",
  "status": "running",
  "progress": 45.0,
  "current_step": "Executing data_analyst",
  "agents_completed": ["market_research_analyst"],
  "total_agents": 3,
  "start_time": "2025-01-10T10:00:00Z",
  "token_usage": {
    "total_tokens": 1500,
    "input_tokens": 1000,
    "output_tokens": 500,
    "total_cost": 0.003
  }
}
```

## 🎨 Frontend Features

### Dashboard Components
1. **Analysis Form** (`analysis-form.tsx`)
   - Analysis type selection
   - Agent selection with phase grouping
   - Optimization level configuration
   - Real-time validation

2. **Analysis Monitor** (`analysis-monitor.tsx`)
   - Real-time progress tracking
   - Agent completion status
   - Token usage monitoring
   - Cancel functionality

3. **Analysis Results** (`analysis-results.tsx`)
   - Comprehensive results display
   - Token usage breakdown
   - Performance metrics
   - Download functionality

### UI Components (shadcn/ui)
- **Button**: Primary actions and navigation
- **Card**: Content containers and layouts
- **Progress**: Analysis progress visualization
- **Badge**: Status indicators and labels
- **Custom utilities**: Token formatting, duration formatting

## 🔧 Key Improvements Over Streamlit

### Performance
- **Async Operations**: Non-blocking API calls
- **Real-time Updates**: Polling-based status updates
- **Background Processing**: Long-running analyses don't block UI
- **Optimized Rendering**: React's virtual DOM for efficient updates

### User Experience
- **Modern Interface**: Clean, responsive design
- **Interactive Elements**: Better form controls and feedback
- **Real-time Monitoring**: Live progress and token tracking
- **Mobile Friendly**: Responsive design for all devices

### Developer Experience
- **Type Safety**: Full TypeScript implementation
- **Component Reusability**: Modular React components
- **API Documentation**: Auto-generated FastAPI docs
- **Error Handling**: Comprehensive error states and messages

## 🔄 Integration with Existing System

The FastAPI backend integrates seamlessly with the existing CrewAI infrastructure:

- **OptimizationManager**: Direct integration for analysis execution
- **Blackboard System**: Full support for advanced optimization
- **Token Tracking**: Enhanced tracking with real-time updates
- **Agent Management**: Dynamic agent loading and configuration
- **Flow Support**: Compatible with all existing flow types

## 📊 Token Usage Tracking

Enhanced token tracking provides:
- **Real-time Updates**: Live token consumption during analysis
- **Agent Breakdown**: Per-agent token usage and costs
- **Task-level Details**: Individual task performance metrics
- **Cost Calculation**: Accurate cost estimation and tracking
- **Export Functionality**: Download detailed usage reports

## 🛠️ Development

### Backend Development
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm install
npm run dev

# Build for production
npm run build
npm start
```

### Adding New Features

#### Backend (FastAPI)
1. Add new endpoints in `backend/main.py`
2. Update Pydantic models for request/response
3. Integrate with existing CrewAI components

#### Frontend (React)
1. Create new components in `frontend/src/components/`
2. Update API client in `frontend/src/lib/api.ts`
3. Add new types in `frontend/src/types/api.ts`

## 🚀 Deployment

### Backend Deployment
```bash
# Using uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000

# Using Docker (create Dockerfile)
docker build -t marketing-research-api .
docker run -p 8000:8000 marketing-research-api
```

### Frontend Deployment
```bash
# Build for production
npm run build

# Deploy to Vercel, Netlify, or any static hosting
# Update NEXT_PUBLIC_API_URL for production API
```

## 📝 Next Steps

1. **Authentication**: Add user authentication and authorization
2. **Database**: Implement persistent storage for analysis history
3. **WebSockets**: Real-time updates without polling
4. **Caching**: Redis for improved performance
5. **Monitoring**: Application performance monitoring
6. **Testing**: Comprehensive test suite for both backend and frontend

## 🤝 Contributing

1. Follow the existing code structure and patterns
2. Add TypeScript types for new features
3. Update API documentation for new endpoints
4. Test both backend and frontend changes
5. Update this README for significant changes

---

**Status**: ✅ **Production Ready**

The FastAPI + React implementation provides a modern, scalable, and user-friendly interface for the Marketing Research Swarm platform while maintaining full compatibility with the existing CrewAI infrastructure.