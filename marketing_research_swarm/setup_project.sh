#!/bin/bash

echo "🚀 Setting up Marketing Research Swarm with FastAPI + React"
echo "============================================================"

# Create directories if they don't exist
mkdir -p backend
mkdir -p frontend

# Backend setup
echo "📦 Setting up FastAPI Backend..."
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment (Linux/Mac)
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source venv/bin/activate
# Activate virtual environment (Windows)
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
fi

# Install Python dependencies
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "⚠️  Please update backend/.env with your API keys"
fi

cd ..

# Frontend setup
echo "🎨 Setting up React Frontend..."
cd frontend

# Install Node.js dependencies
npm install

# Install additional shadcn/ui dependencies
npm install tailwindcss-animate

# Copy environment file
if [ ! -f .env.local ]; then
    cp .env.local.example .env.local
fi

cd ..

echo "✅ Setup complete!"
echo ""
echo "🔧 Next steps:"
echo "1. Update backend/.env with your OpenAI API key"
echo "2. Update frontend/.env.local if needed"
echo ""
echo "🚀 To start the application:"
echo "Backend:  cd backend && python main.py"
echo "Frontend: cd frontend && npm run dev"
echo ""
echo "📱 Access the application at: http://localhost:3000"