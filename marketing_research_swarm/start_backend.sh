#!/bin/bash

echo "🚀 Starting FastAPI Backend..."

cd backend

# Activate virtual environment
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
fi

# Start the FastAPI server
echo "Starting server on http://localhost:8000"
python main.py