#!/bin/bash

# AI Math Tutor - Development Setup Script

echo "🧮 Setting up AI Math Tutor development environment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file from example if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ Please edit .env file with your actual API keys and configuration"
fi

# Build and start services
echo "🐳 Building and starting Docker containers..."
docker-compose up --build -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."

# Check Math Engine
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Math Engine is running on http://localhost:8001"
else
    echo "❌ Math Engine is not responding"
fi

# Check API Gateway
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API Gateway is running on http://localhost:8000"
else
    echo "❌ API Gateway is not responding"
fi

# Check Frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ Frontend is running on http://localhost:3000"
else
    echo "❌ Frontend is not responding"
fi

echo ""
echo "🎉 Setup complete! Your AI Math Tutor is ready for development."
echo ""
echo "📋 Available services:"
echo "   • Frontend:     http://localhost:3000"
echo "   • API Gateway:  http://localhost:8000"
echo "   • Math Engine:  http://localhost:8001"
echo "   • Database:     localhost:5432"
echo "   • Redis:        localhost:6379"
echo ""
echo "🛠️  Development commands:"
echo "   • View logs:    docker-compose logs -f"
echo "   • Stop all:     docker-compose down"
echo "   • Restart:      docker-compose restart"
echo ""
echo "📚 Next steps:"
echo "   1. Edit .env file with your API keys"
echo "   2. Open http://localhost:3000 in your browser"
echo "   3. Start implementing the remaining tasks!"