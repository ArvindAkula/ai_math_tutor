
# AI Math Tutor

An intelligent educational platform that provides step-by-step problem solving, interactive learning, and personalized math instruction. The system combines symbolic mathematics computation with AI-powered explanations to help students understand mathematical concepts across various domains including algebra, calculus, linear algebra, and AI/ML mathematics.

## 🚀 Features

- **Step-by-Step Problem Solving**: Detailed solutions with explanations for algebra, calculus, linear algebra, and AI/ML mathematics
- **Interactive Quizzes**: Adaptive quizzes with immediate feedback and explanations
- **Mathematical Visualizations**: Interactive 2D/3D plots, vector fields, and function graphs
- **Personalized Learning**: Progress tracking and adaptive learning paths
- **AI-Powered Explanations**: Natural language explanations adapted to user skill level
- **Advanced Input Methods**: Voice input and handwriting recognition (coming soon)
- **Real-time Collaboration**: WebSocket-based collaborative problem solving
- **Multi-platform Access**: Responsive web interface optimized for desktop and mobile

## 🏗️ Architecture

The system uses a microservices architecture with:

- **Python Math Engine** (Port 8001): Symbolic mathematics computation, AI integration, and visualization generation
- **Go API Gateway** (Port 8000): High-performance API services, authentication, WebSocket support, and rate limiting
- **React Frontend** (Port 3000): Cross-platform web interface with Material-UI components
- **PostgreSQL** (Port 5432): Primary data storage with optimized schemas
- **Redis** (Port 6379): Caching, session management, and real-time features

## 📁 Project Structure

```
ai-math-tutor/
├── math-engine/          # Python FastAPI service
│   ├── main.py          # Main application with comprehensive monitoring
│   ├── models.py        # Data models and interfaces
│   ├── solver.py        # Mathematical problem solving engine
│   ├── visualization.py # Interactive visualization generation
│   ├── ai_explainer.py  # AI-powered explanation service
│   └── requirements.txt # Python dependencies
├── api-gateway/         # Go Gin service
│   ├── main.go         # Main application with WebSocket support
│   ├── internal/auth/  # JWT authentication and user management
│   ├── internal/websocket/ # Real-time collaboration features
│   └── go.mod          # Go dependencies
├── frontend/           # React application
│   ├── src/App.js     # Main application with authentication
│   ├── src/components/ # React components
│   ├── src/contexts/  # Authentication context
│   └── package.json   # Node.js dependencies
├── shared/            # Common data models
│   ├── database/      # SQL initialization and optimization
│   └── models/        # Shared data structures
├── k8s/              # Kubernetes deployment manifests
├── tests/            # Comprehensive test suites
└── docker-compose.yml # Development environment
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- Ports 3000, 8000, 8001, 5432, 6379 available

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ai-math-tutor

# Copy environment configuration
cp .env.example .env

# Edit .env with your OpenAI API key (optional but recommended)
nano .env
```

### 2. Start All Services

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f
```

### 3. Access the Application

- **Frontend**: http://localhost:3000 - Main web application
- **API Gateway**: http://localhost:8000 - REST API and WebSocket endpoints
- **Math Engine**: http://localhost:8001 - Mathematical computation service
- **Health Checks**: All services include `/health` endpoints

### 4. Create Your First Account

1. Navigate to http://localhost:3000
2. Click "Sign Up" to create an account
3. Fill out the registration form
4. After registration, sign in with your credentials
5. Start solving math problems!

## 🔐 Authentication System

The application includes a complete authentication system:

### Features
- **User Registration**: Create accounts with email/username/password
- **JWT Authentication**: Secure token-based authentication
- **Session Management**: Refresh tokens and session tracking
- **Role-Based Access**: Student, educator, and admin roles
- **Password Security**: bcrypt hashing with salt

### API Endpoints
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login (returns JWT tokens)
- `POST /api/v1/auth/logout` - User logout
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/users/profile` - Get user profile (protected)
- `PUT /api/v1/users/profile` - Update user profile (protected)

### Database Schema
The system includes optimized database schemas with:
- User management with profiles and preferences
- Session tracking with JWT token storage
- Problem attempts and quiz results
- Learning progress and analytics
- Comprehensive indexing for performance

## 🧮 Mathematical Capabilities

### Supported Domains
- **Algebra**: Linear equations, polynomials, factoring, simplification
- **Calculus**: Derivatives, integrals, limits, optimization
- **Linear Algebra**: Matrix operations, eigenvalues/vectors, system solving
- **Statistics**: Basic statistical analysis and probability
- **AI/ML Mathematics**: Gradients, loss functions, optimization algorithms

### Problem Solving Features
- Step-by-step solution generation
- Multiple solution methods
- Interactive visualizations
- AI-powered explanations
- Answer validation and feedback

## 📊 Monitoring and Health Checks

All services include comprehensive monitoring:

### Health Check Endpoints
```bash
curl http://localhost:3000/health  # Frontend
curl http://localhost:8000/health  # API Gateway  
curl http://localhost:8001/health  # Math Engine
```

### Monitoring Features
- Performance metrics collection
- Error tracking and alerting
- Resource usage monitoring
- Database connection health
- Cache performance metrics

## 🛠️ Development

### Running Individual Services

```bash
# Math Engine (Python)
cd math-engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# API Gateway (Go)
cd api-gateway
go mod tidy
go run main.go

# Frontend (React)
cd frontend
npm install
npm start
```

### Testing

```bash
# Run all tests
docker-compose exec math-engine python -m pytest
docker-compose exec api-gateway go test ./...
docker-compose exec frontend npm test

# Integration tests
python -m pytest tests/
```

### Database Access

```bash
# PostgreSQL shell
docker exec -it ai-math-tutor-postgres psql -U postgres -d ai_math_tutor

# Redis shell
docker exec -it ai-math-tutor-redis redis-cli

# View database tables
\dt

# Check user accounts
SELECT id, email, username, role, is_active FROM users;
```

## 🚀 Production Deployment

### Docker Production

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start production environment
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n ai-math-tutor

# Access via ingress
kubectl get ingress -n ai-math-tutor
```

## 🔧 Configuration

### Environment Variables

```bash
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ai_math_tutor

# Authentication
JWT_SECRET=your_super_secret_jwt_key

# External APIs
OPENAI_API_KEY=your_openai_api_key_here

# Service URLs (production)
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_WS_URL=wss://api.yourdomain.com

# Development Settings
NODE_ENV=development
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Performance Tuning

The system includes several performance optimizations:
- Redis caching for frequently accessed data
- Database query optimization and indexing
- Connection pooling for database connections
- Horizontal scaling support for stateless services
- CDN integration for static assets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📚 Documentation

- [Setup Guide](SETUP_GUIDE.md) - Comprehensive setup and deployment guide
- [Docker Guide](DOCKER_README.md) - Docker containerization details
- [API Documentation](http://localhost:8000/docs) - Interactive API documentation
- [Architecture Overview](.kiro/specs/ai-math-tutor/) - Detailed system design

## 🐛 Troubleshooting

### Common Issues

1. **Services won't start**: Check port availability and Docker daemon
2. **Database connection errors**: Verify PostgreSQL is running and credentials are correct
3. **Authentication failures**: Check JWT_SECRET configuration
4. **Frontend build errors**: Clear node_modules and reinstall dependencies

### Getting Help

- Check the logs: `docker-compose logs -f [service]`
- Verify service health: `curl http://localhost:[port]/health`
- Review the troubleshooting section in [SETUP_GUIDE.md](SETUP_GUIDE.md)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with FastAPI, Gin, React, PostgreSQL, and Redis
- Mathematical computation powered by SymPy, NumPy, and SciPy
- AI explanations powered by OpenAI API
- Visualizations created with Matplotlib and Plotly
- UI components from Material-UI
