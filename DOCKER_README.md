# Docker Setup for AI Math Tutor

This document describes the Docker containerization setup for the AI Math Tutor application, including recent fixes and optimizations.

## Overview

The AI Math Tutor uses a multi-service Docker architecture with the following components:

- **Math Engine** (Python FastAPI): Core mathematical computation, AI explanations, and visualization generation
- **API Gateway** (Go Gin): REST API, JWT authentication, WebSocket support, and rate limiting
- **Frontend** (React): Responsive web application with Material-UI components
- **PostgreSQL**: Primary database with optimized schemas and indexing
- **Redis**: Caching, session management, and real-time features

## Recent Updates and Fixes

### Authentication System Enhancements
- ✅ **Complete JWT Authentication**: Registration, login, logout, and token refresh
- ✅ **Database Schema Fixes**: Added missing columns and corrected table structures
- ✅ **Session Management**: Proper session tracking with refresh tokens
- ✅ **Frontend Integration**: Updated React components to work with API responses

### Development Experience Improvements
- ✅ **Hot Reloading Fix**: Added polling environment variables to prevent constant refreshing
- ✅ **Import Path Resolution**: Fixed Python module imports in containerized environment
- ✅ **Proxy Configuration**: Corrected frontend-to-backend communication
- ✅ **Error Handling**: Enhanced error reporting and debugging capabilities

### Performance Optimizations
- ✅ **Memory Allocation**: Increased container memory limits for stable operation
- ✅ **Database Optimization**: TEXT columns for JWT tokens, proper indexing
- ✅ **Caching Strategy**: Redis integration for improved response times
- ✅ **Health Checks**: Comprehensive health monitoring for all services

## Quick Start

### Development Environment

1. Copy environment variables:
   ```bash
   cp .env.example .env
   ```

2. Start all services:
   ```bash
   make up
   # or
   docker-compose up -d
   ```

3. Access the application:
   - Frontend: http://localhost:3000
   - API Gateway: http://localhost:8000
   - Math Engine: http://localhost:8001

### Production Environment

1. Configure production environment:
   ```bash
   cp .env.prod .env.production
   # Edit .env.production with your production values
   ```

2. Start production services:
   ```bash
   make up-prod
   # or
   docker-compose -f docker-compose.prod.yml --env-file .env.production up -d
   ```

## Docker Images

### Multi-Stage Builds

All services use multi-stage Docker builds for optimization:

- **Development stage**: Includes development tools and hot reloading
- **Production stage**: Minimal runtime environment with security hardening

### Image Sizes

The multi-stage builds significantly reduce production image sizes:

- **Math Engine**: ~200MB (vs ~800MB without multi-stage)
- **API Gateway**: ~15MB (vs ~300MB without multi-stage)  
- **Frontend**: ~25MB (vs ~200MB without multi-stage)

## Security Features

### Container Security

- Non-root users in all containers
- Read-only file systems where possible
- Minimal base images (Alpine Linux, scratch)
- Security headers and best practices

### Network Security

- Isolated Docker network
- Service-to-service communication only
- No unnecessary port exposure

## Development Features

### Hot Reloading

Development containers include hot reloading:

- **Math Engine**: uvicorn with --reload
- **API Gateway**: Air for Go hot reloading
- **Frontend**: React development server

### Volume Mounts

Development environment mounts source code for live editing:

```yaml
volumes:
  - ./math-engine:/app:rw
  - ./api-gateway:/app:rw
  - ./frontend:/app:rw
```

## Available Commands

### Makefile Commands

```bash
# Build and start
make build          # Build development images
make build-prod     # Build production images
make up             # Start development environment
make up-prod        # Start production environment

# Management
make down           # Stop all services
make clean          # Remove all containers and images
make logs           # View logs from all services

# Testing
make test           # Run all tests
make test-containers # Run container-specific tests

# Health checks
make health-check   # Check service health status

# Development helpers
make dev-shell-math    # Open shell in math engine
make dev-shell-api     # Open shell in API gateway
make dev-shell-frontend # Open shell in frontend

# Database helpers
make db-migrate     # Run database migrations
make db-shell       # Open database shell
make redis-shell    # Open Redis shell
```

### Direct Docker Compose Commands

```bash
# Development
docker-compose up -d                    # Start all services
docker-compose down                     # Stop all services
docker-compose logs -f [service]        # View logs
docker-compose exec [service] /bin/bash # Open shell

# Production
docker-compose -f docker-compose.prod.yml up -d
docker-compose -f docker-compose.prod.yml down
```

## Environment Variables

### Required Variables

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ai_math_tutor

# Authentication
JWT_SECRET=your_jwt_secret_key

# External APIs
OPENAI_API_KEY=your_openai_api_key

# Service URLs (for production)
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_WS_URL=wss://api.yourdomain.com
```

### Environment Files

- `.env.example`: Template with default development values
- `.env.prod`: Production configuration template
- `.env`: Local development overrides (git-ignored)

## Health Checks

All services include health checks:

```bash
# Check individual services
curl http://localhost:8001/health  # Math Engine
curl http://localhost:8000/health  # API Gateway
curl http://localhost:3000/health  # Frontend

# Check all services
make health-check
```

## Monitoring and Logs

### Centralized Logging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f math-engine
docker-compose logs -f api-gateway
docker-compose logs -f frontend
```

### Resource Monitoring

```bash
# View resource usage
docker stats

# View container details
docker-compose ps
```

## Troubleshooting

### Recently Fixed Issues

#### 1. Authentication System Failures

**Problem**: Login and registration returning 500 errors
**Root Cause**: Database schema mismatch and JWT token size limitations
**Solution Applied**:
```bash
# Fixed database schema
docker exec -it ai-math-tutor-postgres psql -U postgres -d ai_math_tutor -c "
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'student';
ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP;
ALTER TABLE user_sessions ALTER COLUMN refresh_token TYPE TEXT;
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS is_revoked BOOLEAN DEFAULT FALSE;
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS user_agent VARCHAR(255);
ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);
"
```

#### 2. React Hot Reloading Issues

**Problem**: Constant page refreshing in development
**Root Cause**: Docker file watching system conflicts with React HMR
**Solution Applied**: Added polling environment variables
```yaml
environment:
  - WATCHPACK_POLLING=true
  - CHOKIDAR_USEPOLLING=true
```

#### 3. Python Module Import Errors

**Problem**: `ModuleNotFoundError: No module named 'models'` in math-engine
**Root Cause**: Incorrect import paths in containerized environment
**Solution Applied**: 
- Created local `models.py` file in math-engine
- Fixed import statements throughout the codebase
- Updated Dockerfile to properly copy shared dependencies

#### 4. Frontend Memory Issues

**Problem**: React development server running out of memory
**Root Cause**: Insufficient memory allocation for React builds
**Solution Applied**: Increased memory limit from 512M to 2G
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '0.5'
```

### Common Issues

1. **Port conflicts**: Ensure ports 3000, 8000, 8001, 5432, 6379 are available
2. **Permission issues**: Check file permissions and Docker daemon access
3. **Memory issues**: Increase Docker memory limits if needed (React needs 2GB+)
4. **Network issues**: Verify Docker network configuration and service connectivity

### Debug Commands

```bash
# Check container status
docker-compose ps

# View container logs
docker-compose logs [service]

# Execute commands in containers
docker-compose exec [service] /bin/bash

# Rebuild specific service
docker-compose build [service]

# Reset everything
make clean && make build && make up
```

### Performance Optimization

1. **Build cache**: Use Docker build cache effectively
2. **Layer optimization**: Minimize Docker layers
3. **Resource limits**: Set appropriate memory/CPU limits
4. **Volume optimization**: Use named volumes for better performance

## Testing

### Container Tests

```bash
# Run all container tests
make test-containers

# Run specific test categories
python -m pytest tests/docker/test_docker_setup.py::TestDockerConfiguration -v
python -m pytest tests/docker/test_docker_setup.py::TestDockerComposeConfiguration -v
python -m pytest tests/docker/test_docker_setup.py::TestDockerfileOptimization -v
```

### Integration Tests

The test suite verifies:

- Docker image builds successfully
- Container security practices
- Service dependencies
- Port configurations
- Environment variable setup
- Health check functionality

## Production Deployment

### Scaling

Production configuration supports horizontal scaling:

```yaml
deploy:
  replicas: 2
  resources:
    limits:
      memory: 1G
      cpus: '0.5'
```

### Load Balancing

Use nginx or a cloud load balancer for production:

```bash
# Example nginx configuration included
# See docker-compose.prod.yml nginx service
```

### Monitoring

Production setup includes:

- Health checks for all services
- Resource limits and monitoring
- Structured logging
- Performance metrics collection

## File Structure

```
.
├── docker-compose.yml          # Development configuration
├── docker-compose.prod.yml     # Production configuration
├── .env.example               # Environment template
├── .env.prod                  # Production environment template
├── .dockerignore              # Global Docker ignore
├── Makefile                   # Docker management commands
├── redis.conf                 # Redis configuration
├── math-engine/
│   ├── Dockerfile            # Math engine container
│   ├── .dockerignore         # Service-specific ignores
│   └── health_check.py       # Health check script
├── api-gateway/
│   ├── Dockerfile            # API gateway container
│   ├── .dockerignore         # Service-specific ignores
│   └── .air.toml            # Hot reload configuration
├── frontend/
│   ├── Dockerfile            # Frontend container
│   ├── .dockerignore         # Service-specific ignores
│   └── nginx.conf           # Production nginx config
└── tests/docker/
    ├── test_containers.py    # Full container tests
    └── test_docker_setup.py  # Configuration tests
```

This Docker setup provides a robust, scalable, and secure foundation for the AI Math Tutor application with support for both development and production environments.