# AI Math Tutor - Complete Setup Guide

This guide provides comprehensive instructions for setting up and deploying the AI Math Tutor application in both development and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Production Deployment](#production-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Monitoring and Maintenance](#monitoring-and-maintenance)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: macOS, Linux, or Windows with WSL2
- **Memory**: Minimum 8GB RAM (16GB recommended for development)
- **Storage**: At least 20GB free space
- **Network**: Internet connection for downloading dependencies

### Required Software

#### Development Environment

```bash
# Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Docker Compose (if not included with Docker)
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Python 3.11+
sudo apt-get update
sudo apt-get install python3.11 python3.11-pip python3.11-venv

# Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Go 1.21+
wget https://go.dev/dl/go1.21.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.21.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc
```

#### Production Environment

```bash
# Kubernetes cluster (choose one)
# - Google Kubernetes Engine (GKE)
# - Amazon Elastic Kubernetes Service (EKS)
# - Azure Kubernetes Service (AKS)
# - Self-managed cluster

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Helm (optional, for package management)
curl https://baltocdn.com/helm/signing.asc | gpg --dearmor | sudo tee /usr/share/keyrings/helm.gpg > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/helm.gpg] https://baltocdn.com/helm/stable/debian/ all main" | sudo tee /etc/apt/sources.list.d/helm-stable-debian.list
sudo apt-get update
sudo apt-get install helm
```

## Development Setup

### 1. Clone and Setup Repository

```bash
# Clone the repository
git clone <repository-url>
cd ai-math-tutor

# Copy environment configuration
cp .env.example .env

# Edit .env with your configuration
nano .env
```

### 2. Configure Environment Variables

Edit `.env` file with your settings:

```bash
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_DB=ai_math_tutor

# External API Keys (Optional but recommended for AI explanations)
OPENAI_API_KEY=your_openai_api_key_here

# JWT Configuration (REQUIRED for authentication)
JWT_SECRET=your_super_secret_jwt_key_change_in_production

# Development settings
NODE_ENV=development
LOG_LEVEL=INFO
ENVIRONMENT=development

# React Development (fixes hot reloading issues in Docker)
WATCHPACK_POLLING=true
CHOKIDAR_USEPOLLING=true
```

### 3. Start Development Environment

```bash
# Using Make (recommended)
make up

# Or using Docker Compose directly
docker-compose up -d

# Check service status
make health-check

# View logs
make logs
```

### 4. Access Development Services

- **Frontend**: http://localhost:3000
- **API Gateway**: http://localhost:8000
- **Math Engine**: http://localhost:8001
- **Database**: localhost:5432
- **Redis**: localhost:6379

### 5. Development Workflow

```bash
# Run tests
make test

# Run container tests
make test-containers

# Rebuild specific service
docker-compose build math-engine
docker-compose up -d math-engine

# Access service shells
make dev-shell-math      # Math engine shell
make dev-shell-api       # API gateway shell
make dev-shell-frontend  # Frontend shell

# Database operations
make db-shell           # PostgreSQL shell
make redis-shell        # Redis shell
```

## Production Deployment

### 1. Production Environment Setup

```bash
# Copy production environment template
cp .env.prod .env.production

# Edit production configuration
nano .env.production
```

Configure production environment variables:

```bash
# Database Configuration (use strong passwords)
POSTGRES_USER=ai_math_tutor_user
POSTGRES_PASSWORD=STRONG_PRODUCTION_PASSWORD
POSTGRES_DB=ai_math_tutor

# Security (generate strong secrets)
JWT_SECRET=$(openssl rand -base64 32)

# External APIs
OPENAI_API_KEY=your_production_openai_key

# Frontend URLs
REACT_APP_API_URL=https://api.yourdomain.com
REACT_APP_WS_URL=wss://api.yourdomain.com

# Production settings
NODE_ENV=production
LOG_LEVEL=INFO
ENVIRONMENT=production
```

### 2. Docker Production Deployment

```bash
# Build production images
make build-prod

# Start production environment
make up-prod

# Verify deployment
docker-compose -f docker-compose.prod.yml ps
```

## Kubernetes Deployment

### 1. Prepare Kubernetes Cluster

#### Google Kubernetes Engine (GKE)

```bash
# Create GKE cluster
gcloud container clusters create ai-math-tutor \
    --zone=us-central1-a \
    --num-nodes=3 \
    --machine-type=e2-standard-4 \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=10

# Get credentials
gcloud container clusters get-credentials ai-math-tutor --zone=us-central1-a
```

#### Amazon EKS

```bash
# Create EKS cluster
eksctl create cluster \
    --name ai-math-tutor \
    --region us-west-2 \
    --nodegroup-name standard-workers \
    --node-type m5.large \
    --nodes 3 \
    --nodes-min 1 \
    --nodes-max 10 \
    --managed
```

#### Azure AKS

```bash
# Create AKS cluster
az aks create \
    --resource-group myResourceGroup \
    --name ai-math-tutor \
    --node-count 3 \
    --node-vm-size Standard_D2s_v3 \
    --enable-cluster-autoscaler \
    --min-count 1 \
    --max-count 10

# Get credentials
az aks get-credentials --resource-group myResourceGroup --name ai-math-tutor
```

### 2. Configure Kubernetes Secrets

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Update secrets with base64 encoded values
echo -n "your_postgres_password" | base64
echo -n "your_jwt_secret" | base64
echo -n "your_openai_api_key" | base64

# Edit secrets file
nano k8s/secrets.yaml

# Apply secrets
kubectl apply -f k8s/secrets.yaml
```

### 3. Deploy to Kubernetes

```bash
# Using deployment script (recommended)
./deploy.sh deploy

# Or step by step
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/math-engine.yaml
kubectl apply -f k8s/api-gateway.yaml
kubectl apply -f k8s/frontend.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/monitoring.yaml

# Verify deployment
./deploy.sh verify
```

### 4. Configure Domain and SSL

```bash
# Update ingress with your domain
nano k8s/ingress.yaml

# Install cert-manager for SSL certificates
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.12.0/cert-manager.yaml

# Create cluster issuer for Let's Encrypt
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: your-email@domain.com
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF

# Apply updated ingress
kubectl apply -f k8s/ingress.yaml
```

### 5. Kubernetes Management Commands

```bash
# Deployment management
make k8s-deploy          # Deploy to Kubernetes
make k8s-verify          # Verify deployment
make k8s-status          # Show deployment status
make k8s-scale REPLICAS=5 # Scale to 5 replicas
make k8s-rollback        # Rollback deployment
make k8s-cleanup         # Clean up deployment

# Direct kubectl commands
kubectl get pods -n ai-math-tutor
kubectl get services -n ai-math-tutor
kubectl get ingress -n ai-math-tutor
kubectl logs -f deployment/math-engine -n ai-math-tutor
kubectl describe pod <pod-name> -n ai-math-tutor
```

## Monitoring and Maintenance

### 1. Application Monitoring

```bash
# Check application health
curl https://api.yourdomain.com/health
curl https://yourdomain.com/health

# View application logs
kubectl logs -f deployment/api-gateway -n ai-math-tutor
kubectl logs -f deployment/math-engine -n ai-math-tutor

# Monitor resource usage
kubectl top pods -n ai-math-tutor
kubectl top nodes
```

### 2. Database Maintenance

```bash
# Database backup
kubectl exec -it deployment/postgres -n ai-math-tutor -- pg_dump -U postgres ai_math_tutor > backup.sql

# Database restore
kubectl exec -i deployment/postgres -n ai-math-tutor -- psql -U postgres ai_math_tutor < backup.sql

# Database shell access
kubectl exec -it deployment/postgres -n ai-math-tutor -- psql -U postgres -d ai_math_tutor
```

### 3. Scaling Operations

```bash
# Manual scaling
kubectl scale deployment/math-engine --replicas=5 -n ai-math-tutor

# Check HPA status
kubectl get hpa -n ai-math-tutor

# View HPA events
kubectl describe hpa math-engine-hpa -n ai-math-tutor
```

### 4. Updates and Rollbacks

```bash
# Update application
docker build -t your-registry/ai-math-tutor-math-engine:v2.0 ./math-engine
docker push your-registry/ai-math-tutor-math-engine:v2.0
kubectl set image deployment/math-engine math-engine=your-registry/ai-math-tutor-math-engine:v2.0 -n ai-math-tutor

# Check rollout status
kubectl rollout status deployment/math-engine -n ai-math-tutor

# Rollback if needed
kubectl rollout undo deployment/math-engine -n ai-math-tutor
```

## Troubleshooting

### Authentication Issues (Recently Fixed)

#### Login/Registration Failures

**Symptoms**: "Failed to register user" or "Failed to login" errors

**Root Causes and Solutions**:

1. **Database Schema Mismatch**:
   ```bash
   # Add missing columns to users table
   docker exec -it ai-math-tutor-postgres psql -U postgres -d ai_math_tutor -c "
   ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(50) DEFAULT 'student';
   ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP;
   "
   
   # Fix user_sessions table structure
   docker exec -it ai-math-tutor-postgres psql -U postgres -d ai_math_tutor -c "
   ALTER TABLE user_sessions RENAME COLUMN session_token TO refresh_token;
   ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS is_revoked BOOLEAN DEFAULT FALSE;
   ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS user_agent VARCHAR(255);
   ALTER TABLE user_sessions ADD COLUMN IF NOT EXISTS ip_address VARCHAR(45);
   ALTER TABLE user_sessions ALTER COLUMN refresh_token TYPE TEXT;
   "
   ```

2. **JWT Token Size Issues**:
   ```bash
   # Increase refresh_token column size to handle JWT tokens
   docker exec -it ai-math-tutor-postgres psql -U postgres -d ai_math_tutor -c "
   ALTER TABLE user_sessions ALTER COLUMN refresh_token TYPE TEXT;
   "
   ```

3. **Frontend API Integration**:
   - Ensure proxy configuration points to correct API gateway
   - Update Login/Register components to handle correct API response format
   - Verify CORS configuration allows frontend requests

#### React Hot Reloading Issues

**Symptoms**: Constant page refreshing in development

**Solution**: Add polling environment variables to docker-compose.yml:
```yaml
environment:
  - WATCHPACK_POLLING=true
  - CHOKIDAR_USEPOLLING=true
```

### Common Issues

#### 1. Container Won't Start

```bash
# Check pod status
kubectl describe pod <pod-name> -n ai-math-tutor

# Check logs
kubectl logs <pod-name> -n ai-math-tutor

# Common fixes
- Check resource limits
- Verify environment variables
- Check image availability
- Verify secrets and configmaps
```

#### 2. Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n ai-math-tutor

# Test service connectivity
kubectl run test-pod --image=busybox -it --rm -- /bin/sh
# Inside pod: wget -qO- http://api-gateway-service:8000/health

# Check ingress
kubectl describe ingress ai-math-tutor-ingress -n ai-math-tutor
```

#### 3. Database Connection Issues

```bash
# Check database pod
kubectl logs deployment/postgres -n ai-math-tutor

# Test database connectivity
kubectl exec -it deployment/postgres -n ai-math-tutor -- pg_isready -U postgres

# Check database service
kubectl get service postgres-service -n ai-math-tutor
```

#### 4. Performance Issues

```bash
# Check resource usage
kubectl top pods -n ai-math-tutor
kubectl top nodes

# Check HPA status
kubectl get hpa -n ai-math-tutor

# Scale manually if needed
kubectl scale deployment/math-engine --replicas=10 -n ai-math-tutor
```

### Debugging Commands

```bash
# Get cluster info
kubectl cluster-info

# Check node status
kubectl get nodes

# Check all resources in namespace
kubectl get all -n ai-math-tutor

# Describe problematic resources
kubectl describe pod <pod-name> -n ai-math-tutor
kubectl describe service <service-name> -n ai-math-tutor

# Check events
kubectl get events -n ai-math-tutor --sort-by='.lastTimestamp'

# Port forward for local testing
kubectl port-forward service/api-gateway-service 8000:8000 -n ai-math-tutor
```

### Performance Tuning

#### Resource Optimization

```yaml
# Adjust resource requests and limits
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "500m"
```

#### Auto-scaling Tuning

```yaml
# Adjust HPA thresholds
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 60  # Lower for more aggressive scaling
```

#### Database Performance

```bash
# Monitor database performance
kubectl exec -it deployment/postgres -n ai-math-tutor -- psql -U postgres -d ai_math_tutor -c "SELECT * FROM pg_stat_activity;"

# Optimize database queries
kubectl exec -it deployment/postgres -n ai-math-tutor -- psql -U postgres -d ai_math_tutor -f /docker-entrypoint-initdb.d/02-optimize.sql
```

## Security Best Practices

### 1. Secrets Management

- Use Kubernetes secrets for sensitive data
- Rotate secrets regularly
- Use external secret management systems (AWS Secrets Manager, Azure Key Vault, etc.)

### 2. Network Security

- Implement network policies
- Use TLS for all communications
- Restrict ingress to necessary ports only

### 3. Container Security

- Run containers as non-root users
- Use read-only root filesystems
- Scan images for vulnerabilities
- Keep base images updated

### 4. Access Control

- Implement RBAC (Role-Based Access Control)
- Use service accounts with minimal permissions
- Enable audit logging

This setup guide provides a comprehensive foundation for deploying and maintaining the AI Math Tutor application in both development and production environments.