#!/bin/bash

# AI Math Tutor Kubernetes Deployment Script
# This script deploys the AI Math Tutor application to a Kubernetes cluster

set -e

# Configuration
NAMESPACE="ai-math-tutor"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:5000}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if kubectl is installed
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install kubectl first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build and push Docker images
build_and_push_images() {
    log_info "Building and pushing Docker images..."
    
    # Build math engine image
    log_info "Building math-engine image..."
    docker build -t "${DOCKER_REGISTRY}/ai-math-tutor-math-engine:${IMAGE_TAG}" \
        --target production ./math-engine
    docker push "${DOCKER_REGISTRY}/ai-math-tutor-math-engine:${IMAGE_TAG}"
    
    # Build API gateway image
    log_info "Building api-gateway image..."
    docker build -t "${DOCKER_REGISTRY}/ai-math-tutor-api-gateway:${IMAGE_TAG}" \
        --target production ./api-gateway
    docker push "${DOCKER_REGISTRY}/ai-math-tutor-api-gateway:${IMAGE_TAG}"
    
    # Build frontend image
    log_info "Building frontend image..."
    docker build -t "${DOCKER_REGISTRY}/ai-math-tutor-frontend:${IMAGE_TAG}" \
        --target production ./frontend
    docker push "${DOCKER_REGISTRY}/ai-math-tutor-frontend:${IMAGE_TAG}"
    
    log_success "Docker images built and pushed successfully"
}

# Update image references in manifests
update_image_references() {
    log_info "Updating image references in manifests..."
    
    # Update math-engine image
    sed -i.bak "s|image: ai-math-tutor-math-engine:latest|image: ${DOCKER_REGISTRY}/ai-math-tutor-math-engine:${IMAGE_TAG}|g" k8s/math-engine.yaml
    
    # Update api-gateway image
    sed -i.bak "s|image: ai-math-tutor-api-gateway:latest|image: ${DOCKER_REGISTRY}/ai-math-tutor-api-gateway:${IMAGE_TAG}|g" k8s/api-gateway.yaml
    
    # Update frontend image
    sed -i.bak "s|image: ai-math-tutor-frontend:latest|image: ${DOCKER_REGISTRY}/ai-math-tutor-frontend:${IMAGE_TAG}|g" k8s/frontend.yaml
    
    log_success "Image references updated"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log_info "Deploying to Kubernetes cluster..."
    
    # Create namespace and basic resources
    log_info "Creating namespace and basic resources..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/storage.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/secrets.yaml
    
    # Deploy database and cache
    log_info "Deploying database and cache services..."
    kubectl apply -f k8s/postgres.yaml
    kubectl apply -f k8s/redis.yaml
    
    # Wait for database and cache to be ready
    log_info "Waiting for database and cache to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/postgres -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/redis -n ${NAMESPACE}
    
    # Deploy application services
    log_info "Deploying application services..."
    kubectl apply -f k8s/math-engine.yaml
    kubectl apply -f k8s/api-gateway.yaml
    kubectl apply -f k8s/frontend.yaml
    
    # Wait for application services to be ready
    log_info "Waiting for application services to be ready..."
    kubectl wait --for=condition=available --timeout=300s deployment/math-engine -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/api-gateway -n ${NAMESPACE}
    kubectl wait --for=condition=available --timeout=300s deployment/frontend -n ${NAMESPACE}
    
    # Deploy ingress and monitoring
    log_info "Deploying ingress and monitoring..."
    kubectl apply -f k8s/ingress.yaml
    kubectl apply -f k8s/monitoring.yaml
    
    log_success "Deployment completed successfully"
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check pod status
    log_info "Checking pod status..."
    kubectl get pods -n ${NAMESPACE}
    
    # Check service status
    log_info "Checking service status..."
    kubectl get services -n ${NAMESPACE}
    
    # Check ingress status
    log_info "Checking ingress status..."
    kubectl get ingress -n ${NAMESPACE}
    
    # Check HPA status
    log_info "Checking HPA status..."
    kubectl get hpa -n ${NAMESPACE}
    
    # Run health checks
    log_info "Running health checks..."
    
    # Check if all pods are running
    if kubectl get pods -n ${NAMESPACE} | grep -v Running | grep -v Completed | tail -n +2 | grep -q .; then
        log_warning "Some pods are not in Running state"
        kubectl get pods -n ${NAMESPACE} | grep -v Running | grep -v Completed
    else
        log_success "All pods are running"
    fi
    
    # Check if services are accessible
    log_info "Checking service accessibility..."
    
    # Port forward to test services (in background)
    kubectl port-forward -n ${NAMESPACE} service/api-gateway-service 8000:8000 &
    PF_PID=$!
    sleep 5
    
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "API Gateway health check passed"
    else
        log_warning "API Gateway health check failed"
    fi
    
    # Clean up port forward
    kill $PF_PID 2>/dev/null || true
    
    log_success "Deployment verification completed"
}

# Rollback deployment
rollback_deployment() {
    log_warning "Rolling back deployment..."
    
    # Rollback deployments
    kubectl rollout undo deployment/math-engine -n ${NAMESPACE}
    kubectl rollout undo deployment/api-gateway -n ${NAMESPACE}
    kubectl rollout undo deployment/frontend -n ${NAMESPACE}
    
    # Wait for rollback to complete
    kubectl rollout status deployment/math-engine -n ${NAMESPACE}
    kubectl rollout status deployment/api-gateway -n ${NAMESPACE}
    kubectl rollout status deployment/frontend -n ${NAMESPACE}
    
    log_success "Rollback completed"
}

# Cleanup deployment
cleanup_deployment() {
    log_warning "Cleaning up deployment..."
    
    # Delete all resources in namespace
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    
    log_success "Cleanup completed"
}

# Scale deployment
scale_deployment() {
    local replicas=${1:-3}
    log_info "Scaling deployment to ${replicas} replicas..."
    
    kubectl scale deployment/math-engine --replicas=${replicas} -n ${NAMESPACE}
    kubectl scale deployment/api-gateway --replicas=${replicas} -n ${NAMESPACE}
    kubectl scale deployment/frontend --replicas=${replicas} -n ${NAMESPACE}
    
    log_success "Scaling completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo "===================="
    
    echo -e "\n${BLUE}Pods:${NC}"
    kubectl get pods -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Services:${NC}"
    kubectl get services -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Ingress:${NC}"
    kubectl get ingress -n ${NAMESPACE}
    
    echo -e "\n${BLUE}HPA:${NC}"
    kubectl get hpa -n ${NAMESPACE}
    
    echo -e "\n${BLUE}Resource Usage:${NC}"
    kubectl top pods -n ${NAMESPACE} 2>/dev/null || echo "Metrics server not available"
}

# Main function
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            build_and_push_images
            update_image_references
            deploy_to_kubernetes
            verify_deployment
            ;;
        "verify")
            verify_deployment
            ;;
        "rollback")
            rollback_deployment
            ;;
        "cleanup")
            cleanup_deployment
            ;;
        "scale")
            scale_deployment $2
            ;;
        "status")
            show_status
            ;;
        "help"|"-h"|"--help")
            echo "AI Math Tutor Deployment Script"
            echo "Usage: $0 [command] [options]"
            echo ""
            echo "Commands:"
            echo "  deploy    - Deploy the application (default)"
            echo "  verify    - Verify the deployment"
            echo "  rollback  - Rollback to previous version"
            echo "  cleanup   - Remove all resources"
            echo "  scale N   - Scale to N replicas"
            echo "  status    - Show deployment status"
            echo "  help      - Show this help message"
            echo ""
            echo "Environment Variables:"
            echo "  DOCKER_REGISTRY - Docker registry URL (default: localhost:5000)"
            echo "  IMAGE_TAG       - Docker image tag (default: latest)"
            echo "  ENVIRONMENT     - Deployment environment (default: production)"
            ;;
        *)
            log_error "Unknown command: $1"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"