# AI Math Tutor - Docker Management Makefile

.PHONY: help build build-dev build-prod up up-dev up-prod down clean test test-containers logs health-check

# Default target
help:
	@echo "AI Math Tutor Docker Management"
	@echo "================================"
	@echo ""
	@echo "Available targets:"
	@echo "  build         - Build all Docker images for development"
	@echo "  build-dev     - Build development images"
	@echo "  build-prod    - Build production images"
	@echo "  up            - Start development environment"
	@echo "  up-dev        - Start development environment (explicit)"
	@echo "  up-prod       - Start production environment"
	@echo "  down          - Stop and remove containers"
	@echo "  clean         - Remove all containers, images, and volumes"
	@echo "  test          - Run all tests"
	@echo "  test-containers - Run container-specific tests"
	@echo "  logs          - Show logs from all services"
	@echo "  health-check  - Check health of all services"
	@echo ""

# Build targets
build: build-dev

build-dev:
	@echo "Building development images..."
	docker-compose -f docker-compose.yml build

build-prod:
	@echo "Building production images..."
	docker-compose -f docker-compose.prod.yml build

# Start/stop targets
up: up-dev

up-dev:
	@echo "Starting development environment..."
	docker-compose -f docker-compose.yml up -d
	@echo "Services starting... Use 'make logs' to view logs"
	@echo "Services will be available at:"
	@echo "  Frontend: http://localhost:3000"
	@echo "  API Gateway: http://localhost:8000"
	@echo "  Math Engine: http://localhost:8001"

up-prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "Production services started"

down:
	@echo "Stopping all services..."
	docker-compose -f docker-compose.yml down
	docker-compose -f docker-compose.prod.yml down

# Cleanup targets
clean:
	@echo "Cleaning up Docker resources..."
	docker-compose -f docker-compose.yml down -v --rmi all
	docker-compose -f docker-compose.prod.yml down -v --rmi all
	docker system prune -f

# Testing targets
test:
	@echo "Running all tests..."
	python -m pytest tests/ -v

test-containers:
	@echo "Running container tests..."
	python -m pytest tests/docker/test_containers.py -v

# Monitoring targets
logs:
	@echo "Showing logs from all services..."
	docker-compose -f docker-compose.yml logs -f

health-check:
	@echo "Checking service health..."
	@echo "Math Engine Health:"
	@curl -f http://localhost:8001/health 2>/dev/null || echo "Math Engine: UNHEALTHY"
	@echo ""
	@echo "API Gateway Health:"
	@curl -f http://localhost:8000/health 2>/dev/null || echo "API Gateway: UNHEALTHY"
	@echo ""
	@echo "Frontend Health:"
	@curl -f http://localhost:3000/health 2>/dev/null || echo "Frontend: UNHEALTHY"

# Development helpers
dev-shell-math:
	@echo "Opening shell in math engine container..."
	docker-compose -f docker-compose.yml exec math-engine /bin/bash

dev-shell-api:
	@echo "Opening shell in API gateway container..."
	docker-compose -f docker-compose.yml exec api-gateway /bin/sh

dev-shell-frontend:
	@echo "Opening shell in frontend container..."
	docker-compose -f docker-compose.yml exec frontend /bin/sh

# Database helpers
db-migrate:
	@echo "Running database migrations..."
	docker-compose -f docker-compose.yml exec postgres psql -U postgres -d ai_math_tutor -f /docker-entrypoint-initdb.d/01-init.sql

db-shell:
	@echo "Opening database shell..."
	docker-compose -f docker-compose.yml exec postgres psql -U postgres -d ai_math_tutor

# Redis helpers
redis-shell:
	@echo "Opening Redis shell..."
	docker-compose -f docker-compose.yml exec redis redis-cli

# Kubernetes deployment targets
k8s-deploy:
	@echo "Deploying to Kubernetes..."
	./deploy.sh deploy

k8s-verify:
	@echo "Verifying Kubernetes deployment..."
	./deploy.sh verify

k8s-status:
	@echo "Showing Kubernetes deployment status..."
	./deploy.sh status

k8s-scale:
	@echo "Scaling Kubernetes deployment..."
	./deploy.sh scale $(REPLICAS)

k8s-rollback:
	@echo "Rolling back Kubernetes deployment..."
	./deploy.sh rollback

k8s-cleanup:
	@echo "Cleaning up Kubernetes deployment..."
	./deploy.sh cleanup

# Test Kubernetes manifests
test-k8s:
	@echo "Testing Kubernetes manifests..."
	python -m pytest tests/k8s/ -v