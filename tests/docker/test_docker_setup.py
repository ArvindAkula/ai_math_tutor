#!/usr/bin/env python3
"""
Basic Docker setup tests that don't require the docker Python package.
These tests verify Docker configuration files and basic setup.
"""

import pytest
import subprocess
import os
import yaml
from pathlib import Path

class TestDockerConfiguration:
    """Test Docker configuration files."""
    
    def test_dockerfiles_exist(self):
        """Test that all required Dockerfiles exist."""
        dockerfiles = [
            "math-engine/Dockerfile",
            "api-gateway/Dockerfile", 
            "frontend/Dockerfile"
        ]
        
        for dockerfile in dockerfiles:
            assert Path(dockerfile).exists(), f"Dockerfile not found: {dockerfile}"
    
    def test_dockerignore_files_exist(self):
        """Test that .dockerignore files exist for optimization."""
        dockerignore_files = [
            ".dockerignore",
            "math-engine/.dockerignore",
            "api-gateway/.dockerignore",
            "frontend/.dockerignore"
        ]
        
        for dockerignore in dockerignore_files:
            assert Path(dockerignore).exists(), f".dockerignore not found: {dockerignore}"
    
    def test_docker_compose_files_exist(self):
        """Test that docker-compose files exist."""
        compose_files = [
            "docker-compose.yml",
            "docker-compose.prod.yml"
        ]
        
        for compose_file in compose_files:
            assert Path(compose_file).exists(), f"Compose file not found: {compose_file}"
    
    def test_docker_compose_syntax(self):
        """Test that docker-compose files have valid syntax."""
        # Test development compose file
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "config", "--quiet"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Development compose syntax error: {result.stderr}"
        
        # Test production compose file with env file
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.prod.yml", "--env-file", ".env.prod", "config", "--quiet"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Production compose syntax error: {result.stderr}"
    
    def test_environment_files_exist(self):
        """Test that environment files exist."""
        env_files = [
            ".env.example",
            ".env.prod"
        ]
        
        for env_file in env_files:
            assert Path(env_file).exists(), f"Environment file not found: {env_file}"
    
    def test_required_environment_variables(self):
        """Test that required environment variables are defined."""
        required_vars = [
            "POSTGRES_USER",
            "POSTGRES_PASSWORD", 
            "JWT_SECRET",
            "OPENAI_API_KEY"
        ]
        
        # Check .env.example
        with open(".env.example", "r") as f:
            env_content = f.read()
            for var in required_vars:
                assert var in env_content, f"Required variable {var} not found in .env.example"
        
        # Check .env.prod
        with open(".env.prod", "r") as f:
            env_content = f.read()
            for var in required_vars:
                assert var in env_content, f"Required variable {var} not found in .env.prod"


class TestDockerComposeConfiguration:
    """Test Docker Compose configuration details."""
    
    def test_service_dependencies(self):
        """Test that service dependencies are correctly defined."""
        with open("docker-compose.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config["services"]
        
        # Math engine should depend on postgres and redis
        math_engine_deps = services["math-engine"]["depends_on"]
        assert "postgres" in math_engine_deps
        assert "redis" in math_engine_deps
        
        # API gateway should depend on postgres, redis, and math-engine
        api_gateway_deps = services["api-gateway"]["depends_on"]
        assert "postgres" in api_gateway_deps
        assert "redis" in api_gateway_deps
        assert "math-engine" in api_gateway_deps
        
        # Frontend should depend on api-gateway
        frontend_deps = services["frontend"]["depends_on"]
        assert "api-gateway" in frontend_deps
    
    def test_port_configurations(self):
        """Test that ports are properly configured."""
        with open("docker-compose.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        services = compose_config["services"]
        
        # Check expected ports
        expected_ports = {
            "postgres": ["5432:5432"],
            "redis": ["6379:6379"],
            "math-engine": ["8001:8001"],
            "api-gateway": ["8000:8000"],
            "frontend": ["3000:3000"]
        }
        
        for service, expected in expected_ports.items():
            if "ports" in services[service]:
                assert services[service]["ports"] == expected, f"Port mismatch for {service}"
    
    def test_volume_configurations(self):
        """Test that volumes are properly configured."""
        with open("docker-compose.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        # Check that persistent volumes are defined
        volumes = compose_config.get("volumes", {})
        expected_volumes = ["postgres_data", "redis_data", "math_engine_cache", "frontend_node_modules"]
        
        for volume in expected_volumes:
            assert volume in volumes, f"Volume {volume} not defined"
    
    def test_network_configuration(self):
        """Test that network is properly configured."""
        with open("docker-compose.yml", "r") as f:
            compose_config = yaml.safe_load(f)
        
        # Check that network is defined
        networks = compose_config.get("networks", {})
        assert "ai-math-tutor-network" in networks, "Network ai-math-tutor-network not defined"
        
        # Check that services use the network
        services = compose_config["services"]
        for service_name, service_config in services.items():
            if "networks" in service_config:
                assert "ai-math-tutor-network" in service_config["networks"], f"Service {service_name} not using correct network"


class TestDockerfileOptimization:
    """Test that Dockerfiles are optimized."""
    
    def test_multi_stage_builds(self):
        """Test that Dockerfiles use multi-stage builds."""
        dockerfiles = [
            "math-engine/Dockerfile",
            "api-gateway/Dockerfile",
            "frontend/Dockerfile"
        ]
        
        for dockerfile in dockerfiles:
            with open(dockerfile, "r") as f:
                content = f.read()
                # Check for multi-stage build indicators
                assert "FROM" in content, f"No FROM instruction in {dockerfile}"
                assert "as " in content.lower(), f"No multi-stage build in {dockerfile}"
    
    def test_health_checks_defined(self):
        """Test that health checks are defined where appropriate."""
        # Check math engine Dockerfile for health check
        with open("math-engine/Dockerfile", "r") as f:
            content = f.read()
            assert "HEALTHCHECK" in content, "No health check in math-engine Dockerfile"
    
    def test_security_practices(self):
        """Test that Dockerfiles follow security best practices."""
        dockerfiles = [
            "math-engine/Dockerfile",
            "api-gateway/Dockerfile"
        ]
        
        for dockerfile in dockerfiles:
            with open(dockerfile, "r") as f:
                content = f.read()
                # Check for non-root user
                assert "USER" in content, f"No non-root user defined in {dockerfile}"


class TestMakefileConfiguration:
    """Test Makefile for Docker operations."""
    
    def test_makefile_exists(self):
        """Test that Makefile exists."""
        assert Path("Makefile").exists(), "Makefile not found"
    
    def test_makefile_targets(self):
        """Test that required Makefile targets exist."""
        with open("Makefile", "r") as f:
            content = f.read()
        
        required_targets = [
            "build", "build-dev", "build-prod",
            "up", "up-dev", "up-prod", 
            "down", "clean",
            "test", "test-containers",
            "logs", "health-check"
        ]
        
        for target in required_targets:
            assert f"{target}:" in content, f"Makefile target {target} not found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])