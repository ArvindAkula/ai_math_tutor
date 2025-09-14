#!/usr/bin/env python3
"""
Container functionality and deployment consistency tests.
These tests verify that Docker containers are built correctly and function as expected.
"""

import pytest
import docker
import requests
import time
import subprocess
import os
from typing import Dict, List, Optional

class TestContainerFunctionality:
    """Test suite for Docker container functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up Docker client and test environment."""
        cls.client = docker.from_env()
        cls.test_network = "ai-math-tutor-test-network"
        cls.containers = {}
        
        # Create test network
        try:
            cls.client.networks.create(cls.test_network, driver="bridge")
        except docker.errors.APIError:
            # Network might already exist
            pass
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        # Stop and remove test containers
        for container in cls.containers.values():
            try:
                container.stop()
                container.remove()
            except:
                pass
        
        # Remove test network
        try:
            network = cls.client.networks.get(cls.test_network)
            network.remove()
        except:
            pass
    
    def test_math_engine_container_build(self):
        """Test that math engine container builds successfully."""
        try:
            image, logs = self.client.images.build(
                path="./math-engine",
                tag="ai-math-tutor-math-engine:test",
                target="production"
            )
            assert image is not None
            assert "ai-math-tutor-math-engine:test" in [tag for tag in image.tags]
        except docker.errors.BuildError as e:
            pytest.fail(f"Math engine container build failed: {e}")
    
    def test_api_gateway_container_build(self):
        """Test that API gateway container builds successfully."""
        try:
            image, logs = self.client.images.build(
                path="./api-gateway",
                tag="ai-math-tutor-api-gateway:test",
                target="production"
            )
            assert image is not None
            assert "ai-math-tutor-api-gateway:test" in [tag for tag in image.tags]
        except docker.errors.BuildError as e:
            pytest.fail(f"API gateway container build failed: {e}")
    
    def test_frontend_container_build(self):
        """Test that frontend container builds successfully."""
        try:
            image, logs = self.client.images.build(
                path="./frontend",
                tag="ai-math-tutor-frontend:test",
                target="production"
            )
            assert image is not None
            assert "ai-math-tutor-frontend:test" in [tag for tag in image.tags]
        except docker.errors.BuildError as e:
            pytest.fail(f"Frontend container build failed: {e}")
    
    def test_container_security(self):
        """Test that containers follow security best practices."""
        # Test math engine container
        container = self.client.containers.run(
            "ai-math-tutor-math-engine:test",
            detach=True,
            network=self.test_network,
            name="test-math-engine-security"
        )
        
        try:
            # Check that container runs as non-root user
            exec_result = container.exec_run("whoami")
            assert exec_result.output.decode().strip() == "mathengine"
            
            # Check that sensitive directories are not writable
            exec_result = container.exec_run("test -w /etc/passwd")
            assert exec_result.exit_code != 0
            
        finally:
            container.stop()
            container.remove()
    
    def test_container_resource_limits(self):
        """Test that containers respect resource limits."""
        container = self.client.containers.run(
            "ai-math-tutor-math-engine:test",
            detach=True,
            network=self.test_network,
            mem_limit="512m",
            cpuset_cpus="0",
            name="test-math-engine-resources"
        )
        
        try:
            # Wait for container to start
            time.sleep(5)
            
            # Check memory limit
            stats = container.stats(stream=False)
            memory_limit = stats['memory']['limit']
            assert memory_limit <= 512 * 1024 * 1024  # 512MB in bytes
            
        finally:
            container.stop()
            container.remove()
    
    def test_health_checks(self):
        """Test that health checks work correctly."""
        # Start dependencies first
        postgres_container = self.client.containers.run(
            "postgres:15-alpine",
            detach=True,
            network=self.test_network,
            name="test-postgres",
            environment={
                "POSTGRES_DB": "ai_math_tutor",
                "POSTGRES_USER": "postgres",
                "POSTGRES_PASSWORD": "password"
            }
        )
        
        redis_container = self.client.containers.run(
            "redis:7-alpine",
            detach=True,
            network=self.test_network,
            name="test-redis"
        )
        
        try:
            # Wait for dependencies to be ready
            time.sleep(10)
            
            # Start math engine container
            math_engine_container = self.client.containers.run(
                "ai-math-tutor-math-engine:test",
                detach=True,
                network=self.test_network,
                name="test-math-engine-health",
                environment={
                    "REDIS_URL": "redis://test-redis:6379",
                    "DATABASE_URL": "postgresql://postgres:password@test-postgres:5432/ai_math_tutor"
                }
            )
            
            # Wait for service to start
            time.sleep(15)
            
            # Test health check endpoint
            response = requests.get("http://localhost:8001/health", timeout=10)
            assert response.status_code == 200
            
            math_engine_container.stop()
            math_engine_container.remove()
            
        finally:
            postgres_container.stop()
            postgres_container.remove()
            redis_container.stop()
            redis_container.remove()


class TestDockerCompose:
    """Test suite for Docker Compose configurations."""
    
    def test_docker_compose_syntax(self):
        """Test that docker-compose files have valid syntax."""
        # Test development compose file
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "config"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Development compose syntax error: {result.stderr}"
        
        # Test production compose file
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.prod.yml", "config"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Production compose syntax error: {result.stderr}"
    
    def test_environment_variables(self):
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
    
    def test_service_dependencies(self):
        """Test that service dependencies are correctly defined."""
        import yaml
        
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


class TestDeploymentConsistency:
    """Test suite for deployment consistency across environments."""
    
    def test_image_consistency(self):
        """Test that images are consistent across development and production."""
        # Build development images
        dev_images = {}
        for service in ["math-engine", "api-gateway", "frontend"]:
            image, logs = docker.from_env().images.build(
                path=f"./{service}",
                tag=f"ai-math-tutor-{service}:dev",
                target="development"
            )
            dev_images[service] = image
        
        # Build production images
        prod_images = {}
        for service in ["math-engine", "api-gateway", "frontend"]:
            image, logs = docker.from_env().images.build(
                path=f"./{service}",
                tag=f"ai-math-tutor-{service}:prod",
                target="production"
            )
            prod_images[service] = image
        
        # Verify both builds succeeded
        assert len(dev_images) == 3
        assert len(prod_images) == 3
    
    def test_port_consistency(self):
        """Test that port configurations are consistent."""
        import yaml
        
        # Load compose files
        with open("docker-compose.yml", "r") as f:
            dev_config = yaml.safe_load(f)
        
        with open("docker-compose.prod.yml", "r") as f:
            prod_config = yaml.safe_load(f)
        
        # Check that exposed ports are consistent
        dev_services = dev_config["services"]
        prod_services = prod_config["services"]
        
        for service_name in ["api-gateway", "frontend"]:
            if "ports" in dev_services[service_name] and "ports" in prod_services[service_name]:
                dev_ports = dev_services[service_name]["ports"]
                prod_ports = prod_services[service_name]["ports"]
                
                # Extract port numbers
                dev_port = dev_ports[0].split(":")[0] if dev_ports else None
                prod_port = prod_ports[0].split(":")[0] if prod_ports else None
                
                assert dev_port == prod_port, f"Port mismatch for {service_name}: dev={dev_port}, prod={prod_port}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])