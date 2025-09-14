#!/usr/bin/env python3
"""
Kubernetes deployment reliability and scaling tests.
These tests verify that Kubernetes manifests are valid and deployments work correctly.
"""

import pytest
import subprocess
import yaml
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

class TestKubernetesManifests:
    """Test Kubernetes manifest files."""
    
    def test_manifest_files_exist(self):
        """Test that all required manifest files exist."""
        required_manifests = [
            "k8s/namespace.yaml",
            "k8s/configmap.yaml",
            "k8s/secrets.yaml",
            "k8s/storage.yaml",
            "k8s/postgres.yaml",
            "k8s/redis.yaml",
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml",
            "k8s/ingress.yaml",
            "k8s/monitoring.yaml"
        ]
        
        for manifest in required_manifests:
            assert Path(manifest).exists(), f"Manifest file not found: {manifest}"
    
    def test_manifest_yaml_syntax(self):
        """Test that all manifest files have valid YAML syntax."""
        manifest_files = list(Path("k8s").glob("*.yaml"))
        
        for manifest_file in manifest_files:
            with open(manifest_file, "r") as f:
                try:
                    yaml.safe_load_all(f)
                except yaml.YAMLError as e:
                    pytest.fail(f"Invalid YAML syntax in {manifest_file}: {e}")
    
    def test_kubernetes_manifest_validation(self):
        """Test that Kubernetes manifests are valid using kubectl dry-run."""
        # Check if kubectl is available and can connect to cluster
        kubectl_check = subprocess.run(
            ["kubectl", "cluster-info"],
            capture_output=True,
            text=True
        )
        
        if kubectl_check.returncode != 0:
            pytest.skip("kubectl not available or no cluster connection for validation")
        
        manifest_files = list(Path("k8s").glob("*.yaml"))
        
        for manifest_file in manifest_files:
            result = subprocess.run(
                ["kubectl", "apply", "--dry-run=client", "-f", str(manifest_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Skip if kubectl is not available
                if "command not found" in result.stderr or "not found" in result.stderr or "connection refused" in result.stderr:
                    pytest.skip("kubectl not available for validation")
                else:
                    pytest.fail(f"Invalid Kubernetes manifest {manifest_file}: {result.stderr}")


class TestDeploymentConfiguration:
    """Test deployment configuration details."""
    
    def test_resource_limits_defined(self):
        """Test that resource limits are defined for all deployments."""
        deployment_files = [
            "k8s/postgres.yaml",
            "k8s/redis.yaml", 
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml"
        ]
        
        for deployment_file in deployment_files:
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        containers = doc["spec"]["template"]["spec"]["containers"]
                        
                        for container in containers:
                            assert "resources" in container, f"No resources defined in {deployment_file}"
                            assert "limits" in container["resources"], f"No resource limits in {deployment_file}"
                            assert "requests" in container["resources"], f"No resource requests in {deployment_file}"
    
    def test_health_checks_defined(self):
        """Test that health checks are defined for application containers."""
        app_deployment_files = [
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml"
        ]
        
        for deployment_file in app_deployment_files:
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        containers = doc["spec"]["template"]["spec"]["containers"]
                        
                        for container in containers:
                            assert "livenessProbe" in container, f"No liveness probe in {deployment_file}"
                            assert "readinessProbe" in container, f"No readiness probe in {deployment_file}"
    
    def test_security_context_defined(self):
        """Test that security contexts are defined for application containers."""
        app_deployment_files = [
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml"
        ]
        
        for deployment_file in app_deployment_files:
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        containers = doc["spec"]["template"]["spec"]["containers"]
                        
                        for container in containers:
                            assert "securityContext" in container, f"No security context in {deployment_file}"
                            security_context = container["securityContext"]
                            assert security_context.get("runAsNonRoot") == True, f"Not running as non-root in {deployment_file}"
                            assert security_context.get("readOnlyRootFilesystem") == True, f"Root filesystem not read-only in {deployment_file}"


class TestAutoScalingConfiguration:
    """Test auto-scaling configuration."""
    
    def test_hpa_defined_for_services(self):
        """Test that HPA is defined for scalable services."""
        scalable_services = ["math-engine", "api-gateway", "frontend"]
        
        for service in scalable_services:
            deployment_file = f"k8s/{service}.yaml"
            
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                hpa_found = False
                for doc in docs:
                    if doc and doc.get("kind") == "HorizontalPodAutoscaler":
                        hpa_found = True
                        
                        # Verify HPA configuration
                        assert "scaleTargetRef" in doc["spec"], f"No scaleTargetRef in HPA for {service}"
                        assert "minReplicas" in doc["spec"], f"No minReplicas in HPA for {service}"
                        assert "maxReplicas" in doc["spec"], f"No maxReplicas in HPA for {service}"
                        assert "metrics" in doc["spec"], f"No metrics in HPA for {service}"
                        
                        # Verify scaling behavior
                        if "behavior" in doc["spec"]:
                            behavior = doc["spec"]["behavior"]
                            assert "scaleUp" in behavior or "scaleDown" in behavior, f"No scaling behavior defined for {service}"
                
                assert hpa_found, f"No HPA found for {service}"
    
    def test_hpa_metrics_configuration(self):
        """Test that HPA metrics are properly configured."""
        hpa_files = [
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml", 
            "k8s/frontend.yaml"
        ]
        
        for hpa_file in hpa_files:
            with open(hpa_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                for doc in docs:
                    if doc and doc.get("kind") == "HorizontalPodAutoscaler":
                        metrics = doc["spec"]["metrics"]
                        
                        # Should have CPU and memory metrics
                        metric_types = [metric["resource"]["name"] for metric in metrics if metric["type"] == "Resource"]
                        assert "cpu" in metric_types, f"No CPU metric in {hpa_file}"
                        assert "memory" in metric_types, f"No memory metric in {hpa_file}"


class TestServiceDiscovery:
    """Test service discovery configuration."""
    
    def test_services_defined_for_deployments(self):
        """Test that services are defined for all deployments."""
        deployment_files = [
            "k8s/postgres.yaml",
            "k8s/redis.yaml",
            "k8s/math-engine.yaml", 
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml"
        ]
        
        for deployment_file in deployment_files:
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                deployment_found = False
                service_found = False
                
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        deployment_found = True
                    elif doc and doc.get("kind") == "Service":
                        service_found = True
                
                if deployment_found:
                    assert service_found, f"No service found for deployment in {deployment_file}"
    
    def test_service_selectors_match_deployments(self):
        """Test that service selectors match deployment labels."""
        deployment_files = [
            "k8s/postgres.yaml",
            "k8s/redis.yaml", 
            "k8s/math-engine.yaml",
            "k8s/api-gateway.yaml",
            "k8s/frontend.yaml"
        ]
        
        for deployment_file in deployment_files:
            with open(deployment_file, "r") as f:
                docs = list(yaml.safe_load_all(f))
                
                deployment_labels = None
                service_selector = None
                
                for doc in docs:
                    if doc and doc.get("kind") == "Deployment":
                        deployment_labels = doc["spec"]["template"]["metadata"]["labels"]
                    elif doc and doc.get("kind") == "Service":
                        service_selector = doc["spec"]["selector"]
                
                if deployment_labels and service_selector:
                    for key, value in service_selector.items():
                        assert key in deployment_labels, f"Service selector key {key} not in deployment labels in {deployment_file}"
                        assert deployment_labels[key] == value, f"Service selector value mismatch in {deployment_file}"


class TestLoadBalancing:
    """Test load balancing configuration."""
    
    def test_ingress_configuration(self):
        """Test that ingress is properly configured."""
        with open("k8s/ingress.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))
            
            ingress_found = False
            for doc in docs:
                if doc and doc.get("kind") == "Ingress":
                    ingress_found = True
                    
                    # Verify ingress configuration
                    assert "spec" in doc, "No spec in ingress"
                    assert "rules" in doc["spec"], "No rules in ingress spec"
                    assert "tls" in doc["spec"], "No TLS configuration in ingress"
                    
                    # Verify annotations for load balancing
                    annotations = doc["metadata"].get("annotations", {})
                    assert "kubernetes.io/ingress.class" in annotations, "No ingress class annotation"
                    assert "nginx.ingress.kubernetes.io/ssl-redirect" in annotations, "No SSL redirect annotation"
            
            assert ingress_found, "No ingress configuration found"
    
    def test_network_policy_defined(self):
        """Test that network policies are defined for security."""
        with open("k8s/ingress.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))
            
            network_policy_found = False
            for doc in docs:
                if doc and doc.get("kind") == "NetworkPolicy":
                    network_policy_found = True
                    
                    # Verify network policy configuration
                    assert "spec" in doc, "No spec in network policy"
                    assert "policyTypes" in doc["spec"], "No policy types in network policy"
                    assert "ingress" in doc["spec"], "No ingress rules in network policy"
                    assert "egress" in doc["spec"], "No egress rules in network policy"
            
            assert network_policy_found, "No network policy found"


class TestMonitoringConfiguration:
    """Test monitoring and alerting configuration."""
    
    def test_service_monitors_defined(self):
        """Test that service monitors are defined for metrics collection."""
        with open("k8s/monitoring.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))
            
            service_monitors = []
            for doc in docs:
                if doc and doc.get("kind") == "ServiceMonitor":
                    service_monitors.append(doc)
            
            assert len(service_monitors) > 0, "No service monitors found"
            
            # Verify service monitor configuration
            for monitor in service_monitors:
                assert "spec" in monitor, "No spec in service monitor"
                assert "selector" in monitor["spec"], "No selector in service monitor"
                assert "endpoints" in monitor["spec"], "No endpoints in service monitor"
    
    def test_prometheus_rules_defined(self):
        """Test that Prometheus alerting rules are defined."""
        with open("k8s/monitoring.yaml", "r") as f:
            docs = list(yaml.safe_load_all(f))
            
            prometheus_rule_found = False
            for doc in docs:
                if doc and doc.get("kind") == "PrometheusRule":
                    prometheus_rule_found = True
                    
                    # Verify prometheus rule configuration
                    assert "spec" in doc, "No spec in prometheus rule"
                    assert "groups" in doc["spec"], "No groups in prometheus rule"
                    
                    for group in doc["spec"]["groups"]:
                        assert "rules" in group, "No rules in prometheus rule group"
                        
                        for rule in group["rules"]:
                            if "alert" in rule:
                                assert "expr" in rule, "No expression in alert rule"
                                assert "labels" in rule, "No labels in alert rule"
                                assert "annotations" in rule, "No annotations in alert rule"
            
            assert prometheus_rule_found, "No prometheus rules found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])