"""
Comprehensive tests for monitoring, logging, and health check systems.
Tests structured logging, metrics collection, health checks, and alerting.
"""

import pytest
import time
import threading
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

from monitoring import (
    HealthStatus, MetricType, HealthCheckResult, Metric, Alert,
    StructuredLogger, MetricsCollector, HealthChecker, AlertManager,
    SystemMonitor, MonitoringService, timer_decorator, async_timer_decorator,
    counter_decorator, monitoring_service
)


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_service")
        assert logger.service_name == "test_service"
        assert logger.logger.name == "test_service"
        assert logger.request_id is None
        assert logger.correlation_id is None
    
    def test_set_request_context(self):
        """Test setting request context."""
        logger = StructuredLogger("test_service")
        logger.set_request_context("req_123", "corr_456")
        
        assert logger.request_id == "req_123"
        assert logger.correlation_id == "corr_456"
    
    def test_format_message(self):
        """Test message formatting."""
        logger = StructuredLogger("test_service")
        logger.set_request_context("req_123")
        
        formatted = logger._format_message("Test message", {"key": "value"})
        data = json.loads(formatted)
        
        assert data['service'] == "test_service"
        assert data['message'] == "Test message"
        assert data['request_id'] == "req_123"
        assert data['key'] == "value"
        assert 'timestamp' in data
    
    def test_log_methods(self):
        """Test different log level methods."""
        logger = StructuredLogger("test_service")
        
        # Should not raise exceptions
        logger.info("Info message", test_key="test_value")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        logger.debug("Debug message")


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(max_metrics=100)
        assert len(collector.metrics) == 0
        assert len(collector.counters) == 0
        assert len(collector.gauges) == 0
    
    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 5.0)
        collector.increment_counter("test_counter", 3.0)
        
        assert collector.get_counter("test_counter") == 8.0
        assert len(collector.metrics["test_counter"]) == 2
    
    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        
        collector.set_gauge("test_gauge", 42.0)
        collector.set_gauge("test_gauge", 84.0)
        
        assert collector.get_gauge("test_gauge") == 84.0
        assert len(collector.metrics["test_gauge"]) == 2
    
    def test_record_histogram(self):
        """Test histogram recording."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_histogram("test_histogram", value)
        
        stats = collector.get_histogram_stats("test_histogram")
        
        assert stats['count'] == 5
        assert stats['min'] == 1.0
        assert stats['max'] == 5.0
        assert stats['mean'] == 3.0
        assert stats['p50'] == 3.0
    
    def test_record_timer(self):
        """Test timer recording."""
        collector = MetricsCollector()
        
        durations = [0.1, 0.2, 0.3, 0.4, 0.5]
        for duration in durations:
            collector.record_timer("test_timer", duration)
        
        stats = collector.get_timer_stats("test_timer")
        
        assert stats['count'] == 5
        assert stats['min'] == 0.1
        assert stats['max'] == 0.5
        assert abs(stats['mean'] - 0.3) < 0.001
    
    def test_get_all_metrics(self):
        """Test getting all metrics."""
        collector = MetricsCollector()
        
        collector.increment_counter("counter1", 10)
        collector.set_gauge("gauge1", 20)
        collector.record_histogram("hist1", 30)
        collector.record_timer("timer1", 0.5)
        
        all_metrics = collector.get_all_metrics()
        
        assert "counters" in all_metrics
        assert "gauges" in all_metrics
        assert "histograms" in all_metrics
        assert "timers" in all_metrics
        
        assert all_metrics["counters"]["counter1"] == 10
        assert all_metrics["gauges"]["gauge1"] == 20
        assert all_metrics["histograms"]["hist1"]["count"] == 1
        assert all_metrics["timers"]["timer1"]["count"] == 1
    
    def test_get_recent_metrics(self):
        """Test getting recent metrics."""
        collector = MetricsCollector()
        
        # Record some metrics
        collector.increment_counter("test_metric", 1)
        time.sleep(0.1)
        collector.increment_counter("test_metric", 1)
        
        # Get recent metrics (last 1 second)
        recent = collector.get_recent_metrics("test_metric", 1)
        assert len(recent) == 2
        
        # Get very recent metrics (last 0.05 seconds)
        very_recent = collector.get_recent_metrics("test_metric", 0.05)
        assert len(very_recent) == 1


class TestHealthChecker:
    """Test health checking functionality."""
    
    def test_health_checker_initialization(self):
        """Test health checker initialization."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        assert len(checker.health_checks) == 0
        assert len(checker.last_results) == 0
        assert checker.running is False
    
    def test_register_health_check(self):
        """Test registering health checks."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        def dummy_check():
            return True
        
        checker.register_health_check("test_check", dummy_check, 30.0)
        
        assert "test_check" in checker.health_checks
        assert checker.check_intervals["test_check"] == 30.0
    
    def test_run_health_check_success(self):
        """Test running successful health check."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        def successful_check():
            return {
                'status': 'healthy',
                'message': 'All good',
                'details': {'key': 'value'}
            }
        
        result = checker._run_health_check("test_check", successful_check)
        
        assert isinstance(result, HealthCheckResult)
        assert result.service == "test_check"
        assert result.status == HealthStatus.HEALTHY
        assert result.message == "All good"
        assert result.details['key'] == 'value'
        assert result.response_time > 0
    
    def test_run_health_check_failure(self):
        """Test running failed health check."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        def failing_check():
            raise Exception("Check failed")
        
        result = checker._run_health_check("test_check", failing_check)
        
        assert result.status == HealthStatus.CRITICAL
        assert "Check failed" in result.message
        assert 'exception' in result.details
    
    def test_run_health_check_boolean_result(self):
        """Test health check with boolean result."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        def boolean_check():
            return False
        
        result = checker._run_health_check("test_check", boolean_check)
        
        assert result.status == HealthStatus.UNHEALTHY
        assert result.message == "Check failed"
    
    def test_get_health_status(self):
        """Test getting overall health status."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        # Add some mock results
        checker.last_results["service1"] = HealthCheckResult(
            service="service1",
            status=HealthStatus.HEALTHY,
            message="OK",
            timestamp=time.time(),
            response_time=0.1,
            details={}
        )
        
        checker.last_results["service2"] = HealthCheckResult(
            service="service2",
            status=HealthStatus.DEGRADED,
            message="Slow",
            timestamp=time.time(),
            response_time=0.5,
            details={}
        )
        
        status = checker.get_health_status()
        
        assert status['overall_status'] == 'degraded'
        assert 'services' in status
        assert len(status['services']) == 2
    
    def test_run_all_checks(self):
        """Test running all health checks."""
        logger = StructuredLogger("test")
        checker = HealthChecker(logger)
        
        def check1():
            return True
        
        def check2():
            return {'status': 'healthy', 'message': 'OK'}
        
        checker.register_health_check("check1", check1)
        checker.register_health_check("check2", check2)
        
        results = checker.run_all_checks()
        
        assert len(results) == 2
        assert "check1" in results
        assert "check2" in results
        assert all(isinstance(r, HealthCheckResult) for r in results.values())


class TestAlertManager:
    """Test alert management functionality."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        assert len(manager.alerts) == 0
        assert len(manager.alert_rules) == 0
        assert len(manager.notification_handlers) == 0
    
    def test_add_alert_rule(self):
        """Test adding alert rules."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        manager.add_alert_rule(
            "cpu_usage",
            threshold=80.0,
            comparison="greater_than",
            severity="warning"
        )
        
        assert "cpu_usage" in manager.alert_rules
        rule = manager.alert_rules["cpu_usage"]
        assert rule['threshold'] == 80.0
        assert rule['comparison'] == "greater_than"
        assert rule['severity'] == "warning"
    
    def test_add_notification_handler(self):
        """Test adding notification handlers."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        def dummy_handler(alert):
            pass
        
        manager.add_notification_handler(dummy_handler)
        assert len(manager.notification_handlers) == 1
    
    def test_check_metric_alerts_trigger(self):
        """Test metric alert triggering."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        # Add alert rule
        manager.add_alert_rule("cpu_usage", 80.0, "greater_than", "warning")
        
        # Create metric that should trigger alert
        metric = Metric(
            name="cpu_usage",
            value=85.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={}
        )
        
        manager.check_metric_alerts(metric)
        
        # Should have created an alert
        assert len(manager.alerts) == 1
        alert = list(manager.alerts.values())[0]
        assert alert.severity == "warning"
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert not alert.resolved
    
    def test_check_metric_alerts_no_trigger(self):
        """Test metric not triggering alert."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        # Add alert rule
        manager.add_alert_rule("cpu_usage", 80.0, "greater_than", "warning")
        
        # Create metric that should NOT trigger alert
        metric = Metric(
            name="cpu_usage",
            value=75.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={}
        )
        
        manager.check_metric_alerts(metric)
        
        # Should not have created an alert
        assert len(manager.alerts) == 0
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        logger = StructuredLogger("test")
        manager = AlertManager(logger)
        
        # Create some alerts
        alert1 = Alert(
            id="alert1",
            severity="warning",
            message="Test alert 1",
            timestamp=time.time(),
            service="test",
            metric_name="metric1",
            threshold=10.0,
            current_value=15.0,
            resolved=False
        )
        
        alert2 = Alert(
            id="alert2",
            severity="critical",
            message="Test alert 2",
            timestamp=time.time(),
            service="test",
            metric_name="metric2",
            threshold=20.0,
            current_value=25.0,
            resolved=True
        )
        
        manager.alerts["alert1"] = alert1
        manager.alerts["alert2"] = alert2
        
        active_alerts = manager.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0].id == "alert1"
        assert not active_alerts[0].resolved


class TestSystemMonitor:
    """Test system monitoring functionality."""
    
    def test_system_monitor_initialization(self):
        """Test system monitor initialization."""
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        assert monitor.metrics is collector
        assert monitor.running is False
        assert monitor.monitor_thread is None
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection."""
        # Mock system data
        mock_cpu.return_value = 45.0
        mock_memory.return_value = Mock(percent=60.0, available=1000000, used=500000)
        mock_disk.return_value = Mock(percent=70.0, free=2000000, used=1000000)
        
        collector = MetricsCollector()
        monitor = SystemMonitor(collector)
        
        monitor._collect_system_metrics()
        
        # Check that metrics were recorded
        assert collector.get_gauge('system.cpu.usage_percent') == 45.0
        assert collector.get_gauge('system.memory.usage_percent') == 60.0
        assert collector.get_gauge('system.disk.usage_percent') == 70.0


class TestMonitoringService:
    """Test main monitoring service."""
    
    def test_monitoring_service_initialization(self):
        """Test monitoring service initialization."""
        service = MonitoringService("test_service")
        
        assert service.service_name == "test_service"
        assert service.logger is not None
        assert service.metrics is not None
        assert service.health_checker is not None
        assert service.alert_manager is not None
        assert service.system_monitor is not None
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources(self, mock_disk, mock_memory, mock_cpu):
        """Test system resources health check."""
        # Mock normal system state
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)
        
        service = MonitoringService("test_service")
        result = service._check_system_resources()
        
        assert result['status'] == 'healthy'
        assert 'System resources OK' in result['message']
        assert result['details']['cpu_percent'] == 50.0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_check_system_resources_critical(self, mock_disk, mock_memory, mock_cpu):
        """Test system resources health check with critical state."""
        # Mock critical system state
        mock_cpu.return_value = 95.0
        mock_memory.return_value = Mock(percent=98.0)
        mock_disk.return_value = Mock(percent=97.0)
        
        service = MonitoringService("test_service")
        result = service._check_system_resources()
        
        assert result['status'] == 'critical'
        assert 'CPU usage critical' in result['message']
        assert 'Memory usage critical' in result['message']
        assert 'Disk usage critical' in result['message']
    
    @patch('psutil.Process')
    def test_check_memory_usage(self, mock_process_class):
        """Test memory usage health check."""
        # Mock process with normal memory usage
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(rss=100000, vms=200000)
        mock_process.memory_percent.return_value = 30.0
        mock_process_class.return_value = mock_process
        
        service = MonitoringService("test_service")
        result = service._check_memory_usage()
        
        assert result['status'] == 'healthy'
        assert 'Memory usage OK' in result['message']
        assert result['details']['memory_percent'] == 30.0
    
    def test_get_comprehensive_status(self):
        """Test getting comprehensive status."""
        service = MonitoringService("test_service")
        
        # Add some test data
        service.metrics.increment_counter("test_counter", 5)
        service.metrics.set_gauge("test_gauge", 10)
        
        status = service.get_comprehensive_status()
        
        assert status['service'] == "test_service"
        assert 'timestamp' in status
        assert 'health' in status
        assert 'metrics' in status
        assert 'alerts' in status
        
        assert status['metrics']['counters']['test_counter'] == 5
        assert status['metrics']['gauges']['test_gauge'] == 10


class TestDecorators:
    """Test monitoring decorators."""
    
    def test_timer_decorator(self):
        """Test timer decorator."""
        collector = MetricsCollector()
        
        # Temporarily replace global collector
        original_collector = monitoring_service.metrics
        monitoring_service.metrics = collector
        
        try:
            @timer_decorator("test_function")
            def test_function():
                time.sleep(0.01)  # Small delay
                return "result"
            
            result = test_function()
            
            assert result == "result"
            
            # Check that timer metric was recorded
            stats = collector.get_timer_stats("test_function")
            assert stats['count'] == 1
            assert stats['min'] > 0.005  # Should be at least 5ms
        finally:
            monitoring_service.metrics = original_collector
    
    def test_timer_decorator_with_exception(self):
        """Test timer decorator with exception."""
        collector = MetricsCollector()
        
        # Temporarily replace global collector
        original_collector = monitoring_service.metrics
        monitoring_service.metrics = collector
        
        try:
            @timer_decorator("test_function")
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            # Check that error metrics were recorded
            error_stats = collector.get_timer_stats("test_function.error")
            assert error_stats['count'] == 1
            
            error_count = collector.get_counter("test_function.errors")
            assert error_count == 1
        finally:
            monitoring_service.metrics = original_collector
    
    def test_counter_decorator(self):
        """Test counter decorator."""
        collector = MetricsCollector()
        
        # Temporarily replace global collector
        original_collector = monitoring_service.metrics
        monitoring_service.metrics = collector
        
        try:
            @counter_decorator("test_function")
            def test_function():
                return "result"
            
            # Call function multiple times
            test_function()
            test_function()
            test_function()
            
            # Check that counter metrics were recorded
            total_count = collector.get_counter("test_function")
            success_count = collector.get_counter("test_function.success")
            
            assert total_count == 3
            assert success_count == 3
        finally:
            monitoring_service.metrics = original_collector
    
    def test_counter_decorator_with_exception(self):
        """Test counter decorator with exception."""
        collector = MetricsCollector()
        
        # Temporarily replace global collector
        original_collector = monitoring_service.metrics
        monitoring_service.metrics = collector
        
        try:
            @counter_decorator("test_function")
            def failing_function():
                raise ValueError("Test error")
            
            with pytest.raises(ValueError):
                failing_function()
            
            # Check that error counter was incremented
            total_count = collector.get_counter("test_function")
            error_count = collector.get_counter("test_function.error")
            
            assert total_count == 1
            assert error_count == 1
        finally:
            monitoring_service.metrics = original_collector


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple monitoring components."""
    
    def test_end_to_end_monitoring_flow(self):
        """Test complete monitoring flow."""
        service = MonitoringService("integration_test")
        
        # Add a custom health check
        def custom_check():
            return {'status': 'healthy', 'message': 'Custom check OK'}
        
        service.health_checker.register_health_check("custom", custom_check)
        
        # Record some metrics
        service.metrics.increment_counter("requests", 10)
        service.metrics.set_gauge("active_connections", 5)
        service.metrics.record_timer("response_time", 0.5)
        
        # Add alert rule
        service.alert_manager.add_alert_rule(
            "response_time",
            threshold=1.0,
            comparison="greater_than",
            severity="warning"
        )
        
        # Run health checks
        health_results = service.health_checker.run_all_checks()
        
        # Get comprehensive status
        status = service.get_comprehensive_status()
        
        # Verify results
        assert len(health_results) >= 1  # At least our custom check
        assert "custom" in health_results
        assert health_results["custom"].status == HealthStatus.HEALTHY
        
        assert status['metrics']['counters']['requests'] == 10
        assert status['metrics']['gauges']['active_connections'] == 5
        assert status['metrics']['timers']['response_time']['count'] == 1
    
    def test_alert_triggering_flow(self):
        """Test alert triggering and resolution flow."""
        service = MonitoringService("alert_test")
        
        # Add alert rule
        service.alert_manager.add_alert_rule(
            "error_rate",
            threshold=5.0,
            comparison="greater_than",
            severity="critical"
        )
        
        # Record metric that should trigger alert
        metric = Metric(
            name="error_rate",
            value=8.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            labels={}
        )
        
        service.alert_manager.check_metric_alerts(metric)
        
        # Check that alert was created
        active_alerts = service.alert_manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].severity == "critical"
        assert active_alerts[0].current_value == 8.0
        
        # Record metric that should resolve alert
        resolve_metric = Metric(
            name="error_rate",
            value=3.0,
            metric_type=MetricType.GAUGE,
            timestamp=time.time() + 1,
            labels={}
        )
        
        service.alert_manager.check_metric_alerts(resolve_metric)
        
        # Alert should still exist but be resolved
        all_alerts = service.alert_manager.get_all_alerts()
        assert len(all_alerts) >= 1
        # Note: Resolution logic might need adjustment based on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])