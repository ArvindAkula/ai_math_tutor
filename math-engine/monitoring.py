"""
Comprehensive monitoring, logging, and health check system for the AI Math Tutor.
Provides structured logging, performance metrics, health checks, and alerting.
"""

import time
import json
import logging
import asyncio
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import sys
import os
from datetime import datetime, timedelta
import uuid

# Add shared models to path


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    status: HealthStatus
    message: str
    timestamp: float
    response_time: float
    details: Dict[str, Any]


@dataclass
class Metric:
    """Performance metric data."""
    name: str
    value: float
    metric_type: MetricType
    timestamp: float
    labels: Dict[str, str]
    unit: str = ""


@dataclass
class Alert:
    """Alert notification."""
    id: str
    severity: str
    message: str
    timestamp: float
    service: str
    metric_name: str
    threshold: float
    current_value: float
    resolved: bool = False


class StructuredLogger:
    """Enhanced structured logging system."""
    
    def __init__(self, service_name: str = "math_engine", log_level: int = logging.INFO):
        """Initialize structured logger."""
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(log_level)
        
        # Create custom formatter for structured logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler(f'{service_name}_monitoring.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            # If file logging fails, continue with console only
            pass
        
        # Request tracking
        self.request_id = None
        self.correlation_id = None
    
    def set_request_context(self, request_id: str, correlation_id: str = None):
        """Set request context for correlation."""
        self.request_id = request_id
        self.correlation_id = correlation_id or request_id
    
    def _format_message(self, message: str, extra_data: Dict[str, Any] = None) -> str:
        """Format message with structured data."""
        log_data = {
            'service': self.service_name,
            'timestamp': datetime.now().isoformat(),
            'message': message
        }
        
        if self.request_id:
            log_data['request_id'] = self.request_id
        if self.correlation_id:
            log_data['correlation_id'] = self.correlation_id
        
        if extra_data:
            log_data.update(extra_data)
        
        return json.dumps(log_data)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self.logger.info(self._format_message(message, kwargs))
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self.logger.warning(self._format_message(message, kwargs))
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self.logger.error(self._format_message(message, kwargs))
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self.logger.critical(self._format_message(message, kwargs))
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self.logger.debug(self._format_message(message, kwargs))


class MetricsCollector:
    """Collects and manages performance metrics."""
    
    def __init__(self, max_metrics: int = 10000):
        """Initialize metrics collector."""
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def increment_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Increment a counter metric."""
        with self.lock:
            self.counters[name] += value
            self._record_metric(name, value, MetricType.COUNTER, labels or {})
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set a gauge metric value."""
        with self.lock:
            self.gauges[name] = value
            self._record_metric(name, value, MetricType.GAUGE, labels or {})
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram value."""
        with self.lock:
            self.histograms[name].append(value)
            # Keep only last 1000 values for memory efficiency
            if len(self.histograms[name]) > 1000:
                self.histograms[name] = self.histograms[name][-1000:]
            self._record_metric(name, value, MetricType.HISTOGRAM, labels or {})
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """Record a timer duration."""
        with self.lock:
            self.timers[name].append(duration)
            # Keep only last 1000 values for memory efficiency
            if len(self.timers[name]) > 1000:
                self.timers[name] = self.timers[name][-1000:]
            self._record_metric(name, duration, MetricType.TIMER, labels or {})
    
    def _record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str]):
        """Record a metric in the time series."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            labels=labels
        )
        self.metrics[name].append(metric)
    
    def get_counter(self, name: str) -> float:
        """Get current counter value."""
        return self.counters.get(name, 0.0)
    
    def get_gauge(self, name: str) -> float:
        """Get current gauge value."""
        return self.gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self.histograms.get(name, [])
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            'count': count,
            'min': min(sorted_values),
            'max': max(sorted_values),
            'mean': sum(sorted_values) / count,
            'p50': sorted_values[int(count * 0.5)],
            'p90': sorted_values[int(count * 0.9)],
            'p95': sorted_values[int(count * 0.95)],
            'p99': sorted_values[int(count * 0.99)]
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        values = self.timers.get(name, [])
        if not values:
            return {}
        
        sorted_values = sorted(values)
        count = len(sorted_values)
        
        return {
            'count': count,
            'min': min(sorted_values),
            'max': max(sorted_values),
            'mean': sum(sorted_values) / count,
            'p50': sorted_values[int(count * 0.5)],
            'p90': sorted_values[int(count * 0.9)],
            'p95': sorted_values[int(count * 0.95)],
            'p99': sorted_values[int(count * 0.99)]
        }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics."""
        with self.lock:
            return {
                'counters': dict(self.counters),
                'gauges': dict(self.gauges),
                'histograms': {name: self.get_histogram_stats(name) for name in self.histograms},
                'timers': {name: self.get_timer_stats(name) for name in self.timers}
            }
    
    def get_recent_metrics(self, name: str, duration_seconds: int = 300) -> List[Metric]:
        """Get recent metrics for a specific name."""
        cutoff_time = time.time() - duration_seconds
        return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, logger: StructuredLogger):
        """Initialize health checker."""
        self.logger = logger
        self.health_checks: Dict[str, Callable] = {}
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.check_intervals: Dict[str, float] = {}
        self.running = False
        self.check_thread = None
    
    def register_health_check(self, 
                            name: str, 
                            check_func: Callable, 
                            interval_seconds: float = 60.0):
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.check_intervals[name] = interval_seconds
        self.logger.info(f"Registered health check: {name}", interval=interval_seconds)
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.running:
            return
        
        self.running = True
        self.check_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.check_thread.start()
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.running = False
        if self.check_thread:
            self.check_thread.join(timeout=5.0)
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_check_times = defaultdict(float)
        
        while self.running:
            current_time = time.time()
            
            for name, check_func in self.health_checks.items():
                interval = self.check_intervals[name]
                last_check = last_check_times[name]
                
                if current_time - last_check >= interval:
                    try:
                        result = self._run_health_check(name, check_func)
                        self.last_results[name] = result
                        last_check_times[name] = current_time
                        
                        if result.status != HealthStatus.HEALTHY:
                            self.logger.warning(
                                f"Health check failed: {name}",
                                status=result.status.value,
                                message=result.message,
                                response_time=result.response_time
                            )
                    except Exception as e:
                        self.logger.error(
                            f"Health check error: {name}",
                            error=str(e),
                            exception_type=type(e).__name__
                        )
            
            time.sleep(1.0)  # Check every second for due health checks
    
    def _run_health_check(self, name: str, check_func: Callable) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            result = check_func()
            response_time = time.time() - start_time
            
            if isinstance(result, dict):
                return HealthCheckResult(
                    service=name,
                    status=HealthStatus(result.get('status', 'healthy')),
                    message=result.get('message', 'OK'),
                    timestamp=time.time(),
                    response_time=response_time,
                    details=result.get('details', {})
                )
            else:
                return HealthCheckResult(
                    service=name,
                    status=HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY,
                    message="OK" if result else "Check failed",
                    timestamp=time.time(),
                    response_time=response_time,
                    details={}
                )
        except Exception as e:
            return HealthCheckResult(
                service=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check exception: {str(e)}",
                timestamp=time.time(),
                response_time=time.time() - start_time,
                details={'exception': str(e), 'exception_type': type(e).__name__}
            )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of all services."""
        overall_status = HealthStatus.HEALTHY
        
        for result in self.last_results.values():
            if result.status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif result.status == HealthStatus.UNHEALTHY and overall_status != HealthStatus.CRITICAL:
                overall_status = HealthStatus.UNHEALTHY
            elif result.status == HealthStatus.DEGRADED and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'services': {name: asdict(result) for name, result in self.last_results.items()}
        }
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks immediately."""
        results = {}
        for name, check_func in self.health_checks.items():
            results[name] = self._run_health_check(name, check_func)
        return results


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, logger: StructuredLogger):
        """Initialize alert manager."""
        self.logger = logger
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable] = []
    
    def add_alert_rule(self, 
                      metric_name: str,
                      threshold: float,
                      comparison: str = "greater_than",
                      severity: str = "warning",
                      message_template: str = None):
        """Add an alert rule for a metric."""
        self.alert_rules[metric_name] = {
            'threshold': threshold,
            'comparison': comparison,
            'severity': severity,
            'message_template': message_template or f"{metric_name} exceeded threshold"
        }
        self.logger.info(f"Added alert rule for {metric_name}", threshold=threshold, severity=severity)
    
    def add_notification_handler(self, handler: Callable):
        """Add a notification handler function."""
        self.notification_handlers.append(handler)
    
    def check_metric_alerts(self, metric: Metric):
        """Check if a metric triggers any alerts."""
        rule = self.alert_rules.get(metric.name)
        if not rule:
            return
        
        threshold = rule['threshold']
        comparison = rule['comparison']
        
        triggered = False
        if comparison == "greater_than" and metric.value > threshold:
            triggered = True
        elif comparison == "less_than" and metric.value < threshold:
            triggered = True
        elif comparison == "equals" and metric.value == threshold:
            triggered = True
        
        alert_id = f"{metric.name}_{int(metric.timestamp)}"
        
        if triggered:
            if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                alert = Alert(
                    id=alert_id,
                    severity=rule['severity'],
                    message=rule['message_template'].format(
                        metric_name=metric.name,
                        value=metric.value,
                        threshold=threshold
                    ),
                    timestamp=metric.timestamp,
                    service="math_engine",
                    metric_name=metric.name,
                    threshold=threshold,
                    current_value=metric.value
                )
                
                self.alerts[alert_id] = alert
                self._send_notification(alert)
        else:
            # Check if we should resolve an existing alert
            if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                self.alerts[alert_id].resolved = True
                self._send_resolution_notification(self.alerts[alert_id])
    
    def _send_notification(self, alert: Alert):
        """Send alert notification."""
        self.logger.warning(
            f"ALERT: {alert.message}",
            alert_id=alert.id,
            severity=alert.severity,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold=alert.threshold
        )
        
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler failed: {str(e)}")
    
    def _send_resolution_notification(self, alert: Alert):
        """Send alert resolution notification."""
        self.logger.info(
            f"RESOLVED: {alert.message}",
            alert_id=alert.id,
            severity=alert.severity,
            metric_name=alert.metric_name
        )
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        return list(self.alerts.values())


class SystemMonitor:
    """Monitors system resources and performance."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize system monitor."""
        self.metrics = metrics_collector
        self.running = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval_seconds: float = 30.0):
        """Start system resource monitoring."""
        if self.running:
            return
        
        self.running = True
        self.interval = interval_seconds
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self):
        """Main system monitoring loop."""
        while self.running:
            try:
                self._collect_system_metrics()
                time.sleep(self.interval)
            except Exception as e:
                # Log error but continue monitoring
                print(f"System monitoring error: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        self.metrics.set_gauge('system.cpu.usage_percent', cpu_percent)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self.metrics.set_gauge('system.memory.usage_percent', memory.percent)
        self.metrics.set_gauge('system.memory.available_bytes', memory.available)
        self.metrics.set_gauge('system.memory.used_bytes', memory.used)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        self.metrics.set_gauge('system.disk.usage_percent', disk.percent)
        self.metrics.set_gauge('system.disk.free_bytes', disk.free)
        self.metrics.set_gauge('system.disk.used_bytes', disk.used)
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            self.metrics.set_gauge('system.network.bytes_sent', network.bytes_sent)
            self.metrics.set_gauge('system.network.bytes_recv', network.bytes_recv)
        except Exception:
            pass  # Network stats might not be available
        
        # Process-specific metrics
        process = psutil.Process()
        self.metrics.set_gauge('process.memory.rss_bytes', process.memory_info().rss)
        self.metrics.set_gauge('process.cpu.usage_percent', process.cpu_percent())
        self.metrics.set_gauge('process.threads.count', process.num_threads())


class MonitoringService:
    """Main monitoring service that coordinates all monitoring components."""
    
    def __init__(self, service_name: str = "math_engine"):
        """Initialize monitoring service."""
        self.service_name = service_name
        self.logger = StructuredLogger(service_name)
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker(self.logger)
        self.alert_manager = AlertManager(self.logger)
        self.system_monitor = SystemMonitor(self.metrics)
        
        # Setup default health checks
        self._setup_default_health_checks()
        
        # Setup default alert rules
        self._setup_default_alert_rules()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.health_checker.register_health_check(
            "system_resources", 
            self._check_system_resources,
            interval_seconds=30.0
        )
        
        self.health_checker.register_health_check(
            "memory_usage",
            self._check_memory_usage,
            interval_seconds=60.0
        )
    
    def _setup_default_alert_rules(self):
        """Setup default alert rules."""
        self.alert_manager.add_alert_rule(
            "system.cpu.usage_percent",
            threshold=80.0,
            comparison="greater_than",
            severity="warning",
            message_template="High CPU usage: {value}% (threshold: {threshold}%)"
        )
        
        self.alert_manager.add_alert_rule(
            "system.memory.usage_percent",
            threshold=85.0,
            comparison="greater_than",
            severity="critical",
            message_template="High memory usage: {value}% (threshold: {threshold}%)"
        )
        
        self.alert_manager.add_alert_rule(
            "request.response_time",
            threshold=5.0,
            comparison="greater_than",
            severity="warning",
            message_template="Slow response time: {value}s (threshold: {threshold}s)"
        )
    
    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource health."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = HealthStatus.HEALTHY
        issues = []
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            issues.append(f"CPU usage critical: {cpu_percent}%")
        elif cpu_percent > 80:
            status = HealthStatus.DEGRADED
            issues.append(f"CPU usage high: {cpu_percent}%")
        
        if memory.percent > 95:
            status = HealthStatus.CRITICAL
            issues.append(f"Memory usage critical: {memory.percent}%")
        elif memory.percent > 85:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Memory usage high: {memory.percent}%")
        
        if disk.percent > 95:
            status = HealthStatus.CRITICAL
            issues.append(f"Disk usage critical: {disk.percent}%")
        elif disk.percent > 90:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Disk usage high: {disk.percent}%")
        
        return {
            'status': status.value,
            'message': '; '.join(issues) if issues else 'System resources OK',
            'details': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
        }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage health."""
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        status = HealthStatus.HEALTHY
        message = "Memory usage OK"
        
        if memory_percent > 80:
            status = HealthStatus.CRITICAL
            message = f"Process memory usage critical: {memory_percent:.1f}%"
        elif memory_percent > 60:
            status = HealthStatus.DEGRADED
            message = f"Process memory usage high: {memory_percent:.1f}%"
        
        return {
            'status': status.value,
            'message': message,
            'details': {
                'rss_bytes': memory_info.rss,
                'vms_bytes': memory_info.vms,
                'memory_percent': memory_percent
            }
        }
    
    def start_all_monitoring(self):
        """Start all monitoring components."""
        self.health_checker.start_monitoring()
        self.system_monitor.start_monitoring()
        self.logger.info("All monitoring services started")
    
    def stop_all_monitoring(self):
        """Stop all monitoring components."""
        self.health_checker.stop_monitoring()
        self.system_monitor.stop_monitoring()
        self.logger.info("All monitoring services stopped")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'service': self.service_name,
            'timestamp': time.time(),
            'health': self.health_checker.get_health_status(),
            'metrics': self.metrics.get_all_metrics(),
            'alerts': {
                'active': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
                'total': len(self.alert_manager.get_all_alerts())
            }
        }


# Global monitoring service instance
monitoring_service = MonitoringService()


def timer_decorator(metric_name: str):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                monitoring_service.metrics.record_timer(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitoring_service.metrics.record_timer(f"{metric_name}.error", duration)
                monitoring_service.metrics.increment_counter(f"{metric_name}.errors")
                raise
        return wrapper
    return decorator


def async_timer_decorator(metric_name: str):
    """Async decorator to time function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                monitoring_service.metrics.record_timer(metric_name, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitoring_service.metrics.record_timer(f"{metric_name}.error", duration)
                monitoring_service.metrics.increment_counter(f"{metric_name}.errors")
                raise
        return wrapper
    return decorator


def counter_decorator(metric_name: str):
    """Decorator to count function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitoring_service.metrics.increment_counter(metric_name)
            try:
                result = func(*args, **kwargs)
                monitoring_service.metrics.increment_counter(f"{metric_name}.success")
                return result
            except Exception as e:
                monitoring_service.metrics.increment_counter(f"{metric_name}.error")
                raise
        return wrapper
    return decorator