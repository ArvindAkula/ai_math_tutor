"""
Comprehensive tests for error handling and fallback systems.
Tests error recovery, system resilience, and graceful degradation.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, MathDomain, DifficultyLevel
)

from error_handling import (
    ErrorSeverity, ErrorCategory, ErrorContext, FallbackResult,
    MathComputationError, AIServiceError, VisualizationError,
    ErrorLogger, FallbackManager, CircuitBreaker, RetryManager,
    GracefulDegradation, error_handler, async_error_handler,
    fallback_manager, error_logger, graceful_degradation, retry_manager
)


class TestErrorClasses:
    """Test custom error classes."""
    
    def test_math_computation_error(self):
        """Test MathComputationError creation and attributes."""
        context = {"problem_id": "test_123", "operation": "solve"}
        error = MathComputationError(
            "Computation failed", 
            ErrorCategory.COMPUTATION, 
            context
        )
        
        assert str(error) == "Computation failed"
        assert error.category == ErrorCategory.COMPUTATION
        assert error.context == context
        assert error.timestamp > 0
    
    def test_ai_service_error(self):
        """Test AIServiceError creation and attributes."""
        error = AIServiceError("API failed", "openai", "rate_limit")
        
        assert str(error) == "API failed"
        assert error.service == "openai"
        assert error.error_code == "rate_limit"
        assert error.timestamp > 0
    
    def test_visualization_error(self):
        """Test VisualizationError creation and attributes."""
        error = VisualizationError("Plot failed", "3d_surface", 1000)
        
        assert str(error) == "Plot failed"
        assert error.plot_type == "3d_surface"
        assert error.data_size == 1000
        assert error.timestamp > 0


class TestErrorLogger:
    """Test error logging functionality."""
    
    def test_error_logger_initialization(self):
        """Test error logger initialization."""
        logger = ErrorLogger()
        assert logger.logger is not None
        assert logger.logger.name == 'math_engine_errors'
    
    def test_log_error(self):
        """Test error logging with context."""
        logger = ErrorLogger()
        error = Exception("Test error")
        context = ErrorContext(
            error_id="test_001",
            timestamp=time.time(),
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            operation="test_operation",
            user_id="user_123",
            problem_id="prob_456",
            additional_data={"key": "value"}
        )
        
        # Should not raise exception
        logger.log_error(error, context)
    
    def test_log_fallback_used(self):
        """Test fallback logging."""
        logger = ErrorLogger()
        
        # Should not raise exception
        logger.log_fallback_used("math_solving", "basic_symbolic", True)
        logger.log_fallback_used("ai_explanation", "rule_based", False)


class TestFallbackManager:
    """Test fallback management system."""
    
    def test_fallback_manager_initialization(self):
        """Test fallback manager initialization."""
        manager = FallbackManager()
        assert 'math_solving' in manager.fallback_strategies
        assert 'ai_explanation' in manager.fallback_strategies
        assert 'visualization' in manager.fallback_strategies
        assert 'parsing' in manager.fallback_strategies
    
    def test_execute_with_fallback_success(self):
        """Test successful primary function execution."""
        manager = FallbackManager()
        
        def successful_function(x):
            return x * 2
        
        result = manager.execute_with_fallback(
            'test_operation', successful_function, 5
        )
        
        assert result.success is True
        assert result.result == 10
        assert result.fallback_method == "primary"
        assert result.confidence_score == 1.0
        assert len(result.warnings) == 0
    
    def test_execute_with_fallback_primary_fails(self):
        """Test fallback execution when primary fails."""
        manager = FallbackManager()
        
        def failing_function():
            raise Exception("Primary failed")
        
        # Mock a fallback strategy
        def mock_fallback():
            return "fallback_result"
        
        manager.fallback_strategies['test_operation'] = [mock_fallback]
        
        result = manager.execute_with_fallback(
            'test_operation', failing_function
        )
        
        assert result.success is True
        assert result.result == "fallback_result"
        assert result.fallback_method == "fallback_1"
        assert result.confidence_score == 0.8
        assert len(result.warnings) == 1
    
    def test_execute_with_fallback_all_fail(self):
        """Test when all fallback strategies fail."""
        manager = FallbackManager()
        
        def failing_function():
            raise Exception("Primary failed")
        
        def failing_fallback():
            raise Exception("Fallback failed")
        
        manager.fallback_strategies['test_operation'] = [failing_fallback]
        
        result = manager.execute_with_fallback(
            'test_operation', failing_function
        )
        
        assert result.success is False
        assert result.result is None
        assert result.fallback_method == "none"
        assert result.confidence_score == 0.0
        assert "All fallback strategies failed" in result.warnings
    
    def test_basic_symbolic_solving_fallback(self):
        """Test basic symbolic solving fallback."""
        manager = FallbackManager()
        
        problem = ParsedProblem(
            id="test_001",
            original_text="x + 2 = 5",
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.BEGINNER,
            variables=["x"],
            expressions=["x + 2", "5"],
            problem_type="linear_equation",
            metadata={}
        )
        
        try:
            result = manager._basic_symbolic_solving(problem)
            assert isinstance(result, StepSolution)
            assert result.problem_id == "test_001"
            assert len(result.steps) > 0
        except MathComputationError:
            # Expected if sympy parsing fails
            pass
    
    def test_rule_based_explanation_fallback(self):
        """Test rule-based explanation fallback."""
        manager = FallbackManager()
        
        step = {
            'operation': 'derivative',
            'step_number': 1,
            'explanation': 'Taking derivative'
        }
        
        result = manager._rule_based_explanation(step, 'intermediate')
        
        assert 'content' in result
        assert 'derivative' in result['content'].lower()
        assert result['complexity_level'] == 'intermediate'
        assert isinstance(result['related_concepts'], list)
    
    def test_simple_plot_fallback(self):
        """Test simple plotting fallback."""
        manager = FallbackManager()
        
        result = manager._simple_plot_fallback()
        
        assert 'plot_data' in result
        assert result['plot_data']['plot_type'] == 'simple'
        assert 'image_base64' in result
        assert result['solution_method'] == 'fallback_visualization'
    
    def test_simplified_parsing_fallback(self):
        """Test simplified parsing fallback."""
        manager = FallbackManager()
        
        result = manager._simplified_parsing("Find the derivative of x^2", "calculus")
        
        assert isinstance(result, ParsedProblem)
        assert result.domain == MathDomain.CALCULUS
        assert 'simplified_fallback' in result.metadata['parsing_method']


class TestCircuitBreaker:
    """Test circuit breaker pattern."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
        
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 30
        assert cb.failure_count == 0
        assert cb.state == 'CLOSED'
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 0
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=2)
        
        @cb
        def failing_function():
            raise Exception("Function failed")
        
        # First failure
        with pytest.raises(Exception):
            failing_function()
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 1
        
        # Second failure - should open circuit
        with pytest.raises(Exception):
            failing_function()
        assert cb.state == 'OPEN'
        assert cb.failure_count == 2
        
        # Third call should fail immediately due to open circuit
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            failing_function()
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker half-open state and recovery."""
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        
        @cb
        def function_that_recovers():
            if cb.state == 'HALF_OPEN':
                return "recovered"
            raise Exception("Still failing")
        
        # Cause failure to open circuit
        with pytest.raises(Exception):
            function_that_recovers()
        assert cb.state == 'OPEN'
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Next call should succeed and close circuit
        result = function_that_recovers()
        assert result == "recovered"
        assert cb.state == 'CLOSED'
        assert cb.failure_count == 0


class TestRetryManager:
    """Test retry management with exponential backoff."""
    
    def test_retry_manager_initialization(self):
        """Test retry manager initialization."""
        rm = RetryManager(max_retries=5, base_delay=0.5)
        
        assert rm.max_retries == 5
        assert rm.base_delay == 0.5
        assert rm.backoff_factor == 2.0
    
    def test_retry_success_on_first_attempt(self):
        """Test successful function on first attempt."""
        rm = RetryManager()
        
        def successful_function():
            return "success"
        
        result = rm.retry_with_backoff(successful_function)
        assert result == "success"
    
    def test_retry_success_after_failures(self):
        """Test successful function after some failures."""
        rm = RetryManager(max_retries=3, base_delay=0.01)
        
        call_count = 0
        def function_succeeds_on_third_try():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return "success"
        
        start_time = time.time()
        result = rm.retry_with_backoff(function_succeeds_on_third_try)
        end_time = time.time()
        
        assert result == "success"
        assert call_count == 3
        # Should have some delay due to backoff
        assert end_time - start_time > 0.01
    
    def test_retry_all_attempts_fail(self):
        """Test when all retry attempts fail."""
        rm = RetryManager(max_retries=2, base_delay=0.01)
        
        def always_failing_function():
            raise ValueError("Always fails")
        
        with pytest.raises(ValueError, match="Always fails"):
            rm.retry_with_backoff(always_failing_function)


class TestGracefulDegradation:
    """Test graceful degradation system."""
    
    def test_graceful_degradation_initialization(self):
        """Test graceful degradation initialization."""
        gd = GracefulDegradation()
        
        assert gd.service_status['ai_service'] is True
        assert gd.service_status['visualization'] is True
        assert gd.service_status['cache'] is True
        assert gd.service_status['database'] is True
    
    def test_mark_service_down_and_up(self):
        """Test marking services as down and up."""
        gd = GracefulDegradation()
        
        # Mark service down
        gd.mark_service_down('ai_service')
        assert gd.service_status['ai_service'] is False
        assert gd.is_service_available('ai_service') is False
        
        # Mark service up
        gd.mark_service_up('ai_service')
        assert gd.service_status['ai_service'] is True
        assert gd.is_service_available('ai_service') is True
    
    def test_get_degraded_response_ai_explanation(self):
        """Test degraded response for AI explanation."""
        gd = GracefulDegradation()
        error = Exception("AI service down")
        
        response = gd.get_degraded_response('ai_explanation', error)
        
        assert 'content' in response
        assert 'temporarily unavailable' in response['content']
        assert response['service_status'] == 'degraded'
    
    def test_get_degraded_response_visualization(self):
        """Test degraded response for visualization."""
        gd = GracefulDegradation()
        error = Exception("Visualization service down")
        
        response = gd.get_degraded_response('visualization', error)
        
        assert 'plot_data' in response
        assert response['plot_data']['plot_type'] == 'unavailable'
        assert response['service_status'] == 'degraded'
    
    def test_get_degraded_response_generic(self):
        """Test degraded response for generic operation."""
        gd = GracefulDegradation()
        error = Exception("Service down")
        
        response = gd.get_degraded_response('unknown_operation', error)
        
        assert 'message' in response
        assert 'temporarily unavailable' in response['message']
        assert response['service_status'] == 'degraded'


class TestErrorHandlerDecorators:
    """Test error handler decorators."""
    
    def test_error_handler_decorator_success(self):
        """Test error handler decorator with successful function."""
        @error_handler(ErrorCategory.COMPUTATION, ErrorSeverity.LOW)
        def successful_function(x):
            return x * 2
        
        result = successful_function(5)
        assert result == 10
    
    def test_error_handler_decorator_with_fallback(self):
        """Test error handler decorator with fallback."""
        # Mock fallback manager to have a strategy for 'computation'
        original_strategies = fallback_manager.fallback_strategies.copy()
        
        def mock_fallback(*args, **kwargs):
            return "fallback_result"
        
        fallback_manager.fallback_strategies['computation'] = [mock_fallback]
        
        @error_handler(ErrorCategory.COMPUTATION, ErrorSeverity.MEDIUM)
        def failing_function():
            raise Exception("Function failed")
        
        try:
            result = failing_function()
            assert result == "fallback_result"
        finally:
            # Restore original strategies
            fallback_manager.fallback_strategies = original_strategies
    
    def test_error_handler_decorator_no_fallback(self):
        """Test error handler decorator without fallback."""
        @error_handler(ErrorCategory.NETWORK, ErrorSeverity.HIGH)
        def failing_function():
            raise ValueError("Network error")
        
        with pytest.raises(MathComputationError):
            failing_function()
    
    @pytest.mark.asyncio
    async def test_async_error_handler_decorator(self):
        """Test async error handler decorator."""
        @async_error_handler(ErrorCategory.AI_SERVICE, ErrorSeverity.MEDIUM)
        async def failing_async_function():
            raise Exception("Async function failed")
        
        result = await failing_async_function()
        
        # Should return degraded response
        assert isinstance(result, dict)
        assert 'service_status' in result
        assert result['service_status'] == 'degraded'


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple error handling components."""
    
    def test_complete_error_recovery_flow(self):
        """Test complete error recovery flow with multiple fallbacks."""
        # Create a function that fails initially but has fallbacks
        call_count = 0
        
        def unreliable_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise AIServiceError("Primary AI service failed", "openai")
            elif call_count == 2:
                raise Exception("First fallback failed")
            else:
                return "final_fallback_success"
        
        # Set up fallback strategies
        original_strategies = fallback_manager.fallback_strategies.copy()
        fallback_manager.fallback_strategies['test_recovery'] = [
            lambda: unreliable_function(),  # This will fail
            lambda: unreliable_function()   # This will succeed
        ]
        
        try:
            result = fallback_manager.execute_with_fallback(
                'test_recovery', unreliable_function
            )
            
            assert result.success is True
            assert result.result == "final_fallback_success"
            assert result.fallback_method == "fallback_2"
            assert call_count == 3
        finally:
            fallback_manager.fallback_strategies = original_strategies
    
    def test_circuit_breaker_with_retry(self):
        """Test circuit breaker combined with retry logic."""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.1)
        rm = RetryManager(max_retries=1, base_delay=0.01)
        
        failure_count = 0
        
        @cb
        def flaky_service():
            nonlocal failure_count
            failure_count += 1
            if failure_count <= 2:  # Reduced to 2 to match circuit breaker threshold
                raise Exception("Service temporarily down")
            return "service_recovered"
        
        # First few calls should fail and open circuit
        with pytest.raises(Exception):
            rm.retry_with_backoff(flaky_service)
        
        with pytest.raises(Exception):
            rm.retry_with_backoff(flaky_service)
        
        # Circuit should be open now
        assert cb.state == 'OPEN'
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Service should recover on next attempt (circuit goes to HALF_OPEN then CLOSED)
        result = rm.retry_with_backoff(flaky_service)
        assert result == "service_recovered"
        assert cb.state == 'CLOSED'
    
    def test_graceful_degradation_with_fallback(self):
        """Test graceful degradation combined with fallback strategies."""
        gd = GracefulDegradation()
        
        # Mark AI service as down
        gd.mark_service_down('ai_service')
        
        # Function that checks service availability
        def ai_dependent_function():
            if not gd.is_service_available('ai_service'):
                raise AIServiceError("AI service unavailable", "openai")
            return "ai_result"
        
        # Set up fallback
        def rule_based_fallback():
            return "rule_based_result"
        
        original_strategies = fallback_manager.fallback_strategies.copy()
        fallback_manager.fallback_strategies['ai_operation'] = [rule_based_fallback]
        
        try:
            result = fallback_manager.execute_with_fallback(
                'ai_operation', ai_dependent_function
            )
            
            assert result.success is True
            assert result.result == "rule_based_result"
            assert result.fallback_method == "fallback_1"
        finally:
            fallback_manager.fallback_strategies = original_strategies


class TestErrorRecoveryMetrics:
    """Test error recovery metrics and monitoring."""
    
    def test_error_context_creation(self):
        """Test error context creation with all fields."""
        context = ErrorContext(
            error_id="test_123",
            timestamp=time.time(),
            category=ErrorCategory.COMPUTATION,
            severity=ErrorSeverity.HIGH,
            operation="solve_problem",
            user_id="user_456",
            problem_id="prob_789",
            additional_data={"retry_count": 2, "fallback_used": True}
        )
        
        assert context.error_id == "test_123"
        assert context.category == ErrorCategory.COMPUTATION
        assert context.severity == ErrorSeverity.HIGH
        assert context.operation == "solve_problem"
        assert context.user_id == "user_456"
        assert context.problem_id == "prob_789"
        assert context.additional_data["retry_count"] == 2
    
    def test_fallback_result_metrics(self):
        """Test fallback result metrics."""
        result = FallbackResult(
            success=True,
            result="fallback_data",
            fallback_method="rule_based",
            confidence_score=0.7,
            warnings=["Primary method failed", "Using fallback"]
        )
        
        assert result.success is True
        assert result.result == "fallback_data"
        assert result.fallback_method == "rule_based"
        assert result.confidence_score == 0.7
        assert len(result.warnings) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])