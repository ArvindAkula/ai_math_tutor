"""
Comprehensive error handling and fallback systems for the AI Math Tutor.
Provides robust error recovery, graceful degradation, and system resilience.
"""

import logging
import time
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import sys
import os
from functools import wraps
import asyncio

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, ValidationResult, 
    Explanation, Hint, PlotData, ComputationError
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    COMPUTATION = "computation"
    PARSING = "parsing"
    AI_SERVICE = "ai_service"
    VISUALIZATION = "visualization"
    DATABASE = "database"
    CACHE = "cache"
    NETWORK = "network"
    VALIDATION = "validation"
    TIMEOUT = "timeout"
    RESOURCE = "resource"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    operation: str
    user_id: Optional[str]
    problem_id: Optional[str]
    additional_data: Dict[str, Any]


@dataclass
class FallbackResult:
    """Result from fallback operation."""
    success: bool
    result: Any
    fallback_method: str
    confidence_score: float
    warnings: List[str]


class MathComputationError(ComputationError):
    """Enhanced computation error with context."""
    
    def __init__(self, message: str, category: ErrorCategory, context: Dict[str, Any] = None):
        super().__init__(message)
        self.category = category
        self.context = context or {}
        self.timestamp = time.time()


class AIServiceError(Exception):
    """AI service specific errors."""
    
    def __init__(self, message: str, service: str, error_code: str = None):
        super().__init__(message)
        self.service = service
        self.error_code = error_code
        self.timestamp = time.time()


class VisualizationError(Exception):
    """Visualization rendering errors."""
    
    def __init__(self, message: str, plot_type: str, data_size: int = 0):
        super().__init__(message)
        self.plot_type = plot_type
        self.data_size = data_size
        self.timestamp = time.time()


class ErrorLogger:
    """Centralized error logging system."""
    
    def __init__(self, log_level: int = logging.INFO):
        """Initialize error logger."""
        self.logger = logging.getLogger('math_engine_errors')
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        try:
            file_handler = logging.FileHandler('math_engine_errors.log')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception:
            # If file logging fails, continue with console only
            pass
    
    def log_error(self, error: Exception, context: ErrorContext):
        """Log error with context information."""
        error_msg = f"[{context.error_id}] {context.category.value.upper()} ERROR in {context.operation}: {str(error)}"
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(error_msg, extra={'context': context.__dict__})
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(error_msg, extra={'context': context.__dict__})
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(error_msg, extra={'context': context.__dict__})
        else:
            self.logger.info(error_msg, extra={'context': context.__dict__})
    
    def log_fallback_used(self, operation: str, fallback_method: str, success: bool):
        """Log when fallback methods are used."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"FALLBACK {status}: {operation} -> {fallback_method}")


class FallbackManager:
    """Manages fallback strategies for different operations."""
    
    def __init__(self):
        """Initialize fallback manager."""
        self.error_logger = ErrorLogger()
        self.fallback_strategies = {
            'math_solving': [
                self._basic_symbolic_solving,
                self._numerical_approximation,
                self._pattern_matching_solution
            ],
            'ai_explanation': [
                self._rule_based_explanation,
                self._template_explanation,
                self._minimal_explanation
            ],
            'visualization': [
                self._simple_plot_fallback,
                self._text_based_visualization,
                self._no_visualization_fallback
            ],
            'parsing': [
                self._simplified_parsing,
                self._keyword_based_parsing,
                self._manual_parsing_prompt
            ]
        }
    
    def execute_with_fallback(self, 
                            operation: str,
                            primary_function: Callable,
                            *args,
                            **kwargs) -> FallbackResult:
        """
        Execute operation with automatic fallback on failure.
        
        Args:
            operation: Operation name for fallback lookup
            primary_function: Primary function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            FallbackResult with success status and result
        """
        # Try primary function first
        try:
            result = primary_function(*args, **kwargs)
            return FallbackResult(
                success=True,
                result=result,
                fallback_method="primary",
                confidence_score=1.0,
                warnings=[]
            )
        except Exception as e:
            self.error_logger.logger.warning(f"Primary function failed for {operation}: {str(e)}")
        
        # Try fallback strategies
        strategies = self.fallback_strategies.get(operation, [])
        
        for i, fallback_func in enumerate(strategies):
            try:
                result = fallback_func(*args, **kwargs)
                self.error_logger.log_fallback_used(operation, f"fallback_{i+1}", True)
                
                return FallbackResult(
                    success=True,
                    result=result,
                    fallback_method=f"fallback_{i+1}",
                    confidence_score=max(0.1, 0.8 - (i * 0.2)),
                    warnings=[f"Used fallback method {i+1} due to primary failure"]
                )
            except Exception as e:
                self.error_logger.logger.warning(f"Fallback {i+1} failed for {operation}: {str(e)}")
                continue
        
        # All fallbacks failed
        self.error_logger.log_fallback_used(operation, "all_fallbacks", False)
        return FallbackResult(
            success=False,
            result=None,
            fallback_method="none",
            confidence_score=0.0,
            warnings=["All fallback strategies failed"]
        )
    
    def _basic_symbolic_solving(self, problem: ParsedProblem) -> StepSolution:
        """Basic symbolic solving fallback."""
        import sympy as sp
        from models import SolutionStep
        
        try:
            # Extract simple equation
            text = problem.original_text.lower()
            if '=' in text:
                left, right = text.split('=', 1)
                left_expr = sp.sympify(left.strip())
                right_expr = sp.sympify(right.strip())
                
                # Try to solve
                variables = list(left_expr.free_symbols.union(right_expr.free_symbols))
                if variables:
                    var = variables[0]
                    solution = sp.solve(sp.Eq(left_expr, right_expr), var)
                    
                    steps = [
                        SolutionStep(
                            step_number=1,
                            operation="Basic symbolic solving",
                            explanation="Using symbolic computation to solve the equation",
                            mathematical_expression=f"{left_expr} = {right_expr}",
                            intermediate_result=str(solution[0]) if solution else "No solution"
                        )
                    ]
                    
                    return StepSolution(
                        problem_id=problem.id,
                        steps=steps,
                        final_answer=str(solution[0]) if solution else "No solution",
                        solution_method="basic_symbolic",
                        confidence_score=0.6,
                        computation_time=0.1
                    )
        except Exception:
            pass
        
        raise MathComputationError("Basic symbolic solving failed", ErrorCategory.COMPUTATION)
    
    def _numerical_approximation(self, problem: ParsedProblem) -> StepSolution:
        """Numerical approximation fallback."""
        from models import SolutionStep
        
        steps = [
            SolutionStep(
                step_number=1,
                operation="Numerical approximation",
                explanation="Using numerical methods to approximate the solution",
                mathematical_expression=problem.original_text,
                intermediate_result="Approximate solution computed"
            )
        ]
        
        return StepSolution(
            problem_id=problem.id,
            steps=steps,
            final_answer="Numerical approximation (exact method failed)",
            solution_method="numerical_approximation",
            confidence_score=0.4,
            computation_time=0.1
        )
    
    def _pattern_matching_solution(self, problem: ParsedProblem) -> StepSolution:
        """Pattern matching solution fallback."""
        from models import SolutionStep
        
        steps = [
            SolutionStep(
                step_number=1,
                operation="Pattern matching",
                explanation="Using pattern recognition to identify solution approach",
                mathematical_expression=problem.original_text,
                intermediate_result="Pattern-based solution identified"
            )
        ]
        
        return StepSolution(
            problem_id=problem.id,
            steps=steps,
            final_answer="Pattern-based solution (computational methods failed)",
            solution_method="pattern_matching",
            confidence_score=0.3,
            computation_time=0.05
        )
    
    def _rule_based_explanation(self, step: Dict[str, Any], user_level: str = "intermediate") -> Dict[str, Any]:
        """Rule-based explanation fallback."""
        operation = step.get('operation', '').lower()
        
        if 'derivative' in operation:
            content = "This step involves taking the derivative using standard differentiation rules."
        elif 'integral' in operation:
            content = "This step involves integration using standard integration techniques."
        elif 'solve' in operation:
            content = "This step involves solving the equation using algebraic methods."
        elif 'simplify' in operation:
            content = "This step involves simplifying the expression by combining like terms."
        else:
            content = f"This step performs {operation} to progress toward the solution."
        
        return {
            "content": content,
            "complexity_level": user_level,
            "related_concepts": [],
            "examples": []
        }
    
    def _template_explanation(self, step: Dict[str, Any], user_level: str = "intermediate") -> Dict[str, Any]:
        """Template-based explanation fallback."""
        return {
            "content": f"Step {step.get('step_number', 1)}: {step.get('explanation', 'Mathematical operation performed')}",
            "complexity_level": user_level,
            "related_concepts": [],
            "examples": []
        }
    
    def _minimal_explanation(self, step: Dict[str, Any], user_level: str = "intermediate") -> Dict[str, Any]:
        """Minimal explanation fallback."""
        return {
            "content": "Mathematical step completed. Please refer to textbook for detailed explanation.",
            "complexity_level": user_level,
            "related_concepts": [],
            "examples": []
        }
    
    def _simple_plot_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Simple plotting fallback."""
        return {
            "plot_data": {
                "plot_type": "simple",
                "title": "Mathematical Visualization",
                "axis_labels": ["x", "y"],
                "data_points_count": 0,
                "interactive_elements_count": 0
            },
            "image_base64": "",
            "problem_type": "general",
            "solution_method": "fallback_visualization"
        }
    
    def _text_based_visualization(self, *args, **kwargs) -> Dict[str, Any]:
        """Text-based visualization fallback."""
        return {
            "plot_data": {
                "plot_type": "text",
                "title": "Text-based Mathematical Description",
                "axis_labels": [],
                "data_points_count": 0,
                "interactive_elements_count": 0
            },
            "image_base64": "",
            "problem_type": "general",
            "solution_method": "text_visualization",
            "text_description": "Visualization not available. Please refer to the mathematical expressions."
        }
    
    def _no_visualization_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """No visualization fallback."""
        return {
            "plot_data": {
                "plot_type": "none",
                "title": "Visualization Unavailable",
                "axis_labels": [],
                "data_points_count": 0,
                "interactive_elements_count": 0
            },
            "image_base64": "",
            "problem_type": "general",
            "solution_method": "no_visualization",
            "message": "Visualization could not be generated for this problem."
        }
    
    def _simplified_parsing(self, problem_text: str, domain: str = None) -> ParsedProblem:
        """Simplified parsing fallback."""
        from models import MathDomain, DifficultyLevel
        import uuid
        
        # Basic keyword detection
        domain_map = {
            'derivative': MathDomain.CALCULUS,
            'integral': MathDomain.CALCULUS,
            'solve': MathDomain.ALGEBRA,
            'factor': MathDomain.ALGEBRA,
            'matrix': MathDomain.LINEAR_ALGEBRA,
            'vector': MathDomain.LINEAR_ALGEBRA
        }
        
        detected_domain = MathDomain.ALGEBRA  # default
        for keyword, math_domain in domain_map.items():
            if keyword in problem_text.lower():
                detected_domain = math_domain
                break
        
        return ParsedProblem(
            id=str(uuid.uuid4()),
            original_text=problem_text,
            domain=detected_domain,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],  # default variable
            expressions=[problem_text],
            problem_type='general_problem',
            metadata={'parsing_method': 'simplified_fallback'}
        )
    
    def _keyword_based_parsing(self, problem_text: str, domain: str = None) -> ParsedProblem:
        """Keyword-based parsing fallback."""
        from models import MathDomain, DifficultyLevel
        import uuid
        import re
        
        # Extract variables using regex
        variables = list(set(re.findall(r'\b[a-z]\b', problem_text.lower())))
        if not variables:
            variables = ['x']
        
        # Extract expressions (simple approach)
        expressions = [expr.strip() for expr in re.split(r'[,;]', problem_text) if expr.strip()]
        if not expressions:
            expressions = [problem_text]
        
        return ParsedProblem(
            id=str(uuid.uuid4()),
            original_text=problem_text,
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=variables,
            expressions=expressions,
            problem_type='general_problem',
            metadata={'parsing_method': 'keyword_based_fallback'}
        )
    
    def _manual_parsing_prompt(self, problem_text: str, domain: str = None) -> ParsedProblem:
        """Manual parsing prompt fallback."""
        from models import MathDomain, DifficultyLevel
        import uuid
        
        return ParsedProblem(
            id=str(uuid.uuid4()),
            original_text=problem_text,
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=[problem_text],
            problem_type='manual_parsing_required',
            metadata={
                'parsing_method': 'manual_required',
                'message': 'Automatic parsing failed. Manual intervention required.'
            }
        )


class CircuitBreaker:
    """Circuit breaker pattern for external service calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying again (seconds)
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def __call__(self, func):
        """Decorator to apply circuit breaker."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.state == 'OPEN':
                if self._should_attempt_reset():
                    self.state = 'HALF_OPEN'
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.expected_exception as e:
                self._on_failure()
                raise e
        
        return wrapper
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0):
        """
        Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay between retries (seconds)
            max_delay: Maximum delay between retries (seconds)
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry and exponential backoff.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Last exception if all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.backoff_factor ** attempt),
                        self.max_delay
                    )
                    time.sleep(delay)
                else:
                    break
        
        raise last_exception


class GracefulDegradation:
    """Manages graceful degradation of service features."""
    
    def __init__(self):
        """Initialize graceful degradation manager."""
        self.service_status = {
            'ai_service': True,
            'visualization': True,
            'cache': True,
            'database': True
        }
        self.fallback_manager = FallbackManager()
        self.error_logger = ErrorLogger()
    
    def mark_service_down(self, service: str):
        """Mark a service as down."""
        self.service_status[service] = False
        self.error_logger.logger.warning(f"Service marked as down: {service}")
    
    def mark_service_up(self, service: str):
        """Mark a service as up."""
        self.service_status[service] = True
        self.error_logger.logger.info(f"Service marked as up: {service}")
    
    def is_service_available(self, service: str) -> bool:
        """Check if a service is available."""
        return self.service_status.get(service, False)
    
    def get_degraded_response(self, operation: str, error: Exception) -> Dict[str, Any]:
        """Get degraded response when service is unavailable."""
        if operation == 'ai_explanation':
            return {
                'content': 'AI explanation service temporarily unavailable. Please refer to mathematical textbooks.',
                'complexity_level': 'intermediate',
                'related_concepts': [],
                'examples': [],
                'service_status': 'degraded'
            }
        elif operation == 'visualization':
            return {
                'plot_data': {'plot_type': 'unavailable'},
                'message': 'Visualization service temporarily unavailable.',
                'service_status': 'degraded'
            }
        elif operation == 'problem_solving':
            return {
                'steps': [],
                'final_answer': 'Problem solving service temporarily unavailable.',
                'solution_method': 'unavailable',
                'service_status': 'degraded'
            }
        else:
            return {
                'message': f'{operation} service temporarily unavailable.',
                'service_status': 'degraded'
            }


# Global instances
fallback_manager = FallbackManager()
error_logger = ErrorLogger()
graceful_degradation = GracefulDegradation()
retry_manager = RetryManager()


def error_handler(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Decorator for comprehensive error handling.
    
    Args:
        category: Error category
        severity: Error severity level
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            operation = func.__name__
            error_id = f"{operation}_{int(time.time())}"
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error_id=error_id,
                    timestamp=time.time(),
                    category=category,
                    severity=severity,
                    operation=operation,
                    user_id=kwargs.get('user_id'),
                    problem_id=kwargs.get('problem_id'),
                    additional_data={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'exception_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                
                error_logger.log_error(e, context)
                
                # Try fallback if available
                if category.value in fallback_manager.fallback_strategies:
                    fallback_result = fallback_manager.execute_with_fallback(
                        category.value, func, *args, **kwargs
                    )
                    if fallback_result.success:
                        return fallback_result.result
                
                # If no fallback or fallback failed, re-raise with context
                if isinstance(e, (MathComputationError, AIServiceError, VisualizationError)):
                    raise e
                else:
                    raise MathComputationError(
                        f"Operation {operation} failed: {str(e)}", 
                        category, 
                        context.__dict__
                    )
        
        return wrapper
    return decorator


def async_error_handler(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Async version of error handler decorator.
    
    Args:
        category: Error category
        severity: Error severity level
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            operation = func.__name__
            error_id = f"{operation}_{int(time.time())}"
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    error_id=error_id,
                    timestamp=time.time(),
                    category=category,
                    severity=severity,
                    operation=operation,
                    user_id=kwargs.get('user_id'),
                    problem_id=kwargs.get('problem_id'),
                    additional_data={
                        'args_count': len(args),
                        'kwargs_keys': list(kwargs.keys()),
                        'exception_type': type(e).__name__,
                        'traceback': traceback.format_exc()
                    }
                )
                
                error_logger.log_error(e, context)
                
                # For async operations, we'll use graceful degradation
                degraded_response = graceful_degradation.get_degraded_response(operation, e)
                return degraded_response
        
        return wrapper
    return decorator