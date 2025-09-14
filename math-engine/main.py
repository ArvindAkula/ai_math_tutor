"""
AI Math Tutor - Math Engine Service
Main FastAPI application for mathematical computation and AI explanations.
Enhanced with comprehensive monitoring, logging, and health checks.
"""

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os
import time
import uuid

# Import local models
from models import (
    ParsedProblem, StepSolution, ValidationResult, 
    Explanation, Hint, PlotData, MathDomain
)

# Import cache service
from cache_service import (
    cache_service, problem_cache, ai_explanation_cache, visualization_cache
)

# Import database pool
from database_pool import db_pool, optimized_queries

# Import monitoring and error handling
from monitoring import (
    monitoring_service, timer_decorator, async_timer_decorator, counter_decorator
)
from error_handling import (
    error_handler, async_error_handler, ErrorCategory, ErrorSeverity
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management with monitoring."""
    # Startup
    monitoring_service.logger.info("Math Engine starting up...")
    
    # Initialize cache service
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    cache_service.redis_url = redis_url
    await cache_service.connect()
    monitoring_service.logger.info("Cache service connected")
    
    # Initialize database pool
    database_url = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/ai_math_tutor")
    db_pool.database_url = database_url
    await db_pool.initialize()
    monitoring_service.logger.info("Database pool initialized")
    
    # Start monitoring services
    monitoring_service.start_all_monitoring()
    monitoring_service.logger.info("Monitoring services started")
    
    yield
    
    # Shutdown
    monitoring_service.logger.info("Math Engine shutting down...")
    
    # Stop monitoring services
    monitoring_service.stop_all_monitoring()
    monitoring_service.logger.info("Monitoring services stopped")
    
    await cache_service.disconnect()
    monitoring_service.logger.info("Cache service disconnected")
    
    await db_pool.close()
    monitoring_service.logger.info("Database pool closed")


# Create FastAPI app
app = FastAPI(
    title="AI Math Tutor - Math Engine",
    description="Mathematical computation and AI explanation service",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "AI Math Tutor - Math Engine",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
@async_timer_decorator("health_check")
async def health_check():
    """Comprehensive health check with monitoring."""
    monitoring_service.metrics.increment_counter("health_check.requests")
    
    try:
        # Run all health checks
        health_results = monitoring_service.health_checker.run_all_checks()
        overall_status = monitoring_service.health_checker.get_health_status()
        
        monitoring_service.metrics.increment_counter("health_check.success")
        
        return {
            "status": overall_status['overall_status'],
            "timestamp": overall_status['timestamp'],
            "services": overall_status['services'],
            "basic_services": {
                "sympy": "available",
                "numpy": "available", 
                "matplotlib": "available"
            }
        }
    except Exception as e:
        monitoring_service.metrics.increment_counter("health_check.errors")
        monitoring_service.logger.error("Health check failed", error=str(e))
        
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }


@app.get("/health/detailed")
@async_timer_decorator("detailed_health_check")
async def detailed_health_check():
    """Detailed health check with comprehensive system status."""
    monitoring_service.metrics.increment_counter("detailed_health_check.requests")
    
    try:
        comprehensive_status = monitoring_service.get_comprehensive_status()
        monitoring_service.metrics.increment_counter("detailed_health_check.success")
        return comprehensive_status
    except Exception as e:
        monitoring_service.metrics.increment_counter("detailed_health_check.errors")
        monitoring_service.logger.error("Detailed health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")


@app.get("/metrics")
@async_timer_decorator("metrics_endpoint")
async def get_metrics():
    """Get all performance metrics."""
    monitoring_service.metrics.increment_counter("metrics.requests")
    
    try:
        all_metrics = monitoring_service.metrics.get_all_metrics()
        monitoring_service.metrics.increment_counter("metrics.success")
        return {
            "timestamp": time.time(),
            "service": "math_engine",
            "metrics": all_metrics
        }
    except Exception as e:
        monitoring_service.metrics.increment_counter("metrics.errors")
        monitoring_service.logger.error("Metrics retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")


@app.get("/metrics/{metric_name}")
@async_timer_decorator("specific_metric")
async def get_specific_metric(metric_name: str):
    """Get specific metric data."""
    monitoring_service.metrics.increment_counter("specific_metric.requests")
    
    try:
        # Get recent metrics for the specified name
        recent_metrics = monitoring_service.metrics.get_recent_metrics(metric_name, 300)  # Last 5 minutes
        
        if not recent_metrics:
            raise HTTPException(status_code=404, detail=f"Metric '{metric_name}' not found")
        
        # Get statistics based on metric type
        counter_value = monitoring_service.metrics.get_counter(metric_name)
        gauge_value = monitoring_service.metrics.get_gauge(metric_name)
        histogram_stats = monitoring_service.metrics.get_histogram_stats(metric_name)
        timer_stats = monitoring_service.metrics.get_timer_stats(metric_name)
        
        monitoring_service.metrics.increment_counter("specific_metric.success")
        
        return {
            "metric_name": metric_name,
            "timestamp": time.time(),
            "recent_count": len(recent_metrics),
            "counter_value": counter_value,
            "gauge_value": gauge_value,
            "histogram_stats": histogram_stats,
            "timer_stats": timer_stats,
            "recent_metrics": [
                {
                    "value": m.value,
                    "timestamp": m.timestamp,
                    "type": m.metric_type.value,
                    "labels": m.labels
                }
                for m in recent_metrics[-10:]  # Last 10 data points
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        monitoring_service.metrics.increment_counter("specific_metric.errors")
        monitoring_service.logger.error("Specific metric retrieval failed", metric_name=metric_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Metric retrieval error: {str(e)}")


@app.get("/alerts")
@async_timer_decorator("alerts_endpoint")
async def get_alerts():
    """Get current alerts."""
    monitoring_service.metrics.increment_counter("alerts.requests")
    
    try:
        active_alerts = monitoring_service.alert_manager.get_active_alerts()
        all_alerts = monitoring_service.alert_manager.get_all_alerts()
        
        monitoring_service.metrics.increment_counter("alerts.success")
        
        return {
            "timestamp": time.time(),
            "active_alerts": [
                {
                    "id": alert.id,
                    "severity": alert.severity,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "service": alert.service,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "current_value": alert.current_value,
                    "resolved": alert.resolved
                }
                for alert in active_alerts
            ],
            "total_alerts": len(all_alerts),
            "active_count": len(active_alerts)
        }
    except Exception as e:
        monitoring_service.metrics.increment_counter("alerts.errors")
        monitoring_service.logger.error("Alerts retrieval failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Alerts error: {str(e)}")


@app.post("/alerts/rules")
@async_timer_decorator("add_alert_rule")
async def add_alert_rule(
    metric_name: str,
    threshold: float,
    comparison: str = "greater_than",
    severity: str = "warning",
    message_template: str = None
):
    """Add a new alert rule."""
    monitoring_service.metrics.increment_counter("add_alert_rule.requests")
    
    try:
        monitoring_service.alert_manager.add_alert_rule(
            metric_name=metric_name,
            threshold=threshold,
            comparison=comparison,
            severity=severity,
            message_template=message_template
        )
        
        monitoring_service.metrics.increment_counter("add_alert_rule.success")
        monitoring_service.logger.info("Alert rule added", 
                                     metric_name=metric_name, 
                                     threshold=threshold, 
                                     severity=severity)
        
        return {
            "message": f"Alert rule added for {metric_name}",
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity
        }
    except Exception as e:
        monitoring_service.metrics.increment_counter("add_alert_rule.errors")
        monitoring_service.logger.error("Failed to add alert rule", 
                                       metric_name=metric_name, 
                                       error=str(e))
        raise HTTPException(status_code=500, detail=f"Alert rule error: {str(e)}")


# Add middleware for request monitoring
@app.middleware("http")
async def monitoring_middleware(request: Request, call_next):
    """Middleware to monitor all HTTP requests."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Set request context for logging
    monitoring_service.logger.set_request_context(request_id)
    
    # Log request start
    monitoring_service.logger.info("Request started", 
                                 method=request.method, 
                                 url=str(request.url),
                                 request_id=request_id)
    
    # Increment request counter
    monitoring_service.metrics.increment_counter("http.requests.total", 
                                                labels={"method": request.method, "endpoint": request.url.path})
    
    try:
        response = await call_next(request)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Record metrics
        monitoring_service.metrics.record_timer("http.request.duration", response_time,
                                               labels={"method": request.method, "status": str(response.status_code)})
        
        monitoring_service.metrics.increment_counter("http.responses.total",
                                                   labels={"method": request.method, "status": str(response.status_code)})
        
        # Check for slow requests
        if response_time > 2.0:  # Alert on requests slower than 2 seconds
            monitoring_service.metrics.increment_counter("http.requests.slow")
            monitoring_service.logger.warning("Slow request detected",
                                             method=request.method,
                                             url=str(request.url),
                                             response_time=response_time,
                                             status_code=response.status_code)
        
        # Log successful request
        monitoring_service.logger.info("Request completed",
                                     method=request.method,
                                     url=str(request.url),
                                     status_code=response.status_code,
                                     response_time=response_time,
                                     request_id=request_id)
        
        return response
        
    except Exception as e:
        # Calculate response time for failed requests
        response_time = time.time() - start_time
        
        # Record error metrics
        monitoring_service.metrics.increment_counter("http.requests.errors",
                                                   labels={"method": request.method, "error_type": type(e).__name__})
        
        monitoring_service.metrics.record_timer("http.request.duration.error", response_time,
                                               labels={"method": request.method, "error_type": type(e).__name__})
        
        # Log error
        monitoring_service.logger.error("Request failed",
                                       method=request.method,
                                       url=str(request.url),
                                       error=str(e),
                                       error_type=type(e).__name__,
                                       response_time=response_time,
                                       request_id=request_id)
        
        raise


# Placeholder endpoints - will be implemented in subsequent tasks
@app.post("/parse-problem")
async def parse_problem(problem_text: str, domain: str = None):
    """Parse a mathematical problem into structured format."""
    from error_handling import fallback_manager, async_error_handler, ErrorCategory, ErrorSeverity
    
    @async_error_handler(ErrorCategory.PARSING, ErrorSeverity.MEDIUM)
    async def _parse_with_fallback():
        from parser import MathExpressionParser
        
        parser = MathExpressionParser()
        parsed_problem = parser.parse_problem(problem_text, domain)
        
        return {
            "id": parsed_problem.id,
            "original_text": parsed_problem.original_text,
            "domain": parsed_problem.domain.value,
            "difficulty": parsed_problem.difficulty.value,
            "variables": parsed_problem.variables,
            "expressions": parsed_problem.expressions,
            "problem_type": parsed_problem.problem_type,
            "metadata": parsed_problem.metadata
        }
    
    try:
        return await _parse_with_fallback()
    except Exception as e:
        # Use fallback parsing if primary fails
        fallback_result = fallback_manager.execute_with_fallback(
            'parsing', lambda: fallback_manager._simplified_parsing(problem_text, domain)
        )
        
        if fallback_result.success:
            parsed_problem = fallback_result.result
            return {
                "id": parsed_problem.id,
                "original_text": parsed_problem.original_text,
                "domain": parsed_problem.domain.value,
                "difficulty": parsed_problem.difficulty.value,
                "variables": parsed_problem.variables,
                "expressions": parsed_problem.expressions,
                "problem_type": parsed_problem.problem_type,
                "metadata": parsed_problem.metadata,
                "fallback_used": True,
                "confidence_score": fallback_result.confidence_score
            }
        else:
            raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")


@app.post("/solve-step-by-step")
async def solve_step_by_step(problem: dict):
    """Generate step-by-step solution for a mathematical problem."""
    try:
        from solver import MathSolver
        from models import ParsedProblem, MathDomain, DifficultyLevel
        
        # Check cache first
        problem_text = problem.get('original_text', '')
        domain = problem.get('domain', 'algebra')
        
        cached_solution = await problem_cache.get_solution(problem_text, domain)
        if cached_solution:
            return cached_solution
        
        # Convert dict to ParsedProblem object
        parsed_problem = ParsedProblem(
            id=problem.get('id', ''),
            original_text=problem_text,
            domain=MathDomain(domain),
            difficulty=DifficultyLevel(problem.get('difficulty', 2)),
            variables=problem.get('variables', []),
            expressions=problem.get('expressions', []),
            problem_type=problem.get('problem_type', 'general_problem'),
            metadata=problem.get('metadata', {})
        )
        
        solver = MathSolver()
        solution = solver.solve_step_by_step(parsed_problem)
        
        # Convert to dict for JSON response
        solution_dict = {
            "problem_id": solution.problem_id,
            "steps": [
                {
                    "step_number": step.step_number,
                    "operation": step.operation,
                    "explanation": step.explanation,
                    "mathematical_expression": step.mathematical_expression,
                    "intermediate_result": step.intermediate_result,
                    "reasoning": step.reasoning
                }
                for step in solution.steps
            ],
            "final_answer": solution.final_answer,
            "solution_method": solution.solution_method,
            "confidence_score": solution.confidence_score,
            "computation_time": solution.computation_time
        }
        
        # Cache the solution
        await problem_cache.cache_solution(problem_text, domain, solution_dict)
        
        return solution_dict
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Solver error: {str(e)}")


@app.post("/validate-answer")
async def validate_answer(problem: dict, answer: str):
    """Validate a user's answer against the correct solution."""
    try:
        from solver import MathSolver
        from models import ParsedProblem, MathDomain, DifficultyLevel
        
        # Convert dict to ParsedProblem object
        parsed_problem = ParsedProblem(
            id=problem.get('id', ''),
            original_text=problem.get('original_text', ''),
            domain=MathDomain(problem.get('domain', 'algebra')),
            difficulty=DifficultyLevel(problem.get('difficulty', 2)),
            variables=problem.get('variables', []),
            expressions=problem.get('expressions', []),
            problem_type=problem.get('problem_type', 'general_problem'),
            metadata=problem.get('metadata', {})
        )
        
        solver = MathSolver()
        result = solver.validate_answer(parsed_problem, answer)
        
        # Convert to dict for JSON response
        return {
            "is_correct": result.is_correct,
            "user_answer": result.user_answer,
            "correct_answer": result.correct_answer,
            "explanation": result.explanation,
            "partial_credit": result.partial_credit
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")


@app.post("/generate-visualization")
async def generate_visualization(problem: dict, viz_type: str = "auto"):
    """Generate mathematical visualization for a problem."""
    from error_handling import fallback_manager, async_error_handler, ErrorCategory, ErrorSeverity, graceful_degradation
    
    @async_error_handler(ErrorCategory.VISUALIZATION, ErrorSeverity.MEDIUM)
    async def _generate_with_fallback():
        from visualization import VisualizationEngine
        from parser import MathExpressionParser
        from solver import MathSolver
        
        problem_text = problem.get("problem_text", "")
        if not problem_text:
            raise HTTPException(status_code=400, detail="Problem text is required")
        
        # Check cache first
        cached_viz = await visualization_cache.get_visualization(problem, viz_type)
        if cached_viz:
            return cached_viz
        
        # Parse the problem
        parser = MathExpressionParser()
        solver = MathSolver()
        viz_engine = VisualizationEngine()
        
        # Parse and solve the problem
        parsed_problem = parser.parse_problem(problem_text)
        solution = solver.solve_step_by_step(parsed_problem)
        
        # Generate visualization
        plot_data = viz_engine.generate_problem_visualization(parsed_problem, solution)
        
        # Render the plot to base64 image
        image_base64 = viz_engine.plotter.render_plot(plot_data)
        
        viz_result = {
            "plot_data": {
                "plot_type": plot_data.plot_type,
                "title": plot_data.title,
                "axis_labels": plot_data.axis_labels,
                "data_points_count": len(plot_data.data_points),
                "interactive_elements_count": len(plot_data.interactive_elements)
            },
            "image_base64": image_base64,
            "problem_type": parsed_problem.problem_type,
            "solution_method": solution.solution_method
        }
        
        # Cache the visualization
        await visualization_cache.cache_visualization(problem, viz_type, viz_result)
        
        return viz_result
    
    try:
        return await _generate_with_fallback()
    except Exception as e:
        # Use fallback visualization if primary fails
        if graceful_degradation.is_service_available('visualization'):
            graceful_degradation.mark_service_down('visualization')
        
        fallback_result = fallback_manager.execute_with_fallback(
            'visualization', lambda: fallback_manager._simple_plot_fallback()
        )
        
        if fallback_result.success:
            result = fallback_result.result
            result['fallback_used'] = True
            result['confidence_score'] = fallback_result.confidence_score
            result['warnings'] = fallback_result.warnings
            return result
        else:
            raise HTTPException(status_code=500, detail=f"Visualization error: {str(e)}")


@app.post("/explain-step")
async def explain_step(step: dict, user_level: str = "standard"):
    """Generate AI explanation for a solution step."""
    try:
        # Check cache first
        cached_explanation = await ai_explanation_cache.get_explanation(step, user_level)
        if cached_explanation:
            return cached_explanation
        
        # If AI explainer is implemented, use it; otherwise return placeholder
        try:
            from ai_explainer import AIExplainer
            explainer = AIExplainer()
            explanation = explainer.explain_step(step, user_level)
            
            explanation_dict = {
                "content": explanation.content,
                "complexity_level": explanation.complexity_level,
                "related_concepts": explanation.related_concepts,
                "examples": explanation.examples
            }
            
            # Cache the explanation
            await ai_explanation_cache.cache_explanation(step, user_level, explanation_dict)
            
            return explanation_dict
            
        except ImportError:
            # Fallback explanation if AI explainer not implemented
            explanation_dict = {
                "content": f"This step involves {step.get('operation', 'a mathematical operation')}. The expression {step.get('mathematical_expression', '')} results in {step.get('intermediate_result', '')}.",
                "complexity_level": user_level,
                "related_concepts": [],
                "examples": []
            }
            
            # Cache the fallback explanation
            await ai_explanation_cache.cache_explanation(step, user_level, explanation_dict)
            
            return explanation_dict
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explanation error: {str(e)}")


@app.post("/generate-hint")
async def generate_hint(problem: dict, current_step: int = 0):
    """Generate contextual hint for problem solving."""
    # TODO: Implement in task 4.2
    raise HTTPException(status_code=501, detail="Not implemented yet")


# Quiz Generation and Assessment Endpoints

@app.post("/generate-quiz")
async def generate_quiz(topic: str, difficulty: int = 2, num_questions: int = 10, 
                       question_type: str = 'mixed'):
    """Generate a new quiz based on specified parameters."""
    try:
        from quiz_generator import QuizGenerator
        from models import DifficultyLevel
        
        # Convert difficulty integer to enum
        difficulty_map = {1: DifficultyLevel.BEGINNER, 2: DifficultyLevel.INTERMEDIATE, 
                         3: DifficultyLevel.ADVANCED, 4: DifficultyLevel.EXPERT}
        difficulty_level = difficulty_map.get(difficulty, DifficultyLevel.INTERMEDIATE)
        
        generator = QuizGenerator()
        quiz = generator.generate_quiz(topic, difficulty_level, num_questions, question_type)
        
        # Convert to dict for JSON response
        return {
            "id": quiz.id,
            "title": quiz.title,
            "topic": quiz.topic,
            "difficulty": quiz.difficulty.value,
            "time_limit": quiz.time_limit,
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "type": q.question_type.value,
                    "options": q.options,
                    "hints": q.hints,
                    "topic": q.topic
                }
                for q in quiz.questions
            ],
            "created_at": quiz.created_at.isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Quiz generation error: {str(e)}")


@app.post("/start-quiz-session")
async def start_quiz_session(user_id: str, quiz_data: dict):
    """Start a new quiz session."""
    try:
        from quiz_session_manager import QuizSessionManager
        from models import Quiz, Question, QuestionType, DifficultyLevel
        from datetime import datetime
        
        # Convert quiz_data back to Quiz object
        questions = []
        for q_data in quiz_data.get('questions', []):
            question = Question(
                id=q_data['id'],
                text=q_data['text'],
                question_type=QuestionType(q_data['type']),
                options=q_data.get('options', []),
                correct_answer=q_data.get('correct_answer', ''),  # This would come from secure storage
                hints=q_data.get('hints', []),
                difficulty=DifficultyLevel(quiz_data.get('difficulty', 2)),
                topic=q_data.get('topic', 'general')
            )
            questions.append(question)
        
        quiz = Quiz(
            id=quiz_data['id'],
            title=quiz_data['title'],
            questions=questions,
            time_limit=quiz_data.get('time_limit'),
            topic=quiz_data['topic'],
            difficulty=DifficultyLevel(quiz_data['difficulty']),
            created_at=datetime.fromisoformat(quiz_data['created_at'])
        )
        
        session_manager = QuizSessionManager()
        session_id = session_manager.start_quiz_session(user_id, quiz)
        
        return {
            "session_id": session_id,
            "status": "started",
            "current_question": session_manager.get_current_question(session_id).__dict__ if session_manager.get_current_question(session_id) else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Session start error: {str(e)}")


@app.get("/quiz-session/{session_id}/current-question")
async def get_current_question(session_id: str):
    """Get the current question for a quiz session."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        current_question = session_manager.get_current_question(session_id)
        
        if not current_question:
            return {"question": None, "message": "No current question or session completed"}
        
        return {
            "id": current_question.id,
            "text": current_question.text,
            "type": current_question.question_type.value,
            "options": current_question.options,
            "hints": current_question.hints,
            "topic": current_question.topic
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error getting current question: {str(e)}")


@app.post("/quiz-session/{session_id}/submit-answer")
async def submit_answer(session_id: str, question_id: str, user_answer: str, time_taken: int = 0):
    """Submit an answer for the current question."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        result = session_manager.submit_answer(session_id, question_id, user_answer, time_taken)
        
        return {
            "question_id": result.question_id,
            "is_correct": result.is_correct,
            "user_answer": result.user_answer,
            "correct_answer": result.correct_answer,
            "explanation": result.explanation,
            "points_earned": result.points_earned,
            "time_taken": result.time_taken
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Answer submission error: {str(e)}")


@app.post("/quiz-session/{session_id}/use-hint")
async def use_hint(session_id: str, question_id: str, hint_level: int = 1):
    """Use a hint for the current question."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        hint = session_manager.use_hint(session_id, question_id, hint_level)
        
        return {
            "hint": hint,
            "hint_level": hint_level,
            "question_id": question_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Hint error: {str(e)}")


@app.get("/quiz-session/{session_id}/progress")
async def get_quiz_progress(session_id: str):
    """Get current progress of a quiz session."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        progress = session_manager.get_quiz_progress(session_id)
        
        return progress
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Progress error: {str(e)}")


@app.post("/quiz-session/{session_id}/complete")
async def complete_quiz_session(session_id: str):
    """Complete a quiz session and get final results."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        results = session_manager.complete_quiz_session(session_id)
        
        return {
            "quiz_id": results.quiz_id,
            "user_id": results.user_id,
            "total_questions": results.total_questions,
            "correct_answers": results.correct_answers,
            "total_points": results.total_points,
            "points_earned": results.points_earned,
            "time_taken": results.time_taken,
            "completion_rate": results.completion_rate,
            "areas_for_improvement": results.areas_for_improvement
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Completion error: {str(e)}")


@app.get("/user/{user_id}/performance")
async def get_user_performance(user_id: str):
    """Get comprehensive performance metrics for a user."""
    try:
        from quiz_session_manager import QuizSessionManager
        
        session_manager = QuizSessionManager()
        performance = session_manager.get_user_performance_metrics(user_id)
        
        return {
            "accuracy": performance.accuracy,
            "average_time": performance.average_time,
            "streak": performance.streak,
            "topics_mastered": performance.topics_mastered,
            "areas_needing_work": performance.areas_needing_work
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Performance error: {str(e)}")


@app.post("/validate-answer")
async def validate_mathematical_answer(user_answer: str, correct_answer: str, 
                                     problem_type: str = 'general'):
    """Validate a mathematical answer using symbolic computation."""
    try:
        from answer_validator import AnswerValidator
        
        validator = AnswerValidator()
        result = validator.validate_answer(user_answer, correct_answer, problem_type)
        
        return {
            "is_correct": result.is_correct,
            "user_answer": result.user_answer,
            "correct_answer": result.correct_answer,
            "explanation": result.explanation,
            "partial_credit": result.partial_credit
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")


@app.post("/generate-similar-problems")
async def generate_similar_problems(original_problem: str, count: int = 5):
    """Generate similar problems based on an original problem."""
    try:
        from quiz_generator import QuizGenerator
        
        generator = QuizGenerator()
        similar_problems = generator.generate_similar_problems(original_problem, count)
        
        return {
            "original_problem": original_problem,
            "similar_problems": similar_problems,
            "count": len(similar_problems)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Similar problems error: {str(e)}")


# Cache Management and Monitoring Endpoints

@app.get("/cache/metrics")
async def get_cache_metrics():
    """Get cache performance metrics."""
    try:
        metrics = await cache_service.get_metrics()
        return {
            "hit_count": metrics.hit_count,
            "miss_count": metrics.miss_count,
            "total_requests": metrics.total_requests,
            "hit_rate": metrics.hit_rate,
            "miss_rate": metrics.miss_rate,
            "average_response_time": metrics.average_response_time,
            "cache_size": metrics.cache_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache metrics error: {str(e)}")


@app.post("/cache/invalidate")
async def invalidate_cache(cache_type: str = None, pattern: str = None):
    """Invalidate cache entries by type or pattern."""
    try:
        if pattern:
            deleted_count = await cache_service.invalidate_pattern(pattern)
            return {"message": f"Invalidated {deleted_count} cache entries matching pattern: {pattern}"}
        elif cache_type:
            pattern = f"{cache_type}:*"
            deleted_count = await cache_service.invalidate_pattern(pattern)
            return {"message": f"Invalidated {deleted_count} cache entries of type: {cache_type}"}
        else:
            raise HTTPException(status_code=400, detail="Either cache_type or pattern must be provided")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache invalidation error: {str(e)}")


@app.post("/cache/invalidate-user/{user_id}")
async def invalidate_user_cache(user_id: str):
    """Invalidate all cached data for a specific user."""
    try:
        # Invalidate user-specific solutions
        solution_count = await problem_cache.invalidate_user_solutions(user_id)
        
        # Invalidate user session data
        session_pattern = f"user_session:*{user_id}*"
        session_count = await cache_service.invalidate_pattern(session_pattern)
        
        # Invalidate user progress data
        progress_pattern = f"user_progress:*{user_id}*"
        progress_count = await cache_service.invalidate_pattern(progress_pattern)
        
        total_deleted = solution_count + session_count + progress_count
        
        return {
            "message": f"Invalidated {total_deleted} cache entries for user {user_id}",
            "details": {
                "solutions": solution_count,
                "sessions": session_count,
                "progress": progress_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User cache invalidation error: {str(e)}")


@app.get("/cache/health")
async def cache_health_check():
    """Check cache service health and connectivity."""
    try:
        if not cache_service.redis_client:
            await cache_service.connect()
        
        # Test Redis connectivity
        await cache_service.redis_client.ping()
        
        # Get basic Redis info
        info = await cache_service.redis_client.info()
        
        return {
            "status": "healthy",
            "redis_version": info.get("redis_version", "unknown"),
            "connected_clients": info.get("connected_clients", 0),
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0)
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Database Performance and Monitoring Endpoints

@app.get("/database/health")
async def database_health_check():
    """Check database connection health and pool status."""
    try:
        # Test database connectivity
        result = await db_pool.fetchval("SELECT 1")
        
        # Get pool statistics
        pool_stats = db_pool.get_pool_stats()
        
        return {
            "status": "healthy",
            "connection_test": result == 1,
            "pool_stats": pool_stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/database/metrics")
async def get_database_metrics():
    """Get database query performance metrics."""
    try:
        query_metrics = db_pool.get_query_metrics(limit=20)
        slow_queries = db_pool.get_slow_queries()
        pool_stats = db_pool.get_pool_stats()
        
        return {
            "pool_stats": pool_stats,
            "query_metrics": query_metrics,
            "slow_queries": slow_queries,
            "total_unique_queries": len(db_pool.query_metrics)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database metrics error: {str(e)}")


@app.post("/database/optimize")
async def optimize_database():
    """Run database optimization analysis and maintenance."""
    try:
        optimization_result = await db_pool.optimize_queries()
        return {
            "status": "completed",
            "optimization_result": optimization_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database optimization error: {str(e)}")


@app.get("/analytics/user-performance/{user_id}")
async def get_user_performance_analytics(user_id: str):
    """Get comprehensive user performance analytics using optimized queries."""
    try:
        # Get user performance summary
        performance_summary = await optimized_queries.get_user_performance_summary(user_id)
        
        # Get learning recommendations
        recommendations = await optimized_queries.get_user_learning_recommendations(user_id)
        
        # Get progress by topic
        topic_progress = await optimized_queries.get_user_progress_by_topic(user_id)
        
        # Get quiz performance analytics
        quiz_analytics = await optimized_queries.get_quiz_performance_analytics(user_id)
        
        return {
            "user_id": user_id,
            "performance_summary": performance_summary,
            "learning_recommendations": recommendations,
            "topic_progress": topic_progress,
            "quiz_analytics": quiz_analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"User analytics error: {str(e)}")


@app.get("/analytics/problem-recommendations/{user_id}")
async def get_problem_recommendations_for_user(user_id: str, domain: str = None, limit: int = 10):
    """Get personalized problem recommendations for a user."""
    try:
        recommendations = await optimized_queries.get_problem_recommendations(user_id, domain, limit)
        
        return {
            "user_id": user_id,
            "domain": domain,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem recommendations error: {str(e)}")


@app.get("/analytics/problem-difficulty")
async def get_problem_difficulty_insights(domain: str = None):
    """Get insights into problem difficulty and success rates."""
    try:
        insights = await optimized_queries.get_problem_difficulty_insights(domain)
        
        return {
            "domain": domain,
            "insights": insights,
            "count": len(insights)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem difficulty insights error: {str(e)}")


@app.post("/analytics/update-user-progress")
async def batch_update_user_progress(progress_updates: list):
    """Batch update user progress data efficiently."""
    try:
        updated_count = await optimized_queries.batch_update_user_progress(progress_updates)
        
        return {
            "status": "completed",
            "updated_count": updated_count,
            "total_updates": len(progress_updates)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Progress update error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)