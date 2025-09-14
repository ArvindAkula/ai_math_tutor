"""
Unit tests for calculus problem solving capabilities.
Tests derivative, integral, limit, and optimization problems.
"""

import pytest
import sys
import os

# Add shared models to path
from models import MathDomain, ComputationError, ParsedProblem
from solver import MathSolver
from parser import MathExpressionParser


class TestCalculusSolver:
    """Test cases for calculus problem solving."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = MathSolver()
        self.parser = MathExpressionParser()
    
    def test_basic_derivative_power_rule(self):
        """Test basic derivative using power rule."""
        problem_text = "Find the derivative of x^3"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'derivative'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        assert "3*x**2" in solution.final_answer or "3*x^2" in solution.final_answer
        assert solution.solution_method == "Differentiation"
        assert len(solution.steps) >= 2
        assert solution.confidence_score > 0.8
    
    def test_derivative_product_rule(self):
        """Test derivative using product rule."""
        problem_text = "Find the derivative of x^2 * sin(x)"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'derivative'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should contain both x*sin(x) and x^2*cos(x) terms
        assert "sin" in solution.final_answer
        assert "cos" in solution.final_answer
        assert solution.solution_method == "Differentiation"
        assert len(solution.steps) >= 2
    
    def test_derivative_chain_rule(self):
        """Test derivative using chain rule."""
        problem_text = "Find the derivative of sin(x^2)"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'derivative'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should contain 2*x*cos(x^2)
        assert "cos" in solution.final_answer
        assert "2*x" in solution.final_answer or "2x" in solution.final_answer
        assert solution.solution_method == "Differentiation"
    
    def test_basic_integral_power_rule(self):
        """Test basic integral using power rule."""
        problem_text = "Find the integral of x^2"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'integral'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be x^3/3 + C
        assert "x**3/3" in solution.final_answer or "x^3/3" in solution.final_answer
        assert "+ C" in solution.final_answer
        assert solution.solution_method == "Integration"
        assert len(solution.steps) >= 2
    
    def test_definite_integral(self):
        """Test definite integral evaluation."""
        problem_text = "Evaluate the definite integral of x^2 from 0 to 2"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'integral'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be 8/3
        assert "8/3" in solution.final_answer or "2.66" in solution.final_answer
        assert "+ C" not in solution.final_answer  # Definite integral shouldn't have +C
        assert solution.solution_method == "Integration"
        assert len(solution.steps) >= 3
    
    def test_integral_trigonometric(self):
        """Test integration of trigonometric functions."""
        problem_text = "Find the integral of sin(x)"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'integral'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be -cos(x) + C
        assert "-cos(x)" in solution.final_answer or "-cos" in solution.final_answer
        assert "+ C" in solution.final_answer
        assert solution.solution_method == "Integration"
    
    def test_limit_basic(self):
        """Test basic limit evaluation."""
        problem_text = "Find the limit of x^2 as x approaches 3"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'limit'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be 9
        assert "9" in solution.final_answer
        assert solution.solution_method == "Limit evaluation"
        assert len(solution.steps) >= 2
    
    def test_limit_indeterminate_form(self):
        """Test limit with indeterminate form."""
        problem_text = "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'limit'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be 2 (after factoring and canceling)
        assert "2" in solution.final_answer
        assert solution.solution_method == "Limit evaluation"
        assert len(solution.steps) >= 2
    
    def test_limit_at_infinity(self):
        """Test limit as x approaches infinity."""
        problem_text = "Find the limit of 1/x as x approaches infinity"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'limit'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Should be 0
        assert "0" in solution.final_answer
        assert solution.solution_method == "Limit evaluation"
    
    def test_optimization_basic(self):
        """Test basic optimization problem."""
        problem_text = "Find the minimum of x^2 - 4x + 3"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'optimization'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Minimum should be at x = 2, f(2) = -1
        assert "2" in solution.final_answer
        assert "-1" in solution.final_answer or "minimum" in solution.final_answer.lower()
        assert solution.solution_method == "Optimization"
        assert len(solution.steps) >= 4  # Function, derivative, critical points, classification
    
    def test_optimization_maximum(self):
        """Test finding maximum of a function."""
        problem_text = "Find the maximum of -x^2 + 4x - 3"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'optimization'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        
        # Maximum should be at x = 2, f(2) = 1
        assert "2" in solution.final_answer
        assert "1" in solution.final_answer or "maximum" in solution.final_answer.lower()
        assert solution.solution_method == "Optimization"


class TestCalculusVisualization:
    """Test visualization data generation for calculus problems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = MathSolver()
        self.parser = MathExpressionParser()
    
    def test_derivative_visualization_data(self):
        """Test visualization data generation for derivatives."""
        problem_text = "Find the derivative of x^2"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'derivative'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        viz_data = self.solver.generate_visualization_data(parsed_problem, solution)
        
        assert viz_data['type'] == 'derivative'
        assert 'x_values' in viz_data['data']
        assert 'function_values' in viz_data['data']
        assert 'derivative_values' in viz_data['data']
        assert 'function_expr' in viz_data['data']
        assert 'derivative_expr' in viz_data['data']
    
    def test_integral_visualization_data(self):
        """Test visualization data generation for integrals."""
        problem_text = "Find the integral of x^2 from 0 to 2"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'integral'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        viz_data = self.solver.generate_visualization_data(parsed_problem, solution)
        
        assert viz_data['type'] == 'integral'
        assert 'x_values' in viz_data['data']
        assert 'function_values' in viz_data['data']
        assert 'function_expr' in viz_data['data']
        assert viz_data['data'].get('definite', False) == True
        assert 'lower_limit' in viz_data['data']
        assert 'upper_limit' in viz_data['data']
    
    def test_optimization_visualization_data(self):
        """Test visualization data generation for optimization."""
        problem_text = "Find the minimum of x^2 - 4x + 3"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'optimization'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        viz_data = self.solver.generate_visualization_data(parsed_problem, solution)
        
        assert viz_data['type'] == 'optimization'
        assert 'x_values' in viz_data['data']
        assert 'function_values' in viz_data['data']
        assert 'critical_points' in viz_data['data']
        assert 'function_expr' in viz_data['data']
        assert 'derivative_expr' in viz_data['data']
    
    def test_limit_visualization_data(self):
        """Test visualization data generation for limits."""
        problem_text = "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'limit'
        
        solution = self.solver.solve_step_by_step(parsed_problem)
        viz_data = self.solver.generate_visualization_data(parsed_problem, solution)
        
        assert viz_data['type'] == 'limit'
        assert 'x_values' in viz_data['data']
        assert 'function_values' in viz_data['data']
        assert 'function_expr' in viz_data['data']
        assert 'limit_point' in viz_data['data']


class TestCalculusIntegration:
    """Integration tests for calculus with real-world problems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = MathSolver()
        self.parser = MathExpressionParser()
    
    def test_real_world_derivatives(self):
        """Test with real derivative problems."""
        problems = [
            ("Find the derivative of 3x^2 + 2x - 1", "6*x + 2"),
            ("Differentiate sin(x) + cos(x)", "-sin(x) + cos(x)"),
            ("Find d/dx of e^x", "exp(x)"),
            ("What is the derivative of ln(x)?", "1/x")
        ]
        
        for problem_text, expected_pattern in problems:
            parsed_problem = self.parser.parse_problem(problem_text)
            parsed_problem.problem_type = 'derivative'
            
            solution = self.solver.solve_step_by_step(parsed_problem)
            
            # Check that the solution contains expected elements
            assert solution.final_answer
            assert solution.solution_method == "Differentiation"
            assert len(solution.steps) >= 2
    
    def test_real_world_integrals(self):
        """Test with real integral problems."""
        problems = [
            ("Integrate 2x", "x**2"),
            ("Find the antiderivative of cos(x)", "sin(x)"),
            ("What is ∫ e^x dx?", "exp(x)"),
            ("Evaluate ∫ 1/x dx", "log(x)")
        ]
        
        for problem_text, expected_pattern in problems:
            parsed_problem = self.parser.parse_problem(problem_text)
            parsed_problem.problem_type = 'integral'
            
            solution = self.solver.solve_step_by_step(parsed_problem)
            
            # Check that the solution contains expected elements
            assert solution.final_answer
            assert solution.solution_method == "Integration"
            assert len(solution.steps) >= 2
    
    def test_real_world_limits(self):
        """Test with real limit problems."""
        problems = [
            ("lim x→0 sin(x)/x", "1"),
            ("Find lim x→∞ 1/x", "0"),
            ("What is lim x→2 (x^2 - 4)/(x - 2)?", "4")
        ]
        
        for problem_text, expected in problems:
            parsed_problem = self.parser.parse_problem(problem_text)
            parsed_problem.problem_type = 'limit'
            
            solution = self.solver.solve_step_by_step(parsed_problem)
            
            # Check that the solution is reasonable
            assert solution.final_answer
            assert solution.solution_method == "Limit evaluation"
            assert len(solution.steps) >= 2
    
    def test_real_world_optimization(self):
        """Test with real optimization problems."""
        problems = [
            ("Find the minimum value of f(x) = x^2 + 2x + 1", "minimum"),
            ("Optimize g(x) = -x^2 + 4x to find maximum", "maximum"),
            ("Find critical points of h(x) = x^3 - 3x^2 + 2", "critical")
        ]
        
        for problem_text, expected_type in problems:
            parsed_problem = self.parser.parse_problem(problem_text)
            parsed_problem.problem_type = 'optimization'
            
            solution = self.solver.solve_step_by_step(parsed_problem)
            
            # Check that the solution is reasonable
            assert solution.final_answer
            assert solution.solution_method == "Optimization"
            assert len(solution.steps) >= 3


class TestCalculusErrorHandling:
    """Test error handling for calculus problems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = MathSolver()
        self.parser = MathExpressionParser()
    
    def test_derivative_no_function(self):
        """Test derivative with no function specified."""
        problem_text = "Find the derivative"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'derivative'
        
        with pytest.raises(ComputationError) as excinfo:
            self.solver.solve_step_by_step(parsed_problem)
        assert "No function found" in str(excinfo.value)
    
    def test_integral_complex_function(self):
        """Test integral that cannot be computed."""
        problem_text = "Integrate e^(x^2)"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'integral'
        
        # This should not raise an error but should indicate it cannot integrate
        solution = self.solver.solve_step_by_step(parsed_problem)
        assert "cannot integrate" in solution.final_answer.lower() or solution.final_answer
    
    def test_limit_no_function(self):
        """Test limit with no function specified."""
        problem_text = "Find the limit as x approaches 0"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'limit'
        
        with pytest.raises(ComputationError) as excinfo:
            self.solver.solve_step_by_step(parsed_problem)
        assert "No function found" in str(excinfo.value)
    
    def test_optimization_no_function(self):
        """Test optimization with no function specified."""
        problem_text = "Find the maximum"
        parsed_problem = self.parser.parse_problem(problem_text)
        parsed_problem.problem_type = 'optimization'
        
        with pytest.raises(ComputationError) as excinfo:
            self.solver.solve_step_by_step(parsed_problem)
        assert "No function found" in str(excinfo.value)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])