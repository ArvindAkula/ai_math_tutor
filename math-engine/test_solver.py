"""
Unit tests for the mathematical problem solver.
Tests step-by-step solution generation and answer validation.
"""

import pytest
import sys
import os

# Add shared models to path
from models import ParsedProblem, MathDomain, DifficultyLevel, ComputationError
from parser import MathExpressionParser
from solver import MathSolver


class TestMathSolver:
    """Test cases for the MathSolver class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MathExpressionParser()
        self.solver = MathSolver()
    
    def test_solve_simple_linear_equation(self):
        """Test solving a simple linear equation: 2x + 3 = 7"""
        problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 3  # Should have multiple steps
        assert solution.final_answer == "2"
        assert solution.solution_method == "Linear equation solving"
        assert solution.confidence_score >= 0.9
        assert solution.computation_time > 0
        
        # Check that steps are properly ordered
        for i, step in enumerate(solution.steps):
            assert step.step_number == i + 1
            assert step.operation
            assert step.explanation
            assert step.mathematical_expression
            assert step.intermediate_result
    
    def test_solve_quadratic_equation(self):
        """Test solving a quadratic equation: x^2 - 5x + 6 = 0"""
        problem = self.parser.parse_problem("Solve the quadratic equation: x^2 - 5x + 6 = 0")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 4  # Should have multiple steps
        assert "2" in solution.final_answer and "3" in solution.final_answer  # Roots are 2 and 3
        assert solution.solution_method == "Quadratic formula method"
        assert solution.confidence_score >= 0.9
        
        # Check for key steps
        step_operations = [step.operation for step in solution.steps]
        assert any("quadratic" in op.lower() for op in step_operations)
        assert any("discriminant" in op.lower() for op in step_operations)
    
    def test_solve_factoring_problem(self):
        """Test factoring: Factor x^2 - 4"""
        problem = self.parser.parse_problem("Factor x^2 - 4")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert "(x - 2)" in solution.final_answer and "(x + 2)" in solution.final_answer
        assert solution.solution_method == "Factoring"
        assert solution.confidence_score >= 0.8
    
    def test_solve_expansion_problem(self):
        """Test expansion: Expand (x + 2)(x - 3)"""
        problem = self.parser.parse_problem("Expand (x + 2)(x - 3)")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert "x**2" in solution.final_answer or "x^2" in solution.final_answer
        assert solution.solution_method == "Expansion"
        assert solution.confidence_score >= 0.8
    
    def test_solve_simplification_problem(self):
        """Test simplification: Simplify 2x + 3x"""
        problem = self.parser.parse_problem("Simplify 2x + 3x")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert "5*x" in solution.final_answer or "5x" in solution.final_answer
        assert solution.solution_method == "Simplification"
        assert solution.confidence_score >= 0.8
    
    def test_solve_system_of_equations(self):
        """Test system of equations: 2x + y = 1, x - y = 3"""
        problem = self.parser.parse_problem("Solve the system of equations: 2x + y = 1, x - y = 3")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert "x" in solution.final_answer and "y" in solution.final_answer
        assert solution.solution_method == "System solving"
        assert solution.confidence_score >= 0.8
    
    def test_validate_correct_answer(self):
        """Test answer validation with correct answer."""
        problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        result = self.solver.validate_answer(problem, "2")
        
        assert result.is_correct == True
        assert result.user_answer == "2"
        assert result.correct_answer == "2"
        assert result.partial_credit == 1.0
        assert "Correct" in result.explanation
    
    def test_validate_incorrect_answer(self):
        """Test answer validation with incorrect answer."""
        problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        result = self.solver.validate_answer(problem, "5")
        
        assert result.is_correct == False
        assert result.user_answer == "5"
        assert result.correct_answer == "2"
        assert result.partial_credit < 1.0
        assert "Incorrect" in result.explanation
    
    def test_validate_equivalent_answer(self):
        """Test answer validation with mathematically equivalent answer."""
        problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        result = self.solver.validate_answer(problem, "4/2")  # Equivalent to 2
        
        assert result.is_correct == True
        assert result.partial_credit == 1.0
        assert "Correct" in result.explanation
    
    def test_solve_complex_linear_equation(self):
        """Test solving a more complex linear equation: 3(x - 2) + 5 = 2x + 7"""
        problem = self.parser.parse_problem("Solve for x: 3(x - 2) + 5 = 2x + 7")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert solution.final_answer == "8"
        assert solution.confidence_score >= 0.8
    
    def test_solve_quadratic_with_no_real_solutions(self):
        """Test quadratic equation with no real solutions: x^2 + 1 = 0"""
        problem = self.parser.parse_problem("Solve: x^2 + 1 = 0")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 3
        assert "No real solutions" in solution.final_answer or "Complex" in solution.final_answer
        assert solution.confidence_score >= 0.8
    
    def test_solve_quadratic_with_one_solution(self):
        """Test quadratic equation with one solution: x^2 - 4x + 4 = 0"""
        problem = self.parser.parse_problem("Solve: x^2 - 4x + 4 = 0")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 4
        assert solution.final_answer == "2"  # Double root at x = 2
        assert solution.confidence_score >= 0.9
    
    def test_error_handling_invalid_equation(self):
        """Test error handling for invalid equations."""
        # Create a problem with no valid equation
        problem = ParsedProblem(
            id="test-id",
            original_text="This is not a math problem",
            domain=MathDomain.ALGEBRA,
            difficulty=DifficultyLevel.BEGINNER,
            variables=[],
            expressions=[],
            problem_type="linear_equation",
            metadata={}
        )
        
        with pytest.raises(ComputationError):
            self.solver.solve_step_by_step(problem)
    
    def test_step_solution_structure(self):
        """Test that solution steps have proper structure."""
        problem = self.parser.parse_problem("Solve for x: x + 5 = 10")
        solution = self.solver.solve_step_by_step(problem)
        
        for step in solution.steps:
            assert isinstance(step.step_number, int)
            assert step.step_number > 0
            assert isinstance(step.operation, str)
            assert len(step.operation) > 0
            assert isinstance(step.explanation, str)
            assert len(step.explanation) > 0
            assert isinstance(step.mathematical_expression, str)
            assert len(step.mathematical_expression) > 0
            assert isinstance(step.intermediate_result, str)
            assert len(step.intermediate_result) > 0
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation for different problem types."""
        # Linear equation should have high confidence
        linear_problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        linear_solution = self.solver.solve_step_by_step(linear_problem)
        assert linear_solution.confidence_score >= 0.9
        
        # General problem should have lower confidence
        general_problem = self.parser.parse_problem("This is a general math problem")
        general_solution = self.solver.solve_step_by_step(general_problem)
        assert general_solution.confidence_score <= 0.8
    
    def test_computation_time_tracking(self):
        """Test that computation time is properly tracked."""
        problem = self.parser.parse_problem("Solve for x: 2x + 3 = 7")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.computation_time > 0
        assert solution.computation_time < 1.0  # Should be fast for simple problems
    
    def test_multiple_variable_system(self):
        """Test system with multiple variables."""
        problem = self.parser.parse_problem("Solve: x + y = 5, 2x - y = 1")
        solution = self.solver.solve_step_by_step(problem)
        
        assert solution.problem_id == problem.id
        assert len(solution.steps) >= 2
        assert "x" in solution.final_answer
        assert "y" in solution.final_answer
        assert solution.confidence_score >= 0.8


class TestSolverIntegration:
    """Integration tests for the solver with various problem types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MathExpressionParser()
        self.solver = MathSolver()
    
    def test_real_world_algebra_problems(self):
        """Test with real algebra problems students might encounter."""
        problems = [
            ("Solve for x: 3x - 7 = 2x + 5", "12"),
            ("Solve: 5x + 10 = 0", "-2"),
            ("Find x: 4x - 8 = 2x + 6", "7"),
        ]
        
        for problem_text, expected_answer in problems:
            problem = self.parser.parse_problem(problem_text)
            solution = self.solver.solve_step_by_step(problem)
            
            assert solution.final_answer == expected_answer
            assert len(solution.steps) >= 2
            assert solution.confidence_score >= 0.8
    
    def test_quadratic_problems_variety(self):
        """Test various types of quadratic problems."""
        problems = [
            "Solve: x^2 - 9 = 0",  # Difference of squares
            "Solve: x^2 + 6x + 9 = 0",  # Perfect square
            "Solve: 2x^2 - 8x + 6 = 0",  # Factorable
        ]
        
        for problem_text in problems:
            problem = self.parser.parse_problem(problem_text)
            solution = self.solver.solve_step_by_step(problem)
            
            assert len(solution.steps) >= 4
            assert solution.solution_method == "Quadratic formula method"
            assert solution.confidence_score >= 0.8
    
    def test_factoring_problems_variety(self):
        """Test various factoring problems."""
        problems = [
            "Factor: x^2 - 16",  # Difference of squares
            "Factor: x^2 + 5x + 6",  # Trinomial
            "Factor: 2x^2 + 4x",  # Common factor
        ]
        
        for problem_text in problems:
            problem = self.parser.parse_problem(problem_text)
            solution = self.solver.solve_step_by_step(problem)
            
            assert len(solution.steps) >= 2
            assert solution.solution_method == "Factoring"
            assert solution.confidence_score >= 0.8
    
    def test_answer_validation_edge_cases(self):
        """Test answer validation with various edge cases."""
        problem = self.parser.parse_problem("Solve for x: 2x = 8")
        
        # Test various equivalent forms
        equivalent_answers = ["4", "8/2", "2*2", "16/4"]
        for answer in equivalent_answers:
            result = self.solver.validate_answer(problem, answer)
            assert result.is_correct == True
        
        # Test incorrect answers
        incorrect_answers = ["2", "8", "0", "-4"]
        for answer in incorrect_answers:
            result = self.solver.validate_answer(problem, answer)
            assert result.is_correct == False
    
    def test_performance_with_complex_problems(self):
        """Test solver performance with more complex problems."""
        complex_problems = [
            "Solve: (x + 1)(x - 2) = x^2 - 5x + 6",
            "Solve the system: 3x + 2y = 12, x - y = 1",
            "Factor: x^3 - 8",
        ]
        
        for problem_text in complex_problems:
            problem = self.parser.parse_problem(problem_text)
            solution = self.solver.solve_step_by_step(problem)
            
            # Should complete within reasonable time
            assert solution.computation_time < 2.0
            assert len(solution.steps) >= 2
            assert solution.confidence_score >= 0.5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])