"""
Unit tests for the mathematical expression parser.
Tests parsing of various mathematical notation formats and domain identification.
"""

import pytest
import sys
import os

# Add shared models to path
from models import MathDomain, DifficultyLevel, ParseError
from parser import MathExpressionParser


class TestMathExpressionParser:
    """Test cases for the MathExpressionParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MathExpressionParser()
    
    def test_parse_simple_linear_equation(self):
        """Test parsing a simple linear equation."""
        problem_text = "Solve for x: 2x + 3 = 7"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.ALGEBRA
        assert result.difficulty == DifficultyLevel.BEGINNER
        assert 'x' in result.variables
        assert result.problem_type == 'linear_equation'
        assert '2x + 3 = 7' in result.expressions
        assert result.original_text == problem_text
    
    def test_parse_quadratic_equation(self):
        """Test parsing a quadratic equation."""
        problem_text = "Solve the quadratic equation: x^2 - 5x + 6 = 0"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.ALGEBRA
        assert result.difficulty == DifficultyLevel.INTERMEDIATE
        assert 'x' in result.variables
        assert result.problem_type == 'quadratic_equation'
        assert result.metadata['has_equations'] == True
        assert result.metadata['contains_exponents'] == True
    
    def test_parse_derivative_problem(self):
        """Test parsing a calculus derivative problem."""
        problem_text = "Find the derivative of f(x) = x^2 + 3x"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.CALCULUS
        assert result.difficulty == DifficultyLevel.INTERMEDIATE
        assert 'x' in result.variables
        assert result.problem_type == 'derivative'
        assert result.metadata['has_derivatives'] == True
    
    def test_parse_integral_problem(self):
        """Test parsing a calculus integral problem."""
        problem_text = "Evaluate the integral: ∫(2x + 1)dx"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.CALCULUS
        assert result.difficulty == DifficultyLevel.ADVANCED
        assert 'x' in result.variables
        assert result.problem_type == 'integral'
        assert result.metadata['has_integrals'] == True
    
    def test_parse_matrix_problem(self):
        """Test parsing a linear algebra matrix problem."""
        problem_text = "Find the determinant of the matrix A = [[1, 2], [3, 4]]"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.LINEAR_ALGEBRA
        assert result.difficulty == DifficultyLevel.ADVANCED
        assert result.problem_type == 'matrix_operation'
        assert result.metadata['has_matrices'] == True
    
    def test_parse_system_of_equations(self):
        """Test parsing a system of equations."""
        problem_text = "Solve the system of equations: 2x + y = 1, x - y = 3"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.LINEAR_ALGEBRA
        assert 'x' in result.variables
        assert 'y' in result.variables
        assert result.problem_type == 'system_of_equations'
        assert len(result.expressions) >= 2
    
    def test_parse_ai_ml_problem(self):
        """Test parsing an AI/ML mathematics problem."""
        problem_text = "Calculate the gradient of the loss function L(w) = w^2 + 2w"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.AI_ML_MATH
        assert result.difficulty == DifficultyLevel.EXPERT
        assert 'w' in result.variables
        assert result.problem_type == 'general_problem'
    
    def test_domain_hint_override(self):
        """Test that domain hint overrides automatic detection."""
        problem_text = "2x + 3 = 7"  # Would normally be algebra
        result = self.parser.parse_problem(problem_text, domain_hint="calculus")
        
        assert result.domain == MathDomain.CALCULUS
    
    def test_variable_extraction(self):
        """Test extraction of variables from various expressions."""
        test_cases = [
            ("2x + 3y = 7", ['x', 'y']),
            ("solve for t: 3t - 5 = 10", ['t']),
            ("f(x) = x^2 + sin(x)", ['x']),
            ("a + b + c = 0", ['a', 'b', 'c']),
        ]
        
        for text, expected_vars in test_cases:
            result = self.parser.parse_problem(text)
            for var in expected_vars:
                assert var in result.variables
    
    def test_expression_extraction(self):
        """Test extraction of mathematical expressions."""
        problem_text = "Solve (2x + 3) = 7 and find d/dx(x^2)"
        result = self.parser.parse_problem(problem_text)
        
        assert len(result.expressions) > 0
        # Should contain the equation and derivative expression
        expressions_text = ' '.join(result.expressions)
        assert '=' in expressions_text or 'd/dx' in expressions_text
    
    def test_difficulty_assessment(self):
        """Test difficulty level assessment."""
        test_cases = [
            ("2 + 3 = 5", DifficultyLevel.BEGINNER),
            ("x^2 - 4 = 0", DifficultyLevel.INTERMEDIATE),
            ("∫x^2 dx", DifficultyLevel.ADVANCED),
            ("Find the gradient of the neural network loss", DifficultyLevel.EXPERT),
        ]
        
        for text, expected_difficulty in test_cases:
            result = self.parser.parse_problem(text)
            assert result.difficulty == expected_difficulty
    
    def test_metadata_extraction(self):
        """Test extraction of problem metadata."""
        problem_text = "Solve x^2/2 + sqrt(x) = 5"
        result = self.parser.parse_problem(problem_text)
        
        assert result.metadata['has_equations'] == True
        assert result.metadata['contains_fractions'] == True
        assert result.metadata['contains_exponents'] == True
        assert result.metadata['contains_roots'] == True
        assert result.metadata['word_count'] > 0
    
    def test_clean_problem_text(self):
        """Test text cleaning and normalization."""
        dirty_text = "Solve   for  x:   2×x  ÷  3  =  4"
        cleaned = self.parser._clean_problem_text(dirty_text)
        
        assert "×" not in cleaned
        assert "÷" not in cleaned
        assert "*" in cleaned
        assert "/" in cleaned
        assert "  " not in cleaned  # No double spaces
    
    def test_validate_expression(self):
        """Test expression validation."""
        valid_expressions = [
            "x + 2",
            "x^2 + 3*x + 1",
            "sin(x) + cos(x)",
            "2*x*y + z"
        ]
        
        invalid_expressions = [
            "x +",
            "2 ** ** 3",
            "sin(cos(tan(",
            ""
        ]
        
        for expr in valid_expressions:
            assert self.parser.validate_expression(expr) == True
        
        for expr in invalid_expressions:
            assert self.parser.validate_expression(expr) == False
    
    def test_get_expression_variables(self):
        """Test getting variables from specific expressions."""
        test_cases = [
            ("x + y", ['x', 'y']),
            ("2*a*b + c^2", ['a', 'b', 'c']),
            ("sin(t) + cos(t)", ['t']),
            ("5", []),  # No variables
        ]
        
        for expr, expected_vars in test_cases:
            result_vars = self.parser.get_expression_variables(expr)
            for var in expected_vars:
                assert var in result_vars
    
    def test_convert_to_latex(self):
        """Test LaTeX conversion."""
        test_cases = [
            ("x^2", "x^{2}"),
            ("sqrt(x)", "\\sqrt{x}"),
            ("x/y", "\\frac{x}{y}"),
        ]
        
        for expr, expected_latex in test_cases:
            latex_result = self.parser.convert_to_latex(expr)
            # LaTeX output might vary, so we check for key components
            assert "{" in latex_result or expected_latex in latex_result
    
    def test_parse_error_handling(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ParseError):
            self.parser.parse_problem("")  # Empty string
        
        # Very malformed input should still parse but might have limited info
        result = self.parser.parse_problem("random text with no math")
        assert result.domain == MathDomain.ALGEBRA  # Default domain
        assert len(result.variables) == 0
    
    def test_complex_mixed_problem(self):
        """Test parsing a complex problem with mixed mathematical concepts."""
        problem_text = """
        Given the function f(x) = x^2 + 3x - 2, find:
        1. The derivative f'(x)
        2. Solve f(x) = 0
        3. Find the integral ∫f(x)dx
        """
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.CALCULUS  # Should detect calculus due to derivative/integral
        assert 'x' in result.variables
        assert result.metadata['has_derivatives'] == True
        assert result.metadata['has_integrals'] == True
        assert result.metadata['has_equations'] == True
    
    def test_unicode_mathematical_symbols(self):
        """Test handling of Unicode mathematical symbols."""
        problem_text = "Find ∫₀^∞ e^(-x²) dx"
        result = self.parser.parse_problem(problem_text)
        
        assert result.domain == MathDomain.CALCULUS
        assert 'x' in result.variables
        assert result.metadata['has_integrals'] == True
    
    def test_problem_type_identification(self):
        """Test identification of specific problem types."""
        test_cases = [
            ("Factor x^2 - 4", "factoring"),
            ("Expand (x + 2)(x - 3)", "expansion"),
            ("Simplify 2x + 3x", "simplification"),
            ("Find the limit as x approaches 0", "limit"),
            ("Calculate the eigenvalues of matrix A", "eigenvalue_problem"),
        ]
        
        for text, expected_type in test_cases:
            result = self.parser.parse_problem(text)
            assert result.problem_type == expected_type


# Integration tests
class TestParserIntegration:
    """Integration tests for the parser with real-world examples."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = MathExpressionParser()
    
    def test_real_world_algebra_problems(self):
        """Test with real algebra problems students might ask."""
        problems = [
            "Solve for x: 3x - 7 = 2x + 5",
            "Find the roots of x^2 - 6x + 9 = 0",
            "Simplify (2x + 3)(x - 1)",
            "Factor 4x^2 - 16",
        ]
        
        for problem in problems:
            result = self.parser.parse_problem(problem)
            assert result.domain == MathDomain.ALGEBRA
            assert len(result.variables) > 0
            assert result.problem_type in ['linear_equation', 'quadratic_equation', 'expansion', 'factoring', 'simplification']
    
    def test_real_world_calculus_problems(self):
        """Test with real calculus problems."""
        problems = [
            "Find d/dx(x^3 + 2x^2 - x + 1)",
            "Evaluate ∫(3x^2 + 2x)dx",
            "Find the limit of (sin x)/x as x approaches 0",
            "Find the critical points of f(x) = x^3 - 3x^2 + 2",
        ]
        
        for problem in problems:
            result = self.parser.parse_problem(problem)
            assert result.domain == MathDomain.CALCULUS
            assert len(result.variables) > 0
    
    def test_performance_with_long_problems(self):
        """Test parser performance with longer, more complex problems."""
        long_problem = """
        Consider the optimization problem: minimize f(x,y) = x^2 + y^2 + 2xy
        subject to the constraints g(x,y) = x + y - 1 = 0 and h(x,y) = x^2 + y^2 - 4 ≤ 0.
        Use the method of Lagrange multipliers to find the optimal solution.
        Calculate the gradient of f, the gradients of g and h, and set up the
        Lagrangian L(x,y,λ,μ) = f(x,y) + λg(x,y) + μh(x,y).
        """
        
        result = self.parser.parse_problem(long_problem)
        
        # Should handle long problems without errors
        assert result.domain in [MathDomain.AI_ML_MATH, MathDomain.CALCULUS]
        assert 'x' in result.variables
        assert 'y' in result.variables
        assert result.metadata['word_count'] > 50


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])