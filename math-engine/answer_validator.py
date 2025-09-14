"""
Answer Validation and Feedback System for AI Math Tutor
Validates mathematical answers using symbolic computation and provides detailed feedback.
"""

import re
import sympy as sp
from sympy import symbols, simplify, expand, factor, N, sympify, Eq, solve
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add shared models to path
from models import ValidationResult, MathTutorError


class AnswerValidator:
    """Validates mathematical answers using symbolic computation."""
    
    def __init__(self):
        """Initialize the answer validator."""
        self.transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
        
        # Tolerance for numerical comparisons
        self.numerical_tolerance = 1e-10
        
        # Common mathematical constants and their string representations
        self.constants_map = {
            'pi': sp.pi,
            'π': sp.pi,
            'e': sp.E,
            'inf': sp.oo,
            'infinity': sp.oo,
            '∞': sp.oo
        }
    
    def validate_answer(self, user_answer: str, correct_answer: str, 
                       problem_type: str = 'general', context: Dict = None) -> ValidationResult:
        """
        Validate a user's answer against the correct solution.
        
        Args:
            user_answer: User's submitted answer
            correct_answer: The correct answer
            problem_type: Type of mathematical problem
            context: Additional context about the problem
            
        Returns:
            ValidationResult: Detailed validation result with feedback
        """
        try:
            # Normalize both answers
            normalized_user = self._normalize_answer(user_answer)
            normalized_correct = self._normalize_answer(correct_answer)
            
            # Check for exact match first
            if normalized_user == normalized_correct:
                return ValidationResult(
                    is_correct=True,
                    user_answer=user_answer,
                    correct_answer=correct_answer,
                    explanation="Correct! Your answer matches the expected solution.",
                    partial_credit=1.0
                )
            
            # Handle special cases before symbolic parsing
            if self._handle_special_cases(user_answer, correct_answer):
                return ValidationResult(
                    is_correct=True,
                    user_answer=user_answer,
                    correct_answer=correct_answer,
                    explanation="Correct! Your answer is mathematically equivalent.",
                    partial_credit=1.0
                )
            
            # Try simple numerical evaluation first for basic cases
            try:
                if user_answer.replace('/', '').replace('.', '').replace('-', '').isdigit() and \
                   correct_answer.replace('/', '').replace('.', '').replace('-', '').isdigit():
                    user_val = eval(user_answer)  # Safe for simple numerical expressions
                    correct_val = eval(correct_answer)
                    if abs(user_val - correct_val) < 0.0001:
                        return ValidationResult(
                            is_correct=True,
                            user_answer=user_answer,
                            correct_answer=correct_answer,
                            explanation="Correct! Your answer is numerically equivalent.",
                            partial_credit=1.0
                        )
            except:
                pass
            
            # Perform symbolic comparison
            is_correct, partial_credit, explanation = self._symbolic_comparison(
                normalized_user, normalized_correct, problem_type
            )
            
            return ValidationResult(
                is_correct=is_correct,
                user_answer=user_answer,
                correct_answer=correct_answer,
                explanation=explanation,
                partial_credit=partial_credit
            )
            
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                user_answer=user_answer,
                correct_answer=correct_answer,
                explanation=f"Could not validate answer due to parsing error: {str(e)}",
                partial_credit=0.0
            )
    
    def generate_feedback(self, user_answer: str, correct_answer: str, 
                         is_correct: bool, problem_type: str = 'general') -> str:
        """
        Generate detailed feedback for incorrect answers.
        
        Args:
            user_answer: User's submitted answer
            correct_answer: The correct answer
            is_correct: Whether the answer was correct
            problem_type: Type of mathematical problem
            
        Returns:
            Detailed feedback string
        """
        if is_correct:
            return "Excellent work! Your answer is correct."
        
        try:
            # Analyze the error and provide specific feedback
            error_analysis = self._analyze_error(user_answer, correct_answer, problem_type)
            
            feedback_parts = [
                f"Your answer '{user_answer}' is incorrect.",
                f"The correct answer is '{correct_answer}'.",
                "",
                "Analysis of your answer:",
                error_analysis,
                "",
                "Suggestions for improvement:",
                self._generate_improvement_suggestions(user_answer, correct_answer, problem_type)
            ]
            
            return "\n".join(feedback_parts)
            
        except Exception as e:
            return f"Your answer is incorrect. The correct answer is '{correct_answer}'. Please review the problem and try again."
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer string for comparison."""
        if not answer or not isinstance(answer, str):
            return ""
        
        # Remove extra whitespace
        normalized = answer.strip()
        
        # Handle common formatting variations
        normalized = normalized.replace(' ', '')  # Remove all spaces
        normalized = normalized.replace('**', '^')  # Convert ** to ^
        normalized = normalized.replace('sqrt', '√')  # Convert sqrt to symbol
        
        # Handle fractions
        normalized = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', normalized)
        
        # Handle common constants
        for const_str, const_sym in self.constants_map.items():
            normalized = normalized.replace(const_str, str(const_sym))
        
        return normalized
    
    def _symbolic_comparison(self, user_answer: str, correct_answer: str, 
                           problem_type: str) -> Tuple[bool, float, str]:
        """
        Compare answers using symbolic mathematics.
        
        Returns:
            Tuple of (is_correct, partial_credit, explanation)
        """
        try:
            # Handle special cases first
            if self._handle_special_cases(user_answer, correct_answer):
                return True, 1.0, "Correct! Your answer is mathematically equivalent."
            
            # Parse expressions
            user_expr = self._safe_parse_expression(user_answer)
            correct_expr = self._safe_parse_expression(correct_answer)
            
            if user_expr is None or correct_expr is None:
                return self._fallback_comparison(user_answer, correct_answer)
            
            # Check for symbolic equivalence
            if self._are_symbolically_equivalent(user_expr, correct_expr):
                return True, 1.0, "Correct! Your answer is mathematically equivalent to the expected solution."
            
            # Check for partial credit scenarios
            partial_credit, explanation = self._calculate_partial_credit(
                user_expr, correct_expr, problem_type
            )
            
            if partial_credit > 0:
                return False, partial_credit, explanation
            
            # No equivalence found
            return False, 0.0, self._generate_error_explanation(user_answer, correct_answer, problem_type)
            
        except Exception as e:
            return False, 0.0, f"Could not compare answers: {str(e)}"
    
    def _safe_parse_expression(self, expr_str: str) -> Optional[sp.Basic]:
        """Safely parse a mathematical expression."""
        try:
            # Handle multiple solutions (comma-separated)
            if ',' in expr_str:
                parts = [part.strip() for part in expr_str.split(',')]
                parsed_parts = []
                for part in parts:
                    parsed = parse_expr(part, transformations=self.transformations)
                    parsed_parts.append(parsed)
                return parsed_parts
            
            # Handle equations
            if '=' in expr_str:
                left, right = expr_str.split('=', 1)
                left_expr = parse_expr(left.strip(), transformations=self.transformations)
                right_expr = parse_expr(right.strip(), transformations=self.transformations)
                return Eq(left_expr, right_expr)
            
            # Handle vectors/matrices (basic support)
            if expr_str.startswith('[') and expr_str.endswith(']'):
                # Simple vector parsing
                content = expr_str[1:-1]
                elements = [parse_expr(elem.strip(), transformations=self.transformations) 
                           for elem in content.split(',')]
                return elements
            
            # Regular expression parsing
            return parse_expr(expr_str, transformations=self.transformations)
            
        except Exception:
            return None
    
    def _are_symbolically_equivalent(self, expr1, expr2) -> bool:
        """Check if two expressions are symbolically equivalent."""
        try:
            # Handle lists (multiple solutions or vectors)
            if isinstance(expr1, list) and isinstance(expr2, list):
                if len(expr1) != len(expr2):
                    return False
                # Check if all elements match (order doesn't matter for solutions)
                return all(any(self._expressions_equal(e1, e2) for e2 in expr2) for e1 in expr1)
            
            # Handle equations
            if isinstance(expr1, Eq) and isinstance(expr2, Eq):
                return self._expressions_equal(expr1.lhs - expr1.rhs, expr2.lhs - expr2.rhs)
            
            # Handle regular expressions
            return self._expressions_equal(expr1, expr2)
            
        except Exception:
            return False
    
    def _expressions_equal(self, expr1, expr2) -> bool:
        """Check if two SymPy expressions are equal."""
        try:
            # Try direct equality
            if expr1.equals(expr2):
                return True
            
            # Try simplification
            diff = simplify(expr1 - expr2)
            if diff == 0:
                return True
            
            # Try numerical evaluation for expressions with no free symbols
            if not (expr1.free_symbols or expr2.free_symbols):
                val1 = complex(N(expr1))
                val2 = complex(N(expr2))
                return abs(val1 - val2) < self.numerical_tolerance
            
            # Try expanding both expressions
            expanded_diff = simplify(expand(expr1) - expand(expr2))
            if expanded_diff == 0:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _handle_special_cases(self, user_answer: str, correct_answer: str) -> bool:
        """Handle special cases that might not parse well."""
        # Normalize case and whitespace
        user_norm = user_answer.lower().strip()
        correct_norm = correct_answer.lower().strip()
        
        # Handle "no solution" cases
        no_solution_phrases = ['no solution', 'no real solution', 'no real solutions', 
                              'undefined', 'does not exist', 'dne']
        
        user_has_no_solution = any(phrase in user_norm for phrase in no_solution_phrases)
        correct_has_no_solution = any(phrase in correct_norm for phrase in no_solution_phrases)
        
        if user_has_no_solution and correct_has_no_solution:
            return True
        
        # Handle infinity cases
        infinity_phrases = ['infinity', 'inf', '∞', 'unbounded']
        user_has_infinity = any(phrase in user_norm for phrase in infinity_phrases)
        correct_has_infinity = any(phrase in correct_norm for phrase in infinity_phrases)
        
        if user_has_infinity and correct_has_infinity:
            return True
        
        return False
    
    def _fallback_comparison(self, user_answer: str, correct_answer: str) -> Tuple[bool, float, str]:
        """Fallback comparison when symbolic parsing fails."""
        # Simple string comparison after normalization
        user_norm = self._normalize_answer(user_answer).lower()
        correct_norm = self._normalize_answer(correct_answer).lower()
        
        if user_norm == correct_norm:
            return True, 1.0, "Correct!"
        
        # Try basic numerical evaluation for simple cases
        try:
            # Handle simple fractions and numbers
            if '/' in user_answer and user_answer.count('/') == 1:
                parts = user_answer.split('/')
                if len(parts) == 2 and parts[0].strip().isdigit() and parts[1].strip().isdigit():
                    user_val = float(parts[0]) / float(parts[1])
                    try:
                        correct_val = float(correct_answer)
                        if abs(user_val - correct_val) < 0.0001:  # Use larger tolerance
                            return True, 1.0, "Correct! (equivalent numerical value)"
                    except ValueError:
                        pass
        except:
            pass
        
        # Check for common equivalent forms
        equivalents = [
            (r'(\d+)/(\d+)', r'\1/\2'),  # Fraction normalization
            (r'\*', ''),  # Remove multiplication signs
            (r'\^', '**'),  # Power notation
        ]
        
        for pattern, replacement in equivalents:
            user_test = re.sub(pattern, replacement, user_norm)
            correct_test = re.sub(pattern, replacement, correct_norm)
            if user_test == correct_test:
                return True, 1.0, "Correct! (equivalent form)"
        
        return False, 0.0, "Answers do not match."
    
    def _calculate_partial_credit(self, user_expr, correct_expr, problem_type: str) -> Tuple[float, str]:
        """Calculate partial credit for partially correct answers."""
        try:
            # For quadratic equations, check if user found one root
            if problem_type == 'quadratic_equation':
                if isinstance(correct_expr, list) and len(correct_expr) == 2:
                    if isinstance(user_expr, list):
                        # Check how many roots are correct
                        correct_roots = sum(1 for user_root in user_expr 
                                          if any(self._expressions_equal(user_root, correct_root) 
                                               for correct_root in correct_expr))
                        if correct_roots > 0:
                            credit = correct_roots / len(correct_expr)
                            return credit, f"Partial credit: {correct_roots}/{len(correct_expr)} roots correct."
                    else:
                        # Single answer - check if it's one of the roots
                        if any(self._expressions_equal(user_expr, root) for root in correct_expr):
                            return 0.5, "Partial credit: You found one of the two roots."
            
            # For algebraic expressions, check if the form is close
            if problem_type in ['expansion', 'factoring', 'simplification']:
                # Check if the degree and leading coefficient are correct
                try:
                    if hasattr(user_expr, 'as_poly') and hasattr(correct_expr, 'as_poly'):
                        user_poly = user_expr.as_poly()
                        correct_poly = correct_expr.as_poly()
                        
                        if user_poly and correct_poly:
                            if user_poly.degree() == correct_poly.degree():
                                return 0.3, "Partial credit: Correct degree but wrong coefficients."
                except:
                    pass
            
            return 0.0, ""
            
        except Exception:
            return 0.0, ""
    
    def _analyze_error(self, user_answer: str, correct_answer: str, problem_type: str) -> str:
        """Analyze the type of error made by the user."""
        try:
            user_expr = self._safe_parse_expression(user_answer)
            correct_expr = self._safe_parse_expression(correct_answer)
            
            if user_expr is None:
                return "Your answer could not be parsed. Check for syntax errors."
            
            if correct_expr is None:
                return "There was an issue with the expected answer format."
            
            # Analyze specific error types
            if problem_type == 'linear_equation':
                return self._analyze_linear_equation_error(user_expr, correct_expr)
            elif problem_type == 'quadratic_equation':
                return self._analyze_quadratic_error(user_expr, correct_expr)
            elif problem_type in ['expansion', 'factoring']:
                return self._analyze_algebraic_error(user_expr, correct_expr, problem_type)
            elif problem_type in ['derivative', 'integral']:
                return self._analyze_calculus_error(user_expr, correct_expr, problem_type)
            else:
                return self._analyze_general_error(user_expr, correct_expr)
                
        except Exception as e:
            return f"Error analysis failed: {str(e)}"
    
    def _analyze_linear_equation_error(self, user_expr, correct_expr) -> str:
        """Analyze errors in linear equation solving."""
        try:
            # Check if user made a sign error
            if self._expressions_equal(user_expr, -correct_expr):
                return "Sign error: Your answer has the opposite sign. Check your arithmetic when moving terms."
            
            # Check if user forgot to divide by coefficient
            if isinstance(correct_expr, (int, float)) and isinstance(user_expr, (int, float)):
                ratio = user_expr / correct_expr
                if abs(ratio - round(ratio)) < 0.01:  # Close to integer ratio
                    return f"It looks like you may have forgotten to divide by {int(round(ratio))} or made an arithmetic error."
            
            return "Check your algebraic steps. Make sure to perform the same operation on both sides of the equation."
            
        except:
            return "Review the steps for solving linear equations."
    
    def _analyze_quadratic_error(self, user_expr, correct_expr) -> str:
        """Analyze errors in quadratic equation solving."""
        if isinstance(correct_expr, list) and not isinstance(user_expr, list):
            return "This quadratic equation has two solutions. Make sure to find both roots."
        
        return "Check your application of the quadratic formula. Verify the discriminant calculation and the final arithmetic."
    
    def _analyze_algebraic_error(self, user_expr, correct_expr, problem_type: str) -> str:
        """Analyze errors in algebraic manipulation."""
        if problem_type == 'expansion':
            return "When expanding, make sure to multiply each term in the first expression by each term in the second expression (FOIL method)."
        elif problem_type == 'factoring':
            return "Check if your factors multiply back to the original expression. Look for common factors first."
        else:
            return "Review the algebraic manipulation rules for this type of problem."
    
    def _analyze_calculus_error(self, user_expr, correct_expr, problem_type: str) -> str:
        """Analyze errors in calculus problems."""
        if problem_type == 'derivative':
            return "Review the differentiation rules (power rule, product rule, chain rule). Check if you applied them correctly."
        elif problem_type == 'integral':
            return "Review integration techniques. Don't forget the constant of integration for indefinite integrals."
        else:
            return "Review the relevant calculus concepts and rules."
    
    def _analyze_general_error(self, user_expr, correct_expr) -> str:
        """General error analysis."""
        return "Your answer doesn't match the expected result. Review your work step by step."
    
    def _generate_improvement_suggestions(self, user_answer: str, correct_answer: str, 
                                        problem_type: str) -> str:
        """Generate suggestions for improvement."""
        suggestions = []
        
        if problem_type == 'linear_equation':
            suggestions = [
                "• Double-check your arithmetic when moving terms across the equals sign",
                "• Remember to perform the same operation on both sides",
                "• Verify your final answer by substituting back into the original equation"
            ]
        elif problem_type == 'quadratic_equation':
            suggestions = [
                "• Make sure the equation is in standard form (ax² + bx + c = 0)",
                "• Double-check your discriminant calculation (b² - 4ac)",
                "• Be careful with signs in the quadratic formula",
                "• Consider if there might be two solutions"
            ]
        elif problem_type == 'expansion':
            suggestions = [
                "• Use the FOIL method systematically",
                "• Make sure to multiply every term by every other term",
                "• Combine like terms carefully",
                "• Check your work by factoring your result"
            ]
        elif problem_type == 'derivative':
            suggestions = [
                "• Review the power rule: d/dx[x^n] = nx^(n-1)",
                "• Remember that the derivative of a constant is 0",
                "• Apply differentiation rules to each term separately"
            ]
        else:
            suggestions = [
                "• Break the problem down into smaller steps",
                "• Check each step of your work",
                "• Verify your final answer makes sense in the context"
            ]
        
        return "\n".join(suggestions)
    
    def _generate_error_explanation(self, user_answer: str, correct_answer: str, 
                                  problem_type: str) -> str:
        """Generate a detailed explanation of why the answer is incorrect."""
        base_explanation = f"Your answer '{user_answer}' is incorrect. The correct answer is '{correct_answer}'."
        
        error_analysis = self._analyze_error(user_answer, correct_answer, problem_type)
        suggestions = self._generate_improvement_suggestions(user_answer, correct_answer, problem_type)
        
        return f"{base_explanation}\n\n{error_analysis}\n\nSuggestions:\n{suggestions}"