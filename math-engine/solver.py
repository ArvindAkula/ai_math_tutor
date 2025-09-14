"""
Step-by-step mathematical problem solver.
Generates detailed solutions with explanations for various mathematical problems.
Enhanced with comprehensive error handling and fallback systems.
"""

import time
import uuid
from typing import List, Dict, Any, Optional
import sympy as sp
from sympy import symbols, Eq, solve, expand, factor, simplify, diff, integrate, limit, oo, pi, E, sin, cos, tan, exp, log, sqrt
from sympy.solvers import solve_linear_system
from sympy.matrices import Matrix
from sympy.plotting import plot
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
import sys
import os

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, ValidationResult,
    MathDomain, DifficultyLevel, ComputationError
)

# Import error handling
from error_handling import (
    error_handler, ErrorCategory, ErrorSeverity, MathComputationError,
    fallback_manager, retry_manager, CircuitBreaker
)


class MathSolver:
    """Step-by-step mathematical problem solver."""
    
    def __init__(self):
        """Initialize the solver."""
        self.solution_methods = {
            'linear_equation': self._solve_linear_equation,
            'quadratic_equation': self._solve_quadratic_equation,
            'algebraic_equation': self._solve_algebraic_equation,
            'factoring': self._solve_factoring,
            'expansion': self._solve_expansion,
            'simplification': self._solve_simplification,
            'system_of_equations': self._solve_system_of_equations,
            'derivative': self._solve_derivative,
            'integral': self._solve_integral,
            'limit': self._solve_limit,
            'optimization': self._solve_optimization,
            'general_problem': self._solve_general_problem
        }
        
        # Standard transformations for parsing mathematical expressions
        self.transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    
    @error_handler(ErrorCategory.COMPUTATION, ErrorSeverity.HIGH)
    def solve_step_by_step(self, problem: ParsedProblem) -> StepSolution:
        """
        Generate a complete step-by-step solution for a mathematical problem.
        Enhanced with comprehensive error handling and fallback systems.
        
        Args:
            problem: Parsed mathematical problem
            
        Returns:
            StepSolution: Complete solution with steps and explanations
            
        Raises:
            MathComputationError: If the problem cannot be solved
        """
        start_time = time.time()
        
        # Get the appropriate solver method
        solver_method = self.solution_methods.get(
            problem.problem_type, 
            self._solve_general_problem
        )
        
        # Generate the step-by-step solution with retry logic
        steps, final_answer, method = retry_manager.retry_with_backoff(
            solver_method, problem
        )
        
        # Calculate computation time
        computation_time = time.time() - start_time
        
        # Determine confidence score based on problem type and complexity
        confidence_score = self._calculate_confidence_score(problem, steps)
        
        return StepSolution(
            problem_id=problem.id,
            steps=steps,
            final_answer=final_answer,
            solution_method=method,
            confidence_score=confidence_score,
            computation_time=computation_time
        )
    
    @error_handler(ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM)
    def validate_answer(self, problem: ParsedProblem, user_answer: str) -> ValidationResult:
        """
        Validate a user's answer against the correct solution.
        Enhanced with error handling and fallback validation methods.
        
        Args:
            problem: The mathematical problem
            user_answer: User's submitted answer
            
        Returns:
            ValidationResult: Validation result with explanation
        """
        # Get the correct solution
        solution = self.solve_step_by_step(problem)
        correct_answer = solution.final_answer
        
        # Normalize both answers for comparison
        normalized_user = self._normalize_answer(user_answer)
        normalized_correct = self._normalize_answer(correct_answer)
        
        # Check if answers are equivalent
        is_correct = self._are_answers_equivalent(normalized_user, normalized_correct)
        
        # Calculate partial credit if applicable
        partial_credit = self._calculate_partial_credit(
            problem, user_answer, correct_answer, is_correct
        )
        
        # Generate explanation
        explanation = self._generate_validation_explanation(
            problem, user_answer, correct_answer, is_correct
        )
        
        return ValidationResult(
            is_correct=is_correct,
            user_answer=user_answer,
            correct_answer=correct_answer,
            explanation=explanation,
            partial_credit=partial_credit
        )
    
    def _solve_linear_equation(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve linear equations step by step."""
        steps = []
        step_num = 1
        
        # Extract the equation from the problem
        equation_text = self._extract_main_equation(problem)
        if not equation_text:
            raise ComputationError("No equation found in problem")
        
        # Parse the equation using proper transformations
        left_side, right_side = equation_text.split('=')
        left_expr = parse_expr(left_side.strip(), transformations=self.transformations)
        right_expr = parse_expr(right_side.strip(), transformations=self.transformations)
        
        # Get the variable to solve for
        variable = self._get_solve_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Step 1: Show the original equation
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given equation",
            explanation=f"We need to solve for {variable} in the equation",
            mathematical_expression=f"{left_expr} = {right_expr}",
            intermediate_result=f"{left_expr} = {right_expr}"
        ))
        step_num += 1
        
        # Step 2: Move terms to isolate variable
        # Create equation object
        equation = Eq(left_expr, right_expr)
        
        # If it's a simple form like ax + b = c, show the steps
        if left_expr.is_Add:
            # Separate variable terms and constants
            var_terms = []
            constants = []
            
            for term in sp.Add.make_args(left_expr):
                if var_symbol in term.free_symbols:
                    var_terms.append(term)
                else:
                    constants.append(term)
            
            if constants:
                # Step: Subtract constants from both sides
                constant_sum = sum(constants)
                new_right = right_expr - constant_sum
                new_left = sum(var_terms) if var_terms else 0
                
                steps.append(SolutionStep(
                    step_number=step_num,
                    operation=f"Subtract {constant_sum} from both sides",
                    explanation=f"To isolate the variable term, we subtract {constant_sum} from both sides",
                    mathematical_expression=f"{new_left} = {new_right}",
                    intermediate_result=f"{new_left} = {sp.simplify(new_right)}"
                ))
                step_num += 1
                
                left_expr = new_left
                right_expr = sp.simplify(new_right)
        
        # Step 3: Divide by coefficient if needed
        if left_expr != var_symbol:
            # Extract coefficient
            coeff = left_expr.coeff(var_symbol)
            if coeff and coeff != 1:
                final_result = right_expr / coeff
                
                steps.append(SolutionStep(
                    step_number=step_num,
                    operation=f"Divide both sides by {coeff}",
                    explanation=f"To solve for {variable}, we divide both sides by the coefficient {coeff}",
                    mathematical_expression=f"{variable} = {right_expr}/{coeff}",
                    intermediate_result=f"{variable} = {sp.simplify(final_result)}"
                ))
                step_num += 1
                
                final_answer = str(sp.simplify(final_result))
            else:
                final_answer = str(sp.simplify(right_expr))
        else:
            final_answer = str(sp.simplify(right_expr))
        
        # Final step: State the solution
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Solution",
            explanation=f"The value of {variable} that satisfies the equation",
            mathematical_expression=f"{variable} = {final_answer}",
            intermediate_result=f"{variable} = {final_answer}"
        ))
        
        return steps, final_answer, "Linear equation solving"
    
    def _solve_quadratic_equation(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve quadratic equations step by step."""
        steps = []
        step_num = 1
        
        # Extract the equation
        equation_text = self._extract_main_equation(problem)
        if not equation_text:
            raise ComputationError("No equation found in problem")
        
        # Parse the equation using proper transformations
        left_side, right_side = equation_text.split('=')
        left_expr = parse_expr(left_side.strip(), transformations=self.transformations)
        right_expr = parse_expr(right_side.strip(), transformations=self.transformations)
        
        # Get the variable
        variable = self._get_solve_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Step 1: Show original equation
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given quadratic equation",
            explanation=f"We need to solve the quadratic equation for {variable}",
            mathematical_expression=f"{left_expr} = {right_expr}",
            intermediate_result=f"{left_expr} = {right_expr}"
        ))
        step_num += 1
        
        # Step 2: Move everything to one side (standard form)
        standard_form = left_expr - right_expr
        standard_form = sp.expand(standard_form)
        
        if right_expr != 0:
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Rearrange to standard form",
                explanation="Move all terms to one side to get the standard form ax² + bx + c = 0",
                mathematical_expression=f"{standard_form} = 0",
                intermediate_result=f"{standard_form} = 0"
            ))
            step_num += 1
        
        # Step 3: Identify coefficients
        poly = sp.Poly(standard_form, var_symbol)
        coeffs = poly.all_coeffs()
        
        # Pad coefficients if needed (for linear terms that might be missing)
        while len(coeffs) < 3:
            coeffs = [0] + coeffs
        
        a, b, c = coeffs[-3], coeffs[-2], coeffs[-1]
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Identify coefficients",
            explanation=f"In the standard form ax² + bx + c = 0, we have a = {a}, b = {b}, c = {c}",
            mathematical_expression=f"a = {a}, b = {b}, c = {c}",
            intermediate_result=f"Coefficients identified"
        ))
        step_num += 1
        
        # Step 4: Apply quadratic formula
        discriminant = b**2 - 4*a*c
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Calculate discriminant",
            explanation="The discriminant Δ = b² - 4ac determines the nature of the roots",
            mathematical_expression=f"Δ = ({b})² - 4({a})({c}) = {discriminant}",
            intermediate_result=f"Δ = {discriminant}"
        ))
        step_num += 1
        
        # Step 5: Solve using quadratic formula
        if discriminant >= 0:
            sqrt_discriminant = sp.sqrt(discriminant)
            root1 = (-b + sqrt_discriminant) / (2*a)
            root2 = (-b - sqrt_discriminant) / (2*a)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Apply quadratic formula",
                explanation="Using the quadratic formula: x = (-b ± √Δ) / (2a)",
                mathematical_expression=f"{variable} = ({-b} ± √{discriminant}) / (2·{a})",
                intermediate_result=f"{variable} = {sp.simplify(root1)} or {variable} = {sp.simplify(root2)}"
            ))
            
            if discriminant == 0:
                final_answer = str(sp.simplify(root1))
            else:
                final_answer = f"{sp.simplify(root1)}, {sp.simplify(root2)}"
        else:
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Complex roots",
                explanation="Since the discriminant is negative, the equation has complex roots",
                mathematical_expression=f"{variable} = ({-b} ± √{discriminant}) / (2·{a})",
                intermediate_result="Complex roots (no real solutions)"
            ))
            final_answer = "No real solutions"
        
        return steps, final_answer, "Quadratic formula method"
    
    def _solve_algebraic_equation(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve general algebraic equations."""
        steps = []
        step_num = 1
        
        # Extract equation
        equation_text = self._extract_main_equation(problem)
        if not equation_text:
            raise ComputationError("No equation found in problem")
        
        # Parse equation using proper transformations
        left_side, right_side = equation_text.split('=')
        left_expr = parse_expr(left_side.strip(), transformations=self.transformations)
        right_expr = parse_expr(right_side.strip(), transformations=self.transformations)
        
        # Get variable
        variable = self._get_solve_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Step 1: Show original equation
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given equation",
            explanation=f"We need to solve for {variable}",
            mathematical_expression=f"{left_expr} = {right_expr}",
            intermediate_result=f"{left_expr} = {right_expr}"
        ))
        step_num += 1
        
        # Step 2: Solve using SymPy
        equation = Eq(left_expr, right_expr)
        solutions = solve(equation, var_symbol)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Solve the equation",
            explanation=f"Using algebraic methods to solve for {variable}",
            mathematical_expression=f"solve({left_expr} = {right_expr}, {variable})",
            intermediate_result=f"Solutions: {solutions}"
        ))
        
        # Format final answer
        if len(solutions) == 1:
            final_answer = str(solutions[0])
        elif len(solutions) > 1:
            final_answer = ", ".join(str(sol) for sol in solutions)
        else:
            final_answer = "No solution"
        
        return steps, final_answer, "Algebraic solving"
    
    def _solve_factoring(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve factoring problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the expression to factor
        expression_text = self._extract_main_expression(problem)
        if not expression_text:
            # Try to extract from original text if no expression found
            # Look for pattern like "Factor: x^2 - 4"
            import re
            match = re.search(r'factor:?\s*(.+)', problem.original_text.lower())
            if match:
                expression_text = match.group(1).strip()
            else:
                raise ComputationError("No expression found to factor")
        
        expr = parse_expr(expression_text, transformations=self.transformations)
        
        # Step 1: Show original expression
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given expression",
            explanation="We need to factor this expression",
            mathematical_expression=str(expr),
            intermediate_result=str(expr)
        ))
        step_num += 1
        
        # Step 2: Factor the expression
        factored = factor(expr)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Factor the expression",
            explanation="Find the factors of the expression",
            mathematical_expression=f"factor({expr})",
            intermediate_result=str(factored)
        ))
        
        return steps, str(factored), "Factoring"
    
    def _solve_expansion(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve expansion problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the expression to expand
        expression_text = self._extract_main_expression(problem)
        if not expression_text:
            # Try to extract from original text if no expression found
            import re
            match = re.search(r'expand:?\s*(.+)', problem.original_text.lower())
            if match:
                expression_text = match.group(1).strip()
            else:
                raise ComputationError("No expression found to expand")
        
        # For expansion problems, if we have multiple expressions that could be factors, 
        # try to reconstruct the full product from the original text
        if len(problem.expressions) >= 2:
            # Look for pattern like (x + 2)(x - 3) in original text
            import re
            product_match = re.search(r'\([^)]+\)\s*\([^)]+\)', problem.original_text)
            if product_match:
                expression_text = product_match.group(0)
        
        expr = parse_expr(expression_text, transformations=self.transformations)
        
        # Step 1: Show original expression
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given expression",
            explanation="We need to expand this expression",
            mathematical_expression=str(expr),
            intermediate_result=str(expr)
        ))
        step_num += 1
        
        # Step 2: Expand the expression
        expanded = expand(expr)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Expand the expression",
            explanation="Multiply out all terms and combine like terms",
            mathematical_expression=f"expand({expr})",
            intermediate_result=str(expanded)
        ))
        
        return steps, str(expanded), "Expansion"
    
    def _solve_simplification(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve simplification problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the expression to simplify
        expression_text = self._extract_main_expression(problem)
        if not expression_text:
            # Try to extract from original text if no expression found
            import re
            match = re.search(r'simplify:?\s*(.+)', problem.original_text.lower())
            if match:
                expression_text = match.group(1).strip()
            else:
                raise ComputationError("No expression found to simplify")
        
        expr = parse_expr(expression_text, transformations=self.transformations)
        
        # Step 1: Show original expression
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given expression",
            explanation="We need to simplify this expression",
            mathematical_expression=str(expr),
            intermediate_result=str(expr)
        ))
        step_num += 1
        
        # Step 2: Simplify the expression
        simplified = simplify(expr)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Simplify the expression",
            explanation="Combine like terms and reduce to simplest form",
            mathematical_expression=f"simplify({expr})",
            intermediate_result=str(simplified)
        ))
        
        return steps, str(simplified), "Simplification"
    
    def _solve_system_of_equations(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve system of equations step by step."""
        steps = []
        step_num = 1
        
        # Extract equations from the problem
        equations = self._extract_system_equations(problem)
        if len(equations) < 2:
            raise ComputationError("Need at least 2 equations for a system")
        
        # Get variables
        variables = [sp.Symbol(var) for var in problem.variables]
        
        # Step 1: Show the system
        system_text = ", ".join(equations)
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the system of equations",
            explanation=f"We need to solve this system for {', '.join(problem.variables)}",
            mathematical_expression=system_text,
            intermediate_result=system_text
        ))
        step_num += 1
        
        # Step 2: Parse equations
        sympy_equations = []
        for eq_text in equations:
            left, right = eq_text.split('=')
            left_expr = parse_expr(left.strip(), transformations=self.transformations)
            right_expr = parse_expr(right.strip(), transformations=self.transformations)
            sympy_equations.append(Eq(left_expr, right_expr))
        
        # Step 3: Solve the system
        solutions = solve(sympy_equations, variables)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Solve the system",
            explanation="Use substitution or elimination to find the values of all variables",
            mathematical_expression=f"solve({system_text})",
            intermediate_result=str(solutions)
        ))
        
        # Format final answer
        if isinstance(solutions, dict):
            final_answer = ", ".join(f"{var} = {val}" for var, val in solutions.items())
        else:
            final_answer = str(solutions)
        
        return steps, final_answer, "System solving"
    
    def _solve_derivative(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve derivative problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the function to differentiate
        function_text = self._extract_calculus_function(problem, 'derivative')
        if not function_text:
            raise ComputationError("No function found to differentiate")
        
        # Parse the function
        expr = parse_expr(function_text, transformations=self.transformations)
        
        # Get the variable to differentiate with respect to
        variable = self._get_calculus_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Step 1: Show the original function
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given function",
            explanation=f"We need to find the derivative of f({variable}) = {expr}",
            mathematical_expression=f"f({variable}) = {expr}",
            intermediate_result=f"f({variable}) = {expr}"
        ))
        step_num += 1
        
        # Step 2: Apply differentiation rules
        derivative = diff(expr, var_symbol)
        
        # Identify which rules are being applied
        rule_explanation = self._identify_derivative_rules(expr, var_symbol)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Apply differentiation rules",
            explanation=rule_explanation,
            mathematical_expression=f"d/d{variable}[{expr}]",
            intermediate_result=f"f'({variable}) = {derivative}"
        ))
        step_num += 1
        
        # Step 3: Simplify if needed
        simplified_derivative = simplify(derivative)
        if simplified_derivative != derivative:
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Simplify the result",
                explanation="Combine and simplify terms in the derivative",
                mathematical_expression=f"simplify({derivative})",
                intermediate_result=f"f'({variable}) = {simplified_derivative}"
            ))
            final_answer = str(simplified_derivative)
        else:
            final_answer = str(derivative)
        
        return steps, final_answer, "Differentiation"
    
    def _solve_integral(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve integral problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the function to integrate
        function_text = self._extract_calculus_function(problem, 'integral')
        if not function_text:
            raise ComputationError("No function found to integrate")
        
        # Parse the function
        expr = parse_expr(function_text, transformations=self.transformations)
        
        # Get the variable to integrate with respect to
        variable = self._get_calculus_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Check if it's definite or indefinite integral
        is_definite, lower_limit, upper_limit = self._extract_integral_limits(problem)
        
        # Step 1: Show the original integral
        if is_definite:
            integral_notation = f"∫[{lower_limit} to {upper_limit}] {expr} d{variable}"
            explanation = f"We need to evaluate the definite integral from {lower_limit} to {upper_limit}"
        else:
            integral_notation = f"∫ {expr} d{variable}"
            explanation = f"We need to find the indefinite integral (antiderivative) of {expr}"
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given integral",
            explanation=explanation,
            mathematical_expression=integral_notation,
            intermediate_result=integral_notation
        ))
        step_num += 1
        
        # Step 2: Find the antiderivative
        try:
            antiderivative = integrate(expr, var_symbol)
            
            # Identify integration techniques used
            technique_explanation = self._identify_integration_techniques(expr, var_symbol)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Find the antiderivative",
                explanation=technique_explanation,
                mathematical_expression=f"∫ {expr} d{variable}",
                intermediate_result=f"{antiderivative} + C" if not is_definite else str(antiderivative)
            ))
            step_num += 1
            
            # Step 3: Evaluate definite integral if needed
            if is_definite:
                try:
                    # Parse limits
                    lower = sp.sympify(lower_limit) if lower_limit != '-oo' else -oo
                    upper = sp.sympify(upper_limit) if upper_limit != 'oo' else oo
                    
                    definite_result = integrate(expr, (var_symbol, lower, upper))
                    
                    steps.append(SolutionStep(
                        step_number=step_num,
                        operation="Evaluate definite integral",
                        explanation=f"Apply the Fundamental Theorem of Calculus: F({upper_limit}) - F({lower_limit})",
                        mathematical_expression=f"[{antiderivative}]_{{{lower_limit}}}^{{{upper_limit}}}",
                        intermediate_result=str(definite_result)
                    ))
                    
                    final_answer = str(definite_result)
                except:
                    final_answer = f"∫[{lower_limit} to {upper_limit}] {expr} d{variable} (evaluation failed)"
            else:
                final_answer = f"{antiderivative} + C"
                
        except Exception as e:
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Integration attempt",
                explanation="This integral may require advanced techniques or cannot be expressed in elementary functions",
                mathematical_expression=f"∫ {expr} d{variable}",
                intermediate_result="Integration not possible with elementary functions"
            ))
            final_answer = f"∫ {expr} d{variable} (cannot integrate)"
        
        return steps, final_answer, "Integration"
    
    def _solve_limit(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve limit problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the function and limit point
        function_text, limit_point, direction = self._extract_limit_info(problem)
        if not function_text:
            raise ComputationError("No function found for limit calculation")
        
        # Parse the function
        expr = parse_expr(function_text, transformations=self.transformations)
        
        # Get the variable
        variable = self._get_calculus_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Parse limit point
        if limit_point == 'infinity' or limit_point == 'oo':
            limit_val = oo
        elif limit_point == '-infinity' or limit_point == '-oo':
            limit_val = -oo
        else:
            limit_val = sp.sympify(limit_point)
        
        # Step 1: Show the limit problem
        direction_text = ""
        if direction == 'left':
            direction_text = "⁻"
        elif direction == 'right':
            direction_text = "⁺"
        
        limit_notation = f"lim[{variable}→{limit_point}{direction_text}] {expr}"
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given limit",
            explanation=f"We need to find the limit of {expr} as {variable} approaches {limit_point}",
            mathematical_expression=limit_notation,
            intermediate_result=limit_notation
        ))
        step_num += 1
        
        # Step 2: Check for direct substitution
        try:
            direct_sub = expr.subs(var_symbol, limit_val)
            if direct_sub.is_finite and not direct_sub.has(oo):
                steps.append(SolutionStep(
                    step_number=step_num,
                    operation="Direct substitution",
                    explanation=f"Substitute {variable} = {limit_point} directly into the function",
                    mathematical_expression=f"f({limit_point}) = {direct_sub}",
                    intermediate_result=str(direct_sub)
                ))
                final_answer = str(direct_sub)
            else:
                raise ValueError("Indeterminate form")
        except:
            # Step 2: Analyze indeterminate form
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Check for indeterminate form",
                explanation="Direct substitution gives an indeterminate form, need to use limit techniques",
                mathematical_expression=f"Direct substitution: {expr.subs(var_symbol, limit_val)}",
                intermediate_result="Indeterminate form detected"
            ))
            step_num += 1
            
            # Step 3: Calculate the limit
            try:
                if direction == 'left':
                    limit_result = limit(expr, var_symbol, limit_val, '-')
                elif direction == 'right':
                    limit_result = limit(expr, var_symbol, limit_val, '+')
                else:
                    limit_result = limit(expr, var_symbol, limit_val)
                
                technique = self._identify_limit_technique(expr, var_symbol, limit_val)
                
                steps.append(SolutionStep(
                    step_number=step_num,
                    operation="Apply limit techniques",
                    explanation=technique,
                    mathematical_expression=f"lim[{variable}→{limit_point}] {expr}",
                    intermediate_result=str(limit_result)
                ))
                
                final_answer = str(limit_result)
            except:
                final_answer = "Limit does not exist or cannot be computed"
        
        return steps, final_answer, "Limit evaluation"
    
    def _solve_optimization(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve optimization problems step by step."""
        steps = []
        step_num = 1
        
        # Extract the function to optimize
        function_text = self._extract_calculus_function(problem, 'optimization')
        if not function_text:
            raise ComputationError("No function found to optimize")
        
        # Parse the function
        expr = parse_expr(function_text, transformations=self.transformations)
        
        # Get the variable
        variable = self._get_calculus_variable(problem)
        var_symbol = sp.Symbol(variable)
        
        # Determine if we're finding max or min
        is_maximum = 'max' in problem.original_text.lower()
        optimization_type = "maximum" if is_maximum else "minimum"
        
        # Step 1: Show the function to optimize
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given function",
            explanation=f"We need to find the {optimization_type} of f({variable}) = {expr}",
            mathematical_expression=f"f({variable}) = {expr}",
            intermediate_result=f"f({variable}) = {expr}"
        ))
        step_num += 1
        
        # Step 2: Find the first derivative
        first_derivative = diff(expr, var_symbol)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Find the first derivative",
            explanation="To find critical points, we need f'(x) = 0",
            mathematical_expression=f"f'({variable}) = {first_derivative}",
            intermediate_result=f"f'({variable}) = {first_derivative}"
        ))
        step_num += 1
        
        # Step 3: Solve f'(x) = 0 for critical points
        critical_points = solve(first_derivative, var_symbol)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Find critical points",
            explanation="Solve f'(x) = 0 to find where the function has horizontal tangents",
            mathematical_expression=f"{first_derivative} = 0",
            intermediate_result=f"Critical points: {critical_points}"
        ))
        step_num += 1
        
        # Step 4: Use second derivative test
        if critical_points:
            second_derivative = diff(first_derivative, var_symbol)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Find second derivative",
                explanation="Use the second derivative test to classify critical points",
                mathematical_expression=f"f''({variable}) = {second_derivative}",
                intermediate_result=f"f''({variable}) = {second_derivative}"
            ))
            step_num += 1
            
            # Evaluate second derivative at critical points
            classifications = []
            for cp in critical_points:
                if cp.is_real:
                    second_deriv_value = second_derivative.subs(var_symbol, cp)
                    if second_deriv_value > 0:
                        classifications.append(f"{variable} = {cp}: local minimum")
                    elif second_deriv_value < 0:
                        classifications.append(f"{variable} = {cp}: local maximum")
                    else:
                        classifications.append(f"{variable} = {cp}: inconclusive")
            
            classification_text = "; ".join(classifications)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Classify critical points",
                explanation="Apply second derivative test: f''(x) > 0 → minimum, f''(x) < 0 → maximum",
                mathematical_expression="Second derivative test",
                intermediate_result=classification_text
            ))
            
            # Find the optimal point based on what we're looking for
            optimal_points = []
            for cp in critical_points:
                if cp.is_real:
                    second_deriv_value = second_derivative.subs(var_symbol, cp)
                    if (is_maximum and second_deriv_value < 0) or (not is_maximum and second_deriv_value > 0):
                        function_value = expr.subs(var_symbol, cp)
                        optimal_points.append((cp, function_value))
            
            if optimal_points:
                if len(optimal_points) == 1:
                    x_opt, y_opt = optimal_points[0]
                    final_answer = f"{optimization_type} at {variable} = {x_opt}, f({x_opt}) = {y_opt}"
                else:
                    # Multiple optima - choose based on function values
                    if is_maximum:
                        best = max(optimal_points, key=lambda x: x[1])
                    else:
                        best = min(optimal_points, key=lambda x: x[1])
                    x_opt, y_opt = best
                    final_answer = f"Global {optimization_type} at {variable} = {x_opt}, f({x_opt}) = {y_opt}"
            else:
                final_answer = f"No {optimization_type} found among critical points"
        else:
            final_answer = "No critical points found"
        
        return steps, final_answer, "Optimization"
    
    def _solve_general_problem(self, problem: ParsedProblem) -> tuple[List[SolutionStep], str, str]:
        """Solve general mathematical problems."""
        steps = []
        
        # For general problems, try to extract and solve any equations
        if problem.metadata.get('has_equations', False):
            equation_text = self._extract_main_equation(problem)
            if equation_text and '=' in equation_text:
                try:
                    return self._solve_algebraic_equation(problem)
                except:
                    pass
        
        # If no equations, try to evaluate expressions
        expression_text = self._extract_main_expression(problem)
        if expression_text:
            try:
                expr = sp.sympify(expression_text)
                simplified = simplify(expr)
                
                steps.append(SolutionStep(
                    step_number=1,
                    operation="Evaluate expression",
                    explanation="Simplify the given mathematical expression",
                    mathematical_expression=str(expr),
                    intermediate_result=str(simplified)
                ))
                
                return steps, str(simplified), "Expression evaluation"
            except:
                pass
        
        # Fallback for unsupported problems
        steps.append(SolutionStep(
            step_number=1,
            operation="Problem analysis",
            explanation="This problem type is not yet fully supported",
            mathematical_expression=problem.original_text,
            intermediate_result="Solution method not implemented"
        ))
        
        return steps, "Solution not available", "General analysis"
    
    # Helper methods
    
    def _extract_main_equation(self, problem: ParsedProblem) -> Optional[str]:
        """Extract the main equation from the problem."""
        for expr in problem.expressions:
            if '=' in expr:
                return expr.strip()
        return None
    
    def _extract_main_expression(self, problem: ParsedProblem) -> Optional[str]:
        """Extract the main expression from the problem."""
        if problem.expressions:
            # Prioritize longer/more complex expressions that don't contain '='
            valid_expressions = [expr for expr in problem.expressions if '=' not in expr]
            if valid_expressions:
                # Sort by length and complexity (prefer expressions with more operations and variables)
                def complexity_score(expr):
                    score = len(expr) + expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/') + expr.count('^')
                    # Bonus for having variables (letters)
                    score += sum(1 for c in expr if c.isalpha() and c not in ['e', 'i']) * 2
                    # Penalty for ending with operators (incomplete expressions)
                    if expr.strip().endswith(('+', '-', '*', '/', '^')):
                        score -= 20
                    return score
                
                function_text = sorted(valid_expressions, key=complexity_score, reverse=True)[0].strip()
                # Clean up punctuation
                function_text = function_text.rstrip('?.,;:!')
                return function_text
            # If all contain '=', return the first one
            return problem.expressions[0].strip()
        return None
    
    def _extract_system_equations(self, problem: ParsedProblem) -> List[str]:
        """Extract equations from a system of equations problem."""
        equations = []
        for expr in problem.expressions:
            if '=' in expr:
                equations.append(expr.strip())
        return equations
    
    def _get_solve_variable(self, problem: ParsedProblem) -> str:
        """Get the variable to solve for."""
        if problem.variables:
            return problem.variables[0]  # Default to first variable
        return 'x'  # Default fallback
    
    def _calculate_confidence_score(self, problem: ParsedProblem, steps: List[SolutionStep]) -> float:
        """Calculate confidence score for the solution."""
        base_score = 0.8
        
        # Adjust based on problem type
        if problem.problem_type in ['linear_equation', 'quadratic_equation']:
            base_score = 0.95
        elif problem.problem_type in ['factoring', 'expansion', 'simplification']:
            base_score = 0.9
        elif problem.problem_type in ['derivative', 'integral', 'limit']:
            base_score = 0.9  # High confidence for basic calculus operations
        elif problem.problem_type == 'optimization':
            base_score = 0.85
        elif problem.problem_type == 'system_of_equations':
            base_score = 0.85
        
        # Adjust based on number of steps (more steps might indicate complexity)
        if len(steps) > 5:
            base_score -= 0.05
        
        return max(0.5, min(1.0, base_score))
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        try:
            # Try to parse as SymPy expression and simplify
            expr = sp.sympify(answer)
            return str(sp.simplify(expr))
        except:
            # If parsing fails, return cleaned string
            return answer.strip().lower()
    
    def _are_answers_equivalent(self, answer1: str, answer2: str) -> bool:
        """Check if two answers are mathematically equivalent."""
        try:
            expr1 = sp.sympify(answer1)
            expr2 = sp.sympify(answer2)
            return sp.simplify(expr1 - expr2) == 0
        except:
            return answer1.strip().lower() == answer2.strip().lower()
    
    def _calculate_partial_credit(self, problem: ParsedProblem, user_answer: str, 
                                correct_answer: str, is_correct: bool) -> float:
        """Calculate partial credit for an answer."""
        if is_correct:
            return 1.0
        
        # For now, simple partial credit logic
        # Could be enhanced with more sophisticated analysis
        try:
            user_expr = sp.sympify(user_answer)
            correct_expr = sp.sympify(correct_answer)
            
            # Check if the form is similar (e.g., same degree polynomial)
            if user_expr.is_polynomial() and correct_expr.is_polynomial():
                return 0.3  # Some credit for correct form
            
            return 0.1  # Minimal credit for attempting
        except:
            return 0.0
    
    def _generate_validation_explanation(self, problem: ParsedProblem, user_answer: str,
                                       correct_answer: str, is_correct: bool) -> str:
        """Generate explanation for answer validation."""
        if is_correct:
            return f"Correct! Your answer {user_answer} is mathematically equivalent to {correct_answer}."
        else:
            return f"Incorrect. Your answer was {user_answer}, but the correct answer is {correct_answer}. " \
                   f"Please review the solution steps to understand the correct approach."
    
    # Calculus helper methods
    
    def _extract_calculus_function(self, problem: ParsedProblem, operation_type: str) -> Optional[str]:
        """Extract the function from a calculus problem."""
        # Look for function in expressions first - prioritize longer/more complex expressions
        valid_expressions = [expr for expr in problem.expressions if '=' not in expr]
        if valid_expressions:
            # Sort by length and complexity (prefer expressions with more operations and variables)
            def complexity_score(expr):
                score = len(expr) + expr.count('+') + expr.count('-') + expr.count('*') + expr.count('/') + expr.count('^')
                # Bonus for having variables (letters)
                score += sum(1 for c in expr if c.isalpha() and c not in ['e', 'i']) * 2
                # Penalty for ending with operators (incomplete expressions)
                if expr.strip().endswith(('+', '-', '*', '/', '^')):
                    score -= 20
                return score
            
            function_text = sorted(valid_expressions, key=complexity_score, reverse=True)[0].strip()
            # Clean up punctuation
            function_text = function_text.rstrip('?.,;:!')
            return function_text
        
        # Try to extract from original text using patterns
        import re
        text = problem.original_text
        
        patterns = {
            'derivative': [
                r'derivative of\s+(.+?)(?:\s+with respect to|\s*$)',
                r'd/dx\s*\[(.+?)\]',
                r'd/dx\s+(.+?)(?:\s+|$)',
                r'differentiate\s+(.+?)(?:\s+with respect to|\s*$)',
                r'find.*?derivative.*?of\s+(.+?)(?:\s+|$)',
                r'find\s+the\s+derivative\s+of\s+(.+?)(?:\s+|$)'
            ],
            'integral': [
                r'integral of\s+(.+?)(?:\s+with respect to|\s+from|\s*$)',
                r'∫\s*(.+?)\s*d[a-z]',
                r'integrate\s+(.+?)(?:\s+with respect to|\s+from|\s*$)',
                r'antiderivative of\s+(.+?)(?:\s+|$)',
                r'find.*?integral.*?of\s+(.+?)(?:\s+from|\s*$)'
            ],
            'optimization': [
                r'(?:minimum|maximum)\s+of\s+(.+?)(?:\s+|$)',
                r'optimize\s+(.+?)(?:\s+|$)',
                r'find.*?(?:minimum|maximum).*?of\s+(.+?)(?:\s+|$)',
                r'f\(.*?\)\s*=\s*(.+?)(?:\s+|$)'
            ]
        }
        
        for pattern in patterns.get(operation_type, []):
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                function_text = match.group(1).strip()
                # Clean up common endings and punctuation
                function_text = re.sub(r'\s+(with respect to.*|from.*|as.*|where.*)$', '', function_text, flags=re.IGNORECASE)
                function_text = function_text.rstrip('?.,;:!')
                return function_text
        
        # Fallback: try to extract any mathematical expression
        # Look for expressions with variables
        math_expr_patterns = [
            r'([a-z]\^?\d*[\+\-\*/]*[a-z\d\^]*[\+\-\*/]*[a-z\d\^]*)',
            r'(sin\([^)]+\))',
            r'(cos\([^)]+\))',
            r'(tan\([^)]+\))',
            r'(exp\([^)]+\))',
            r'(log\([^)]+\))',
            r'(\d*[a-z]\^?\d*)',
        ]
        
        for pattern in math_expr_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Return the longest match
                return max(matches, key=len)
        
        return None
    
    def _get_calculus_variable(self, problem: ParsedProblem) -> str:
        """Get the variable for calculus operations."""
        # Try to extract from text first
        import re
        match = re.search(r'with respect to\s+([a-z])', problem.original_text, re.IGNORECASE)
        if match:
            return match.group(1)
        
        # Look for variables that appear in the extracted function
        if problem.variables:
            function_text = self._extract_calculus_function(problem, 'derivative')  # Use any operation type
            if function_text:
                # Check which variables from the problem actually appear in the function
                for var in problem.variables:
                    if var in function_text and var not in ['f', 'a', 'i', 'n', 'd', 't', 'h', 'e', 'm', 'u', 'o']:
                        return var
            
            # Filter out common non-mathematical variables
            math_vars = [v for v in problem.variables if v not in ['f', 'a', 'i', 'n', 'd', 't', 'h', 'e', 'm', 'u', 'o']]
            if math_vars:
                return math_vars[0]
        
        # Look for common mathematical variables in the text
        for var in ['x', 'y', 'z', 't']:
            if var in problem.original_text.lower():
                return var
        
        return 'x'  # Default
    
    def _extract_integral_limits(self, problem: ParsedProblem) -> tuple[bool, Optional[str], Optional[str]]:
        """Extract integration limits if present."""
        import re
        
        # Look for definite integral patterns
        patterns = [
            r'from\s+([^to\s]+)\s+to\s+([^\s]+)',
            r'∫\[([^to\]]+)\s+to\s+([^\]]+)\]',
            r'limits?\s+([^to\s]+)\s+to\s+([^\s]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, problem.original_text, re.IGNORECASE)
            if match:
                return True, match.group(1).strip(), match.group(2).strip()
        
        return False, None, None
    
    def _extract_limit_info(self, problem: ParsedProblem) -> tuple[Optional[str], str, Optional[str]]:
        """Extract function, limit point, and direction from limit problem."""
        import re
        text = problem.original_text
        
        # Extract function
        function_text = None
        
        # First try expressions - prioritize fractions and complex expressions
        expressions_by_complexity = sorted(problem.expressions, key=lambda x: len(x), reverse=True)
        for expr in expressions_by_complexity:
            if '=' not in expr:
                function_text = expr.strip()
                break
        
        if not function_text:
            # Try various limit patterns
            patterns = [
                r'limit.*?of\s+(.+?)\s+as\s+[a-z]\s+approaches',
                r'lim.*?of\s+(.+?)\s+as\s+[a-z]',
                r'find\s+lim.*?([a-z]→[^/\s]+)\s+(.+?)(?:\s|$)',
                r'lim\s*\[?[a-z]→[^\]]*\]?\s*(.+?)(?:\s|$)',
                r'limit\s+(.+?)\s+as\s+[a-z]',
                r'lim.*?([^/\s]+/[^/\s]+)',  # For fractions like 1/x
                r'([a-z]\^?\d*[\+\-\*/]*[a-z\d\^]*)',  # General math expressions
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    if 'lim' in pattern and '→' in pattern:
                        # Special case for "lim x→∞ 1/x" format
                        if len(match.groups()) >= 2:
                            function_text = match.group(2).strip()
                        else:
                            function_text = match.group(1).strip()
                    else:
                        function_text = match.group(1).strip()
                    break
        
        # Extract limit point
        limit_point = '0'  # Default
        direction = None
        
        # Look for patterns like "as x approaches 0", "x → infinity", etc.
        patterns = [
            r'as\s+[a-z]\s+approaches\s+([^\s,]+)',
            r'[a-z]\s*→\s*([^\s,]+)',
            r'[a-z]\s*->\s*([^\s,]+)',
            r'x→([^\s,]+)',
            r'approaches\s+([^\s,]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                limit_point = match.group(1).strip()
                break
        
        # Normalize infinity representations
        if limit_point.lower() in ['infinity', '∞', 'inf']:
            limit_point = 'infinity'
        elif limit_point.lower() in ['-infinity', '-∞', '-inf']:
            limit_point = '-infinity'
        
        # Check for one-sided limits
        if 'left' in text.lower() or '⁻' in text:
            direction = 'left'
        elif 'right' in text.lower() or '⁺' in text:
            direction = 'right'
        
        return function_text, limit_point, direction
    
    def _identify_derivative_rules(self, expr, variable) -> str:
        """Identify which differentiation rules are being applied."""
        rules = []
        
        if expr.is_Add:
            rules.append("sum rule")
        if expr.is_Mul:
            rules.append("product rule")
        if expr.has(sp.sin, sp.cos, sp.tan):
            rules.append("trigonometric rules")
        if expr.has(sp.exp):
            rules.append("exponential rule")
        if expr.has(sp.log):
            rules.append("logarithmic rule")
        if expr.is_Pow:
            rules.append("power rule")
        
        if not rules:
            rules.append("basic differentiation rules")
        
        return f"Apply the {', '.join(rules)} to find the derivative"
    
    def _identify_integration_techniques(self, expr, variable) -> str:
        """Identify which integration techniques are being used."""
        techniques = []
        
        if expr.is_Add:
            techniques.append("sum rule for integration")
        if expr.is_Pow and expr.exp.is_number:
            techniques.append("power rule for integration")
        if expr.has(sp.sin, sp.cos, sp.tan):
            techniques.append("trigonometric integration")
        if expr.has(sp.exp):
            techniques.append("exponential integration")
        if expr.has(sp.log):
            techniques.append("integration involving logarithms")
        
        if not techniques:
            techniques.append("basic integration techniques")
        
        return f"Use {', '.join(techniques)} to find the antiderivative"
    
    def _identify_limit_technique(self, expr, variable, limit_point) -> str:
        """Identify which limit technique is being used."""
        if limit_point == oo or limit_point == -oo:
            return "Analyze the behavior as the variable approaches infinity"
        elif expr.has(sp.sin, sp.cos):
            return "Apply trigonometric limit properties"
        elif '0/0' in str(expr.subs(variable, limit_point)):
            return "Apply L'Hôpital's rule or algebraic manipulation for indeterminate form"
        else:
            return "Apply limit properties and algebraic techniques"
    
    def generate_visualization_data(self, problem: ParsedProblem, solution: StepSolution) -> Dict[str, Any]:
        """Generate visualization data for calculus problems."""
        viz_data = {
            'type': 'none',
            'data': {}
        }
        
        try:
            if problem.problem_type == 'derivative':
                viz_data = self._generate_derivative_visualization(problem, solution)
            elif problem.problem_type == 'integral':
                viz_data = self._generate_integral_visualization(problem, solution)
            elif problem.problem_type == 'optimization':
                viz_data = self._generate_optimization_visualization(problem, solution)
            elif problem.problem_type == 'limit':
                viz_data = self._generate_limit_visualization(problem, solution)
        except Exception as e:
            # If visualization fails, return empty data
            viz_data['error'] = str(e)
        
        return viz_data
    
    def _generate_derivative_visualization(self, problem: ParsedProblem, solution: StepSolution) -> Dict[str, Any]:
        """Generate visualization data for derivative problems."""
        function_text = self._extract_calculus_function(problem, 'derivative')
        if not function_text:
            return {'type': 'none', 'data': {}}
        
        try:
            expr = parse_expr(function_text, transformations=self.transformations)
            variable = self._get_calculus_variable(problem)
            var_symbol = sp.Symbol(variable)
            
            # Generate function and derivative data points
            x_vals = list(range(-10, 11))
            y_vals = []
            dy_vals = []
            
            derivative = diff(expr, var_symbol)
            
            for x in x_vals:
                try:
                    y = float(expr.subs(var_symbol, x))
                    dy = float(derivative.subs(var_symbol, x))
                    y_vals.append(y)
                    dy_vals.append(dy)
                except:
                    y_vals.append(None)
                    dy_vals.append(None)
            
            return {
                'type': 'derivative',
                'data': {
                    'x_values': x_vals,
                    'function_values': y_vals,
                    'derivative_values': dy_vals,
                    'function_expr': str(expr),
                    'derivative_expr': str(derivative),
                    'variable': variable
                }
            }
        except:
            return {'type': 'none', 'data': {}}
    
    def _generate_integral_visualization(self, problem: ParsedProblem, solution: StepSolution) -> Dict[str, Any]:
        """Generate visualization data for integral problems."""
        function_text = self._extract_calculus_function(problem, 'integral')
        if not function_text:
            return {'type': 'none', 'data': {}}
        
        try:
            expr = parse_expr(function_text, transformations=self.transformations)
            variable = self._get_calculus_variable(problem)
            var_symbol = sp.Symbol(variable)
            
            # Check for definite integral limits
            is_definite, lower_limit, upper_limit = self._extract_integral_limits(problem)
            
            # Generate function data points
            if is_definite:
                try:
                    lower = float(sp.sympify(lower_limit))
                    upper = float(sp.sympify(upper_limit))
                    x_range = [lower - 1, upper + 1]
                except:
                    x_range = [-5, 5]
            else:
                x_range = [-5, 5]
            
            x_vals = [x_range[0] + i * (x_range[1] - x_range[0]) / 100 for i in range(101)]
            y_vals = []
            
            for x in x_vals:
                try:
                    y = float(expr.subs(var_symbol, x))
                    y_vals.append(y)
                except:
                    y_vals.append(None)
            
            viz_data = {
                'type': 'integral',
                'data': {
                    'x_values': x_vals,
                    'function_values': y_vals,
                    'function_expr': str(expr),
                    'variable': variable
                }
            }
            
            if is_definite:
                viz_data['data']['definite'] = True
                viz_data['data']['lower_limit'] = lower_limit
                viz_data['data']['upper_limit'] = upper_limit
            
            return viz_data
        except:
            return {'type': 'none', 'data': {}}
    
    def _generate_optimization_visualization(self, problem: ParsedProblem, solution: StepSolution) -> Dict[str, Any]:
        """Generate visualization data for optimization problems."""
        function_text = self._extract_calculus_function(problem, 'optimization')
        if not function_text:
            return {'type': 'none', 'data': {}}
        
        try:
            expr = parse_expr(function_text, transformations=self.transformations)
            variable = self._get_calculus_variable(problem)
            var_symbol = sp.Symbol(variable)
            
            # Generate function data points
            x_vals = list(range(-10, 11))
            y_vals = []
            
            for x in x_vals:
                try:
                    y = float(expr.subs(var_symbol, x))
                    y_vals.append(y)
                except:
                    y_vals.append(None)
            
            # Find critical points
            derivative = diff(expr, var_symbol)
            critical_points = solve(derivative, var_symbol)
            
            critical_data = []
            for cp in critical_points:
                if cp.is_real:
                    try:
                        x_val = float(cp)
                        y_val = float(expr.subs(var_symbol, cp))
                        critical_data.append({'x': x_val, 'y': y_val})
                    except:
                        pass
            
            return {
                'type': 'optimization',
                'data': {
                    'x_values': x_vals,
                    'function_values': y_vals,
                    'critical_points': critical_data,
                    'function_expr': str(expr),
                    'derivative_expr': str(derivative),
                    'variable': variable
                }
            }
        except:
            return {'type': 'none', 'data': {}}
    
    def _generate_limit_visualization(self, problem: ParsedProblem, solution: StepSolution) -> Dict[str, Any]:
        """Generate visualization data for limit problems."""
        function_text, limit_point, direction = self._extract_limit_info(problem)
        if not function_text:
            return {'type': 'none', 'data': {}}
        
        try:
            expr = parse_expr(function_text, transformations=self.transformations)
            variable = self._get_calculus_variable(problem)
            var_symbol = sp.Symbol(variable)
            
            # Parse limit point
            if limit_point == 'infinity' or limit_point == 'oo':
                limit_val = float('inf')
                x_range = [-10, 10]
            elif limit_point == '-infinity' or limit_point == '-oo':
                limit_val = float('-inf')
                x_range = [-10, 10]
            else:
                try:
                    limit_val = float(sp.sympify(limit_point))
                    x_range = [limit_val - 2, limit_val + 2]
                except:
                    limit_val = 0
                    x_range = [-2, 2]
            
            # Generate function data points
            x_vals = [x_range[0] + i * (x_range[1] - x_range[0]) / 200 for i in range(201)]
            y_vals = []
            
            for x in x_vals:
                try:
                    if abs(x - limit_val) < 1e-10:  # Skip the exact limit point
                        y_vals.append(None)
                    else:
                        y = float(expr.subs(var_symbol, x))
                        y_vals.append(y)
                except:
                    y_vals.append(None)
            
            return {
                'type': 'limit',
                'data': {
                    'x_values': x_vals,
                    'function_values': y_vals,
                    'function_expr': str(function_text),
                    'limit_point': limit_point,
                    'direction': direction,
                    'variable': variable
                }
            }
        except:
            return {'type': 'none', 'data': {}}