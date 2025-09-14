"""
Specialized AI/ML mathematics module for eigenvalue analysis, optimization, and neural network mathematics.
Provides educational tools with AI context explanations connecting to machine learning applications.
"""

import numpy as np
import sympy as sp
from sympy import symbols, Matrix, eye, solve, simplify, sqrt, I, re, im
from typing import List, Dict, Any, Optional, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, ValidationResult,
    MathDomain, DifficultyLevel, ComputationError
)


class EigenAnalysisResult:
    """Result of eigenvalue and eigenvector analysis."""
    
    def __init__(self, 
                 matrix: Matrix,
                 eigenvalues: List[complex],
                 eigenvectors: List[Matrix],
                 characteristic_polynomial: sp.Expr,
                 geometric_multiplicities: List[int],
                 algebraic_multiplicities: List[int]):
        self.matrix = matrix
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        self.characteristic_polynomial = characteristic_polynomial
        self.geometric_multiplicities = geometric_multiplicities
        self.algebraic_multiplicities = algebraic_multiplicities
        
    def is_diagonalizable(self) -> bool:
        """Check if matrix is diagonalizable."""
        return sum(self.geometric_multiplicities) == self.matrix.rows
    
    def get_dominant_eigenvalue(self) -> complex:
        """Get eigenvalue with largest absolute value."""
        return max(self.eigenvalues, key=abs)
    
    def get_ml_interpretation(self) -> Dict[str, str]:
        """Get machine learning interpretation of eigenanalysis."""
        interpretations = {}
        
        # Dominant eigenvalue interpretation
        dominant = self.get_dominant_eigenvalue()
        if abs(dominant) > 1:
            interpretations['stability'] = "System is unstable (eigenvalue magnitude > 1)"
        elif abs(dominant) < 1:
            interpretations['stability'] = "System is stable (eigenvalue magnitude < 1)"
        else:
            interpretations['stability'] = "System is marginally stable (eigenvalue magnitude = 1)"
        
        # Real vs complex eigenvalues
        real_eigenvals = [ev for ev in self.eigenvalues if im(ev) == 0]
        complex_eigenvals = [ev for ev in self.eigenvalues if im(ev) != 0]
        
        if complex_eigenvals:
            interpretations['dynamics'] = "Complex eigenvalues indicate oscillatory behavior"
        else:
            interpretations['dynamics'] = "Real eigenvalues indicate exponential growth/decay"
        
        # Diagonalizability
        if self.is_diagonalizable():
            interpretations['diagonalization'] = "Matrix is diagonalizable - can be decomposed for efficient computation"
        else:
            interpretations['diagonalization'] = "Matrix is not diagonalizable - Jordan form needed"
        
        return interpretations


class AIMLMathematics:
    """Specialized AI/ML mathematics tools."""
    
    def __init__(self):
        """Initialize AI/ML mathematics module."""
        pass
    
    def analyze_eigenvalues_step_by_step(self, matrix_data: Union[List[List], Matrix]) -> Tuple[List[SolutionStep], EigenAnalysisResult]:
        """
        Perform step-by-step eigenvalue and eigenvector analysis.
        
        Args:
            matrix_data: Input matrix as list of lists or SymPy Matrix
            
        Returns:
            Tuple of solution steps and analysis result
        """
        steps = []
        step_num = 1
        
        # Convert to SymPy Matrix if needed
        if isinstance(matrix_data, list):
            matrix = Matrix(matrix_data)
        else:
            matrix = matrix_data
        
        # Step 1: Show the original matrix
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given matrix",
            explanation="We need to find the eigenvalues and eigenvectors of this matrix",
            mathematical_expression=f"A = {matrix}",
            intermediate_result=f"Matrix A is {matrix.rows}×{matrix.cols}"
        ))
        step_num += 1
        
        # Step 2: Set up characteristic equation
        n = matrix.rows
        lambda_sym = symbols('lambda')
        identity = eye(n)
        char_matrix = matrix - lambda_sym * identity
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Set up characteristic matrix",
            explanation="The characteristic matrix is A - λI, where λ represents eigenvalues",
            mathematical_expression=f"A - λI = {char_matrix}",
            intermediate_result=f"Characteristic matrix: {char_matrix}"
        ))
        step_num += 1
        
        # Step 3: Calculate characteristic polynomial
        char_poly = char_matrix.det()
        char_poly_simplified = simplify(char_poly)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Calculate characteristic polynomial",
            explanation="The characteristic polynomial is det(A - λI) = 0",
            mathematical_expression=f"det(A - λI) = {char_poly}",
            intermediate_result=f"Characteristic polynomial: {char_poly_simplified} = 0"
        ))
        step_num += 1
        
        # Step 4: Solve for eigenvalues with multiplicities
        eigenvalues_with_mult = solve(char_poly_simplified, lambda_sym, multiple=True)
        
        # If multiple=True doesn't work, use roots to get multiplicities
        try:
            from sympy import roots
            eigenval_roots = roots(char_poly_simplified, lambda_sym)
            eigenvalues = []
            for eigenval, mult in eigenval_roots.items():
                eigenvalues.extend([eigenval] * mult)
        except:
            # Fallback to regular solve
            eigenvalues = solve(char_poly_simplified, lambda_sym)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Solve for eigenvalues",
            explanation="Solve the characteristic equation to find eigenvalues",
            mathematical_expression=f"solve({char_poly_simplified} = 0, λ)",
            intermediate_result=f"Eigenvalues: λ = {eigenvalues}"
        ))
        step_num += 1
        
        # Step 5: Find eigenvectors for each eigenvalue
        eigenvectors = []
        geometric_mults = []
        algebraic_mults = []
        
        # Count algebraic multiplicities properly
        from collections import Counter
        eigenval_counts = Counter(eigenvalues)
        unique_eigenvals = list(eigenval_counts.keys())
        
        for eigenval in unique_eigenvals:
            algebraic_mult = eigenval_counts[eigenval]
            algebraic_mults.append(algebraic_mult)
            
            # Find eigenvectors
            eigenvector_matrix = matrix - eigenval * identity
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation=f"Find eigenvectors for λ = {eigenval}",
                explanation=f"Solve (A - {eigenval}I)v = 0 to find eigenvectors",
                mathematical_expression=f"(A - {eigenval}I) = {eigenvector_matrix}",
                intermediate_result=f"Need to solve: {eigenvector_matrix} * v = 0"
            ))
            step_num += 1
            
            # Solve the homogeneous system
            nullspace = eigenvector_matrix.nullspace()
            geometric_mult = len(nullspace)
            geometric_mults.append(geometric_mult)
            
            if nullspace:
                eigenvectors.extend(nullspace)
                eigenvector_str = ", ".join(str(vec) for vec in nullspace)
                
                steps.append(SolutionStep(
                    step_number=step_num,
                    operation=f"Eigenvectors for λ = {eigenval}",
                    explanation=f"The nullspace gives us {geometric_mult} linearly independent eigenvector(s)",
                    mathematical_expression=f"Eigenvector(s): {eigenvector_str}",
                    intermediate_result=f"Geometric multiplicity: {geometric_mult}, Algebraic multiplicity: {algebraic_mult}"
                ))
                step_num += 1
        
        # Create analysis result
        analysis_result = EigenAnalysisResult(
            matrix=matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            characteristic_polynomial=char_poly_simplified,
            geometric_multiplicities=geometric_mults,
            algebraic_multiplicities=algebraic_mults
        )
        
        # Step 6: AI/ML interpretation
        ml_interpretation = analysis_result.get_ml_interpretation()
        
        interpretation_text = []
        for key, value in ml_interpretation.items():
            interpretation_text.append(f"{key.title()}: {value}")
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="AI/ML Context and Applications",
            explanation="Understanding eigenvalues in machine learning and AI applications",
            mathematical_expression="Eigenanalysis Applications",
            intermediate_result="; ".join(interpretation_text)
        ))
        
        return steps, analysis_result
    
    def generate_eigenspace_visualization(self, analysis_result: EigenAnalysisResult) -> Dict[str, Any]:
        """
        Generate visualization data for eigenspaces.
        
        Args:
            analysis_result: Result from eigenvalue analysis
            
        Returns:
            Dictionary containing visualization data
        """
        if analysis_result.matrix.rows != 2:
            return {"error": "Visualization currently supports 2x2 matrices only"}
        
        # Extract 2x2 matrix elements
        matrix = analysis_result.matrix
        a, b = float(matrix[0, 0]), float(matrix[0, 1])
        c, d = float(matrix[1, 0]), float(matrix[1, 1])
        
        # Create visualization data
        viz_data = {
            "matrix": [[a, b], [c, d]],
            "eigenvalues": [],
            "eigenvectors": [],
            "eigenlines": [],
            "transformation_demo": {}
        }
        
        # Process eigenvalues and eigenvectors
        for i, (eigenval, eigenvec) in enumerate(zip(analysis_result.eigenvalues, analysis_result.eigenvectors)):
            if im(eigenval) == 0 and len(eigenvec) >= 2:  # Real eigenvalue with 2D eigenvector
                eigenval_real = float(re(eigenval))
                eigenvec_real = [float(re(eigenvec[0])), float(re(eigenvec[1]))]
                
                viz_data["eigenvalues"].append(eigenval_real)
                viz_data["eigenvectors"].append(eigenvec_real)
                
                # Generate eigenline (line through origin in direction of eigenvector)
                t_vals = np.linspace(-3, 3, 100)
                eigenline_x = [t * eigenvec_real[0] for t in t_vals]
                eigenline_y = [t * eigenvec_real[1] for t in t_vals]
                
                viz_data["eigenlines"].append({
                    "x": eigenline_x,
                    "y": eigenline_y,
                    "eigenvalue": eigenval_real,
                    "label": f"Eigenline λ={eigenval_real:.2f}"
                })
        
        # Generate transformation demonstration
        # Show how unit circle transforms under the matrix
        theta = np.linspace(0, 2*np.pi, 100)
        unit_circle_x = np.cos(theta)
        unit_circle_y = np.sin(theta)
        
        # Apply matrix transformation
        transformed_x = a * unit_circle_x + b * unit_circle_y
        transformed_y = c * unit_circle_x + d * unit_circle_y
        
        viz_data["transformation_demo"] = {
            "original_circle": {"x": unit_circle_x.tolist(), "y": unit_circle_y.tolist()},
            "transformed_shape": {"x": transformed_x.tolist(), "y": transformed_y.tolist()},
            "title": "Unit Circle Transformation"
        }
        
        return viz_data
    
    def explain_ml_applications(self, analysis_result: EigenAnalysisResult) -> Dict[str, str]:
        """
        Generate AI/ML context explanations for eigenanalysis.
        
        Args:
            analysis_result: Result from eigenvalue analysis
            
        Returns:
            Dictionary of ML application explanations
        """
        explanations = {}
        
        # Principal Component Analysis (PCA)
        explanations["PCA"] = """
        In Principal Component Analysis, eigenvalues represent the variance explained by each principal component.
        Larger eigenvalues indicate directions of maximum variance in the data.
        The corresponding eigenvectors define the principal component directions.
        """
        
        # Stability Analysis
        dominant_eigenval = analysis_result.get_dominant_eigenvalue()
        if abs(dominant_eigenval) > 1:
            stability_explanation = """
            In dynamical systems and neural networks, eigenvalues with magnitude > 1 indicate instability.
            This can lead to exploding gradients in deep learning or unstable system behavior.
            """
        else:
            stability_explanation = """
            Eigenvalues with magnitude < 1 indicate stable systems.
            In neural networks, this helps prevent vanishing/exploding gradient problems.
            """
        explanations["Stability"] = stability_explanation
        
        # Spectral Methods
        explanations["Spectral_Methods"] = """
        Eigendecomposition is fundamental to spectral methods in machine learning:
        - Graph neural networks use eigenvalues of the graph Laplacian
        - Spectral clustering uses eigenvectors to find data clusters
        - Kernel methods rely on eigendecomposition of kernel matrices
        """
        
        # Optimization
        explanations["Optimization"] = """
        In optimization (gradient descent, Newton's method):
        - Eigenvalues of the Hessian matrix determine convergence rates
        - Condition number (ratio of largest to smallest eigenvalue) affects optimization difficulty
        - Preconditioning techniques modify eigenvalue distribution for faster convergence
        """
        
        # Dimensionality Reduction
        explanations["Dimensionality_Reduction"] = """
        Eigenvalues help determine how many dimensions to keep:
        - Keep components with large eigenvalues (high variance)
        - Discard components with small eigenvalues (noise)
        - Cumulative eigenvalue sum indicates total variance preserved
        """
        
        return explanations
    
    def solve_eigenvalue_problem(self, problem: ParsedProblem) -> StepSolution:
        """
        Solve eigenvalue problems with AI/ML context.
        
        Args:
            problem: Parsed eigenvalue problem
            
        Returns:
            Complete step-by-step solution
        """
        try:
            # Extract matrix from problem
            matrix = self._extract_matrix_from_problem(problem)
            
            # Perform step-by-step analysis
            steps, analysis_result = self.analyze_eigenvalues_step_by_step(matrix)
            
            # Add ML context explanations
            ml_explanations = self.explain_ml_applications(analysis_result)
            
            # Add final step with comprehensive ML context
            final_step = SolutionStep(
                step_number=len(steps) + 1,
                operation="Machine Learning Applications Summary",
                explanation="How eigenanalysis connects to AI and machine learning",
                mathematical_expression="Eigenanalysis ↔ ML Applications",
                intermediate_result=f"Key applications: {', '.join(ml_explanations.keys())}"
            )
            steps.append(final_step)
            
            # Format final answer
            eigenvals_str = ", ".join(str(ev) for ev in analysis_result.eigenvalues)
            final_answer = f"Eigenvalues: {eigenvals_str}; Eigenvectors computed; ML applications identified"
            
            return StepSolution(
                problem_id=problem.id,
                steps=steps,
                final_answer=final_answer,
                solution_method="Eigenvalue analysis with AI/ML context",
                confidence_score=0.9,
                computation_time=0.0
            )
            
        except Exception as e:
            raise ComputationError(f"Failed to solve eigenvalue problem: {str(e)}")
    
    def _extract_matrix_from_problem(self, problem: ParsedProblem) -> Matrix:
        """Extract matrix from problem text or expressions."""
        # Try to find matrix in problem expressions
        for expr_text in problem.expressions:
            try:
                # Look for matrix notation like [[1,2],[3,4]]
                if '[[' in expr_text and ']]' in expr_text:
                    # Parse matrix notation
                    import ast
                    matrix_list = ast.literal_eval(expr_text)
                    return Matrix(matrix_list)
            except:
                continue
        
        # Try to parse from original text
        import re
        
        # Look for matrix patterns in text
        matrix_patterns = [
            r'\[\[([^\]]+)\]\]',  # [[1,2],[3,4]] format
            r'matrix\s*\(\s*\[([^\]]+)\]\s*\)',  # matrix([1,2],[3,4]) format
        ]
        
        for pattern in matrix_patterns:
            match = re.search(pattern, problem.original_text)
            if match:
                try:
                    # Extract and parse matrix data
                    matrix_str = match.group(0)
                    # Simple parsing for common formats
                    if '[[' in matrix_str:
                        import ast
                        matrix_list = ast.literal_eval(matrix_str)
                        return Matrix(matrix_list)
                except:
                    continue
        
        # Default 2x2 example matrix if none found
        return Matrix([[1, 2], [3, 4]])
    
    def validate_eigenvalue_answer(self, problem: ParsedProblem, user_answer: str, correct_result: EigenAnalysisResult) -> ValidationResult:
        """
        Validate user's eigenvalue computation.
        
        Args:
            problem: Original problem
            user_answer: User's submitted answer
            correct_result: Correct eigenanalysis result
            
        Returns:
            Validation result with detailed feedback
        """
        try:
            # Parse user's eigenvalues
            user_eigenvals = self._parse_eigenvalues_from_answer(user_answer)
            correct_eigenvals = [complex(ev) for ev in correct_result.eigenvalues]
            
            # Check if eigenvalues match (within tolerance)
            tolerance = 1e-10
            is_correct = self._compare_eigenvalue_sets(user_eigenvals, correct_eigenvals, tolerance)
            
            # Calculate partial credit
            partial_credit = self._calculate_eigenvalue_partial_credit(user_eigenvals, correct_eigenvals)
            
            # Generate explanation
            if is_correct:
                explanation = "Correct! Your eigenvalues match the expected solution."
            else:
                explanation = f"Incorrect. Expected eigenvalues: {correct_eigenvals}, Got: {user_eigenvals}"
                explanation += "\nRemember to solve the characteristic equation det(A - λI) = 0."
            
            return ValidationResult(
                is_correct=is_correct,
                user_answer=user_answer,
                correct_answer=str(correct_eigenvals),
                explanation=explanation,
                partial_credit=partial_credit
            )
            
        except Exception as e:
            return ValidationResult(
                is_correct=False,
                user_answer=user_answer,
                correct_answer="Error in validation",
                explanation=f"Could not validate eigenvalue answer: {str(e)}",
                partial_credit=0.0
            )
    
    def _parse_eigenvalues_from_answer(self, answer: str) -> List[complex]:
        """Parse eigenvalues from user's text answer."""
        import re
        
        # Handle SymPy expressions in the answer
        if 'sqrt' in answer:
            try:
                # Try to evaluate SymPy expressions
                from sympy import sympify, N
                # Split by comma and evaluate each part
                parts = answer.split(',')
                eigenvals = []
                for part in parts:
                    part = part.strip()
                    # Remove common prefixes
                    part = re.sub(r'^[λλ₁λ₂=\s]+', '', part)
                    try:
                        # Try to parse as SymPy expression
                        expr = sympify(part)
                        eigenvals.append(complex(N(expr)))
                    except:
                        continue
                if eigenvals:
                    return eigenvals
            except:
                pass
        
        # Look for numbers (including complex)
        number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?(?:\s*[+-]\s*\d*\.?\d*[ij])?'
        matches = re.findall(number_pattern, answer)
        
        eigenvals = []
        for match in matches:
            try:
                # Handle complex numbers
                if 'i' in match or 'j' in match:
                    # Convert i to j for Python complex parsing
                    match = match.replace('i', 'j')
                    eigenvals.append(complex(match))
                else:
                    eigenvals.append(complex(float(match)))
            except:
                continue
        
        return eigenvals
    
    def _compare_eigenvalue_sets(self, set1: List[complex], set2: List[complex], tolerance: float) -> bool:
        """Compare two sets of eigenvalues with tolerance."""
        if len(set1) != len(set2):
            return False
        
        # Sort both sets by magnitude for comparison
        set1_sorted = sorted(set1, key=abs)
        set2_sorted = sorted(set2, key=abs)
        
        for ev1, ev2 in zip(set1_sorted, set2_sorted):
            if abs(ev1 - ev2) > tolerance:
                return False
        
        return True
    
    def _calculate_eigenvalue_partial_credit(self, user_eigenvals: List[complex], correct_eigenvals: List[complex]) -> float:
        """Calculate partial credit for eigenvalue computation."""
        if not user_eigenvals or not correct_eigenvals:
            return 0.0
        
        # Count how many eigenvalues are approximately correct
        correct_count = 0
        tolerance = 1e-6
        
        for user_ev in user_eigenvals:
            for correct_ev in correct_eigenvals:
                if abs(user_ev - correct_ev) < tolerance:
                    correct_count += 1
                    break
        
        return min(1.0, correct_count / len(correct_eigenvals))


class OptimizationResult:
    """Result of optimization analysis."""
    
    def __init__(self,
                 function: sp.Expr,
                 variables: List[sp.Symbol],
                 critical_points: List[Dict[sp.Symbol, float]],
                 gradient: List[sp.Expr],
                 hessian: Matrix,
                 optimization_path: List[Dict[str, float]] = None,
                 convergence_info: Dict[str, Any] = None):
        self.function = function
        self.variables = variables
        self.critical_points = critical_points
        self.gradient = gradient
        self.hessian = hessian
        self.optimization_path = optimization_path or []
        self.convergence_info = convergence_info or {}
    
    def classify_critical_points(self) -> Dict[int, str]:
        """Classify critical points using second derivative test."""
        classifications = {}
        
        for i, point in enumerate(self.critical_points):
            if len(self.variables) == 1:
                # Single variable case
                var = self.variables[0]
                second_deriv = sp.diff(self.function, var, 2)
                second_deriv_value = float(second_deriv.subs(point))
                
                if second_deriv_value > 0:
                    classifications[i] = "local minimum"
                elif second_deriv_value < 0:
                    classifications[i] = "local maximum"
                else:
                    classifications[i] = "inconclusive"
            
            elif len(self.variables) == 2:
                # Two variable case - use Hessian determinant test
                hessian_at_point = self.hessian.subs(point)
                try:
                    det_h = float(hessian_at_point.det())
                    trace_h = float(hessian_at_point.trace())
                    
                    if det_h > 0 and trace_h > 0:
                        classifications[i] = "local minimum"
                    elif det_h > 0 and trace_h < 0:
                        classifications[i] = "local maximum"
                    elif det_h < 0:
                        classifications[i] = "saddle point"
                    else:
                        classifications[i] = "inconclusive"
                except:
                    classifications[i] = "inconclusive"
            else:
                # Higher dimensions - check eigenvalues of Hessian
                try:
                    hessian_at_point = self.hessian.subs(point)
                    eigenvals = hessian_at_point.eigenvals()
                    eigenval_signs = [float(re(ev)) for ev in eigenvals.keys()]
                    
                    if all(ev > 0 for ev in eigenval_signs):
                        classifications[i] = "local minimum"
                    elif all(ev < 0 for ev in eigenval_signs):
                        classifications[i] = "local maximum"
                    elif any(ev > 0 for ev in eigenval_signs) and any(ev < 0 for ev in eigenval_signs):
                        classifications[i] = "saddle point"
                    else:
                        classifications[i] = "inconclusive"
                except:
                    classifications[i] = "inconclusive"
        
        return classifications
    
    def get_ml_optimization_context(self) -> Dict[str, str]:
        """Get machine learning context for optimization results."""
        context = {}
        
        # Gradient descent context
        context["gradient_descent"] = """
        In machine learning, gradients point in the direction of steepest increase.
        Gradient descent moves in the opposite direction (-∇f) to minimize the loss function.
        The magnitude of the gradient indicates how steep the function is at that point.
        """
        
        # Critical points context
        critical_point_types = self.classify_critical_points()
        if any("minimum" in cp_type for cp_type in critical_point_types.values()):
            context["minima"] = """
            Local minima in ML correspond to points where the model has found a good solution.
            Global minimum represents the optimal model parameters.
            Multiple local minima can make optimization challenging.
            """
        
        if any("saddle" in cp_type for cp_type in critical_point_types.values()):
            context["saddle_points"] = """
            Saddle points are common in high-dimensional ML optimization.
            They can slow down gradient descent as gradients become very small.
            Advanced optimizers like Adam help escape saddle points more effectively.
            """
        
        # Hessian context
        context["second_order"] = """
        The Hessian matrix contains second-order information about the loss surface.
        Newton's method uses the Hessian to find better descent directions.
        Condition number of Hessian affects convergence speed of optimization algorithms.
        """
        
        return context


class GradientVisualization:
    """Gradient and optimization visualization tools."""
    
    def __init__(self):
        """Initialize gradient visualization tools."""
        pass
    
    def create_gradient_field_2d(self, 
                                function: sp.Expr, 
                                variables: List[sp.Symbol],
                                x_range: Tuple[float, float] = (-3, 3),
                                y_range: Tuple[float, float] = (-3, 3),
                                grid_density: int = 20) -> Dict[str, Any]:
        """
        Create 2D gradient field visualization data.
        
        Args:
            function: SymPy expression for the function
            variables: List of variables [x, y]
            x_range: Range for x-axis
            y_range: Range for y-axis
            grid_density: Number of grid points per axis
            
        Returns:
            Dictionary containing gradient field data
        """
        if len(variables) != 2:
            raise ValueError("Gradient field visualization requires exactly 2 variables")
        
        x_var, y_var = variables
        
        # Compute gradient
        grad_x = sp.diff(function, x_var)
        grad_y = sp.diff(function, y_var)
        
        # Create grid
        x_vals = np.linspace(x_range[0], x_range[1], grid_density)
        y_vals = np.linspace(y_range[0], y_range[1], grid_density)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluate gradient at grid points
        grad_x_func = sp.lambdify([x_var, y_var], grad_x, 'numpy')
        grad_y_func = sp.lambdify([x_var, y_var], grad_y, 'numpy')
        
        try:
            U = grad_x_func(X, Y)
            V = grad_y_func(X, Y)
        except:
            # Handle cases where function evaluation fails
            U = np.zeros_like(X)
            V = np.zeros_like(Y)
        
        # Evaluate function values for contour plot
        func_lambdified = sp.lambdify([x_var, y_var], function, 'numpy')
        try:
            Z = func_lambdified(X, Y)
        except:
            Z = np.zeros_like(X)
        
        return {
            "x_grid": X.tolist(),
            "y_grid": Y.tolist(),
            "gradient_x": U.tolist(),
            "gradient_y": V.tolist(),
            "function_values": Z.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "function_expr": str(function),
            "gradient_expr": [str(grad_x), str(grad_y)]
        }
    
    def create_loss_surface_3d(self,
                              function: sp.Expr,
                              variables: List[sp.Symbol],
                              x_range: Tuple[float, float] = (-3, 3),
                              y_range: Tuple[float, float] = (-3, 3),
                              grid_density: int = 50) -> Dict[str, Any]:
        """
        Create 3D loss surface visualization data.
        
        Args:
            function: SymPy expression for the function
            variables: List of variables [x, y]
            x_range: Range for x-axis
            y_range: Range for y-axis
            grid_density: Number of grid points per axis
            
        Returns:
            Dictionary containing 3D surface data
        """
        if len(variables) != 2:
            raise ValueError("3D surface visualization requires exactly 2 variables")
        
        x_var, y_var = variables
        
        # Create grid
        x_vals = np.linspace(x_range[0], x_range[1], grid_density)
        y_vals = np.linspace(y_range[0], y_range[1], grid_density)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Evaluate function
        func_lambdified = sp.lambdify([x_var, y_var], function, 'numpy')
        try:
            Z = func_lambdified(X, Y)
        except:
            Z = np.zeros_like(X)
        
        return {
            "x_surface": X.tolist(),
            "y_surface": Y.tolist(),
            "z_surface": Z.tolist(),
            "x_range": x_range,
            "y_range": y_range,
            "function_expr": str(function),
            "title": f"Loss Surface: {function}"
        }
    
    def simulate_gradient_descent(self,
                                 function: sp.Expr,
                                 variables: List[sp.Symbol],
                                 start_point: Dict[sp.Symbol, float],
                                 learning_rate: float = 0.1,
                                 max_iterations: int = 100,
                                 tolerance: float = 1e-6) -> Dict[str, Any]:
        """
        Simulate gradient descent optimization.
        
        Args:
            function: Function to optimize
            variables: Variables to optimize over
            start_point: Starting point for optimization
            learning_rate: Step size for gradient descent
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary containing optimization path and convergence info
        """
        # Compute gradient
        gradient = [sp.diff(function, var) for var in variables]
        
        # Convert to numerical functions
        func_lambdified = sp.lambdify(variables, function, 'numpy')
        grad_lambdified = [sp.lambdify(variables, g, 'numpy') for g in gradient]
        
        # Initialize
        current_point = {var: start_point[var] for var in variables}
        path = [current_point.copy()]
        function_values = [float(func_lambdified(*[current_point[var] for var in variables]))]
        
        for iteration in range(max_iterations):
            # Evaluate gradient at current point
            current_vals = [current_point[var] for var in variables]
            try:
                grad_vals = [float(g(*current_vals)) for g in grad_lambdified]
            except:
                break
            
            # Check convergence
            grad_norm = np.sqrt(sum(g**2 for g in grad_vals))
            if grad_norm < tolerance:
                break
            
            # Update point
            for i, var in enumerate(variables):
                current_point[var] -= learning_rate * grad_vals[i]
            
            path.append(current_point.copy())
            try:
                func_val = float(func_lambdified(*[current_point[var] for var in variables]))
                function_values.append(func_val)
            except:
                break
        
        # Convert path to list format for JSON serialization
        path_list = []
        for point in path:
            path_list.append({str(var): val for var, val in point.items()})
        
        return {
            "optimization_path": path_list,
            "function_values": function_values,
            "iterations": len(path) - 1,
            "converged": grad_norm < tolerance if 'grad_norm' in locals() else False,
            "final_gradient_norm": grad_norm if 'grad_norm' in locals() else None,
            "learning_rate": learning_rate,
            "algorithm": "gradient_descent"
        }


# Add optimization methods to the main AIMLMathematics class
class AIMLMathematics:
    """Specialized AI/ML mathematics tools."""
    
    def __init__(self):
        """Initialize AI/ML mathematics module."""
        self.gradient_viz = GradientVisualization()
    
    def analyze_optimization_step_by_step(self, 
                                        function_text: str, 
                                        variables: List[str]) -> Tuple[List[SolutionStep], OptimizationResult]:
        """
        Perform step-by-step optimization analysis.
        
        Args:
            function_text: String representation of function to optimize
            variables: List of variable names
            
        Returns:
            Tuple of solution steps and optimization result
        """
        steps = []
        step_num = 1
        
        # Parse function and variables
        var_symbols = [sp.Symbol(var) for var in variables]
        try:
            function = sp.sympify(function_text)
        except:
            raise ComputationError(f"Could not parse function: {function_text}")
        
        # Step 1: Show the function
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Start with the given function",
            explanation="We need to find critical points and analyze the optimization landscape",
            mathematical_expression=f"f({', '.join(variables)}) = {function}",
            intermediate_result=f"Function to optimize: {function}"
        ))
        step_num += 1
        
        # Step 2: Compute gradient
        gradient = [sp.diff(function, var) for var in var_symbols]
        gradient_str = f"∇f = [{', '.join(str(g) for g in gradient)}]"
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Compute the gradient",
            explanation="The gradient points in the direction of steepest increase",
            mathematical_expression=gradient_str,
            intermediate_result=f"Gradient: {gradient_str}"
        ))
        step_num += 1
        
        # Step 3: Find critical points
        critical_points = []
        try:
            critical_point_solutions = sp.solve(gradient, var_symbols)
            
            if isinstance(critical_point_solutions, dict):
                critical_points = [critical_point_solutions]
            elif isinstance(critical_point_solutions, list):
                critical_points = critical_point_solutions
            
            critical_points_str = []
            for i, point in enumerate(critical_points):
                if isinstance(point, dict):
                    point_str = f"({', '.join(f'{var}={val}' for var, val in point.items())})"
                else:
                    point_str = str(point)
                critical_points_str.append(point_str)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Find critical points",
                explanation="Critical points occur where the gradient equals zero",
                mathematical_expression="∇f = 0",
                intermediate_result=f"Critical points: {', '.join(critical_points_str) if critical_points_str else 'None found'}"
            ))
            step_num += 1
            
        except Exception as e:
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Find critical points",
                explanation="Attempting to solve ∇f = 0 for critical points",
                mathematical_expression="∇f = 0",
                intermediate_result=f"Could not solve analytically: {str(e)}"
            ))
            step_num += 1
        
        # Step 4: Compute Hessian matrix
        hessian_matrix = Matrix([[sp.diff(function, var1, var2) for var2 in var_symbols] 
                                for var1 in var_symbols])
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Compute Hessian matrix",
            explanation="The Hessian contains second-order derivative information",
            mathematical_expression=f"H = {hessian_matrix}",
            intermediate_result=f"Hessian: {hessian_matrix}"
        ))
        step_num += 1
        
        # Create optimization result
        optimization_result = OptimizationResult(
            function=function,
            variables=var_symbols,
            critical_points=critical_points,
            gradient=gradient,
            hessian=hessian_matrix
        )
        
        # Step 5: Classify critical points
        if critical_points:
            classifications = optimization_result.classify_critical_points()
            classification_text = []
            for i, (point, classification) in enumerate(zip(critical_points, classifications.values())):
                if isinstance(point, dict):
                    point_str = f"({', '.join(f'{var}={val}' for var, val in point.items())})"
                else:
                    point_str = str(point)
                classification_text.append(f"{point_str}: {classification}")
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation="Classify critical points",
                explanation="Use second derivative test to classify critical points",
                mathematical_expression="Second derivative test",
                intermediate_result="; ".join(classification_text)
            ))
            step_num += 1
        
        # Step 6: ML context
        ml_context = optimization_result.get_ml_optimization_context()
        context_summary = []
        for key, value in ml_context.items():
            context_summary.append(f"{key.replace('_', ' ').title()}: {value.split('.')[0]}.")
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Machine Learning Optimization Context",
            explanation="Understanding optimization in the context of machine learning",
            mathematical_expression="Optimization ↔ ML Applications",
            intermediate_result="; ".join(context_summary)
        ))
        
        return steps, optimization_result
    
    def solve_optimization_problem(self, problem: ParsedProblem) -> StepSolution:
        """
        Solve optimization problems with ML context.
        
        Args:
            problem: Parsed optimization problem
            
        Returns:
            Complete step-by-step solution
        """
        try:
            # Extract function and variables from problem
            function_text, variables = self._extract_optimization_from_problem(problem)
            
            # Perform step-by-step analysis
            steps, optimization_result = self.analyze_optimization_step_by_step(function_text, variables)
            
            # Format final answer
            critical_points_str = []
            if optimization_result.critical_points:
                classifications = optimization_result.classify_critical_points()
                for i, (point, classification) in enumerate(zip(optimization_result.critical_points, classifications.values())):
                    if isinstance(point, dict):
                        point_str = f"({', '.join(f'{var}={val}' for var, val in point.items())})"
                    else:
                        point_str = str(point)
                    critical_points_str.append(f"{point_str} ({classification})")
            
            final_answer = f"Critical points: {'; '.join(critical_points_str) if critical_points_str else 'None found'}"
            
            return StepSolution(
                problem_id=problem.id,
                steps=steps,
                final_answer=final_answer,
                solution_method="Optimization analysis with ML context",
                confidence_score=0.9,
                computation_time=0.0
            )
            
        except Exception as e:
            raise ComputationError(f"Failed to solve optimization problem: {str(e)}")
    
    def _extract_optimization_from_problem(self, problem: ParsedProblem) -> Tuple[str, List[str]]:
        """Extract function and variables from optimization problem."""
        # Try to find function in expressions first
        function_text = None
        variables = problem.variables.copy() if problem.variables else []
        
        # Check expressions first (they usually contain the clean mathematical expression)
        for expr_text in problem.expressions:
            if expr_text.strip():
                function_text = expr_text.strip()
                break
        
        # If no function found in expressions, try original text
        if not function_text:
            import re
            # Look for common optimization patterns
            patterns = [
                r'minimize\s+f\([^)]+\)\s*=\s*(.+?)(?:\s+subject|$)',
                r'maximize\s+f\([^)]+\)\s*=\s*(.+?)(?:\s+subject|$)',
                r'optimize\s+f\([^)]+\)\s*=\s*(.+?)(?:\s+subject|$)',
                r'f\([^)]+\)\s*=\s*(.+?)(?:\s+subject|$)',
                r'minimize\s+(.+?)(?:\s+subject|$)',
                r'maximize\s+(.+?)(?:\s+subject|$)',
                r'optimize\s+(.+?)(?:\s+subject|$)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, problem.original_text, re.IGNORECASE)
                if match:
                    function_text = match.group(1).strip()
                    # Clean up common function notation
                    function_text = re.sub(r'^f\([^)]+\)\s*=\s*', '', function_text)
                    break
        
        # Extract variables if not provided
        if not variables and function_text:
            import re
            var_matches = re.findall(r'\b([x-z])\b', function_text)
            variables = list(set(var_matches))
        
        # Default values if extraction fails
        if not function_text:
            function_text = "x**2 + y**2"  # Default quadratic function
        if not variables:
            variables = ['x', 'y']
        
        return function_text, variables