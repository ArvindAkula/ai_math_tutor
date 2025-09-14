"""
Tests for optimization and gradient visualization tools with AI/ML context.
Tests optimization accuracy and visualization performance.
"""

import unittest
import numpy as np
from sympy import symbols, Matrix, sqrt, sin, cos, exp
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from aiml_mathematics import AIMLMathematics, OptimizationResult, GradientVisualization
from models import ParsedProblem, MathDomain, DifficultyLevel


class TestOptimizationAnalysis(unittest.TestCase):
    """Test optimization analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aiml_math = AIMLMathematics()
        self.grad_viz = GradientVisualization()
    
    def test_simple_quadratic_optimization(self):
        """Test optimization of a simple quadratic function."""
        # Test function: f(x) = x^2 + 2x + 1 = (x+1)^2
        function_text = "x**2 + 2*x + 1"
        variables = ["x"]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function_text, variables)
        
        # Check that we have the expected number of steps
        self.assertGreaterEqual(len(steps), 5)
        
        # Check that critical point is found correctly
        # For f(x) = x^2 + 2x + 1, critical point should be at x = -1
        self.assertEqual(len(result.critical_points), 1)
        critical_point = result.critical_points[0]
        x_sym = symbols('x')
        self.assertAlmostEqual(float(critical_point[x_sym]), -1.0, places=6)
        
        # Check classification
        classifications = result.classify_critical_points()
        self.assertEqual(classifications[0], "local minimum")
        
        # Check ML context
        ml_context = result.get_ml_optimization_context()
        self.assertIn('gradient_descent', ml_context)
        self.assertIn('minima', ml_context)
    
    def test_two_variable_optimization(self):
        """Test optimization of a two-variable function."""
        # Test function: f(x,y) = x^2 + y^2 (simple paraboloid)
        function_text = "x**2 + y**2"
        variables = ["x", "y"]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function_text, variables)
        
        # Check critical point at origin
        self.assertEqual(len(result.critical_points), 1)
        critical_point = result.critical_points[0]
        x_sym, y_sym = symbols('x y')
        self.assertAlmostEqual(float(critical_point[x_sym]), 0.0, places=6)
        self.assertAlmostEqual(float(critical_point[y_sym]), 0.0, places=6)
        
        # Check classification
        classifications = result.classify_critical_points()
        self.assertEqual(classifications[0], "local minimum")
        
        # Check Hessian
        expected_hessian = Matrix([[2, 0], [0, 2]])
        self.assertEqual(result.hessian, expected_hessian)
    
    def test_saddle_point_optimization(self):
        """Test optimization with saddle point."""
        # Test function: f(x,y) = x^2 - y^2 (hyperbolic paraboloid)
        function_text = "x**2 - y**2"
        variables = ["x", "y"]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function_text, variables)
        
        # Check critical point at origin
        self.assertEqual(len(result.critical_points), 1)
        critical_point = result.critical_points[0]
        x_sym, y_sym = symbols('x y')
        self.assertAlmostEqual(float(critical_point[x_sym]), 0.0, places=6)
        self.assertAlmostEqual(float(critical_point[y_sym]), 0.0, places=6)
        
        # Check classification as saddle point
        classifications = result.classify_critical_points()
        self.assertEqual(classifications[0], "saddle point")
        
        # Check ML context includes saddle point information
        ml_context = result.get_ml_optimization_context()
        self.assertIn('saddle_points', ml_context)
    
    def test_multiple_critical_points(self):
        """Test function with multiple critical points."""
        # Test function: f(x) = x^3 - 3x (has local max and min)
        function_text = "x**3 - 3*x"
        variables = ["x"]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function_text, variables)
        
        # Should have 2 critical points at x = ±1
        self.assertEqual(len(result.critical_points), 2)
        
        # Sort critical points by x value
        x_sym = symbols('x')
        critical_x_values = sorted([float(cp[x_sym]) for cp in result.critical_points])
        
        self.assertAlmostEqual(critical_x_values[0], -1.0, places=6)
        self.assertAlmostEqual(critical_x_values[1], 1.0, places=6)
        
        # Check classifications
        classifications = result.classify_critical_points()
        # x = -1 should be local maximum, x = 1 should be local minimum
        classification_values = list(classifications.values())
        self.assertIn("local maximum", classification_values)
        self.assertIn("local minimum", classification_values)
    
    def test_gradient_field_2d_visualization(self):
        """Test 2D gradient field visualization."""
        x, y = symbols('x y')
        function = x**2 + y**2
        variables = [x, y]
        
        viz_data = self.grad_viz.create_gradient_field_2d(function, variables)
        
        # Check data structure
        required_keys = ['x_grid', 'y_grid', 'gradient_x', 'gradient_y', 'function_values', 
                        'x_range', 'y_range', 'function_expr', 'gradient_expr']
        for key in required_keys:
            self.assertIn(key, viz_data)
        
        # Check that gradients are computed correctly
        # For f(x,y) = x^2 + y^2, gradient is [2x, 2y]
        self.assertEqual(viz_data['gradient_expr'], ['2*x', '2*y'])
        
        # Check grid dimensions
        self.assertEqual(len(viz_data['x_grid']), 20)  # Default grid density
        self.assertEqual(len(viz_data['y_grid']), 20)
        
        # Check that gradient at origin is [0, 0]
        x_grid = np.array(viz_data['x_grid'])
        y_grid = np.array(viz_data['y_grid'])
        grad_x = np.array(viz_data['gradient_x'])
        grad_y = np.array(viz_data['gradient_y'])
        
        # Find index closest to origin
        center_idx = len(x_grid) // 2
        self.assertAlmostEqual(grad_x[center_idx, center_idx], 0.0, places=1)
        self.assertAlmostEqual(grad_y[center_idx, center_idx], 0.0, places=1)
    
    def test_loss_surface_3d_visualization(self):
        """Test 3D loss surface visualization."""
        x, y = symbols('x y')
        function = x**2 + y**2
        variables = [x, y]
        
        viz_data = self.grad_viz.create_loss_surface_3d(function, variables)
        
        # Check data structure
        required_keys = ['x_surface', 'y_surface', 'z_surface', 'x_range', 'y_range', 
                        'function_expr', 'title']
        for key in required_keys:
            self.assertIn(key, viz_data)
        
        # Check grid dimensions (default 50x50)
        self.assertEqual(len(viz_data['x_surface']), 50)
        self.assertEqual(len(viz_data['y_surface']), 50)
        self.assertEqual(len(viz_data['z_surface']), 50)
        
        # Check that minimum is at origin
        z_surface = np.array(viz_data['z_surface'])
        min_idx = np.unravel_index(np.argmin(z_surface), z_surface.shape)
        
        # Should be approximately at the center of the grid
        center_idx = len(z_surface) // 2
        self.assertAlmostEqual(min_idx[0], center_idx, delta=2)
        self.assertAlmostEqual(min_idx[1], center_idx, delta=2)
    
    def test_gradient_descent_simulation(self):
        """Test gradient descent simulation."""
        x, y = symbols('x y')
        function = x**2 + y**2
        variables = [x, y]
        start_point = {x: 2.0, y: 2.0}
        
        result = self.grad_viz.simulate_gradient_descent(
            function, variables, start_point, 
            learning_rate=0.1, max_iterations=50
        )
        
        # Check result structure
        required_keys = ['optimization_path', 'function_values', 'iterations', 
                        'converged', 'final_gradient_norm', 'learning_rate', 'algorithm']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check that algorithm converged
        self.assertTrue(result['converged'])
        
        # Check that function values decrease
        func_vals = result['function_values']
        self.assertGreater(func_vals[0], func_vals[-1])  # Should decrease
        
        # Check that final point is close to origin
        final_point = result['optimization_path'][-1]
        self.assertAlmostEqual(float(final_point['x']), 0.0, places=1)
        self.assertAlmostEqual(float(final_point['y']), 0.0, places=1)
        
        # Check that path length is reasonable
        self.assertGreater(len(result['optimization_path']), 5)
        self.assertLess(len(result['optimization_path']), 50)
    
    def test_optimization_problem_integration(self):
        """Test integration with problem solving framework."""
        # Create a mock optimization problem
        problem = ParsedProblem(
            id="test_opt_1",
            original_text="Minimize f(x,y) = x^2 + y^2",
            domain=MathDomain.AI_ML_MATH,
            problem_type="optimization",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=["x", "y"],
            expressions=["x**2 + y**2"],
            metadata={}
        )
        
        solution = self.aiml_math.solve_optimization_problem(problem)
        
        # Check solution structure
        self.assertIsNotNone(solution.steps)
        self.assertGreater(len(solution.steps), 5)
        self.assertIn("optimization", solution.solution_method.lower())
        self.assertIn("ml", solution.solution_method.lower())
        
        # Check that ML context is included
        ml_step_found = False
        for step in solution.steps:
            if "ml" in step.operation.lower() or "machine learning" in step.operation.lower():
                ml_step_found = True
                break
        self.assertTrue(ml_step_found)
        
        # Check final answer format
        self.assertIn("critical", solution.final_answer.lower())
    
    def test_optimization_extraction_from_problem(self):
        """Test extraction of optimization function from different problem formats."""
        # Test minimize format
        problem1 = ParsedProblem(
            id="test_extract_1",
            original_text="Minimize f(x,y) = x^2 + y^2",
            domain=MathDomain.AI_ML_MATH,
            problem_type="optimization",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=["x", "y"],
            expressions=["x**2 + y**2"],
            metadata={}
        )
        
        function_text, variables = self.aiml_math._extract_optimization_from_problem(problem1)
        self.assertEqual(function_text, "x**2 + y**2")
        self.assertEqual(set(variables), {"x", "y"})
        
        # Test maximize format
        problem2 = ParsedProblem(
            id="test_extract_2",
            original_text="Maximize g(x) = -x^2 + 4x",
            domain=MathDomain.AI_ML_MATH,
            problem_type="optimization",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=["x"],
            expressions=["-x**2 + 4*x"],
            metadata={}
        )
        
        function_text2, variables2 = self.aiml_math._extract_optimization_from_problem(problem2)
        self.assertEqual(function_text2, "-x**2 + 4*x")
        self.assertEqual(variables2, ["x"])
    
    def test_gradient_field_error_handling(self):
        """Test error handling in gradient field visualization."""
        x = symbols('x')
        function = x**2  # Single variable function
        variables = [x]
        
        # Should raise error for single variable
        with self.assertRaises(ValueError):
            self.grad_viz.create_gradient_field_2d(function, variables)
    
    def test_loss_surface_error_handling(self):
        """Test error handling in loss surface visualization."""
        x = symbols('x')
        function = x**2  # Single variable function
        variables = [x]
        
        # Should raise error for single variable
        with self.assertRaises(ValueError):
            self.grad_viz.create_loss_surface_3d(function, variables)
    
    def test_complex_function_optimization(self):
        """Test optimization of more complex functions."""
        # Test Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
        function_text = "(1-x)**2 + 100*(y-x**2)**2"
        variables = ["x", "y"]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function_text, variables)
        
        # Rosenbrock function has global minimum at (1, 1)
        self.assertEqual(len(result.critical_points), 1)
        critical_point = result.critical_points[0]
        x_sym, y_sym = symbols('x y')
        
        # Check that critical point is at (1, 1)
        self.assertAlmostEqual(float(critical_point[x_sym]), 1.0, places=6)
        self.assertAlmostEqual(float(critical_point[y_sym]), 1.0, places=6)
        
        # Should be classified as minimum
        classifications = result.classify_critical_points()
        self.assertEqual(classifications[0], "local minimum")
    
    def test_gradient_descent_convergence_properties(self):
        """Test gradient descent convergence properties."""
        x, y = symbols('x y')
        
        # Test with different learning rates
        function = x**2 + y**2
        variables = [x, y]
        start_point = {x: 1.0, y: 1.0}
        
        # Small learning rate - should converge slowly
        result_slow = self.grad_viz.simulate_gradient_descent(
            function, variables, start_point, learning_rate=0.01, max_iterations=100
        )
        
        # Large learning rate - should converge faster
        result_fast = self.grad_viz.simulate_gradient_descent(
            function, variables, start_point, learning_rate=0.5, max_iterations=100
        )
        
        # Both should converge
        self.assertTrue(result_slow['converged'])
        self.assertTrue(result_fast['converged'])
        
        # Fast should take fewer iterations
        self.assertLess(result_fast['iterations'], result_slow['iterations'])
    
    def test_ml_optimization_context_completeness(self):
        """Test that ML optimization context covers all important concepts."""
        # Create a result with various critical point types
        x, y = symbols('x y')
        function = x**2 - y**2  # Has saddle point
        variables = [x, y]
        
        steps, result = self.aiml_math.analyze_optimization_step_by_step(function, variables)
        ml_context = result.get_ml_optimization_context()
        
        # Check that all important ML concepts are covered
        expected_concepts = ['gradient_descent', 'second_order']
        for concept in expected_concepts:
            self.assertIn(concept, ml_context)
            self.assertGreater(len(ml_context[concept].strip()), 50)  # Non-trivial explanation
        
        # Should include saddle point context for this function
        self.assertIn('saddle_points', ml_context)


class TestOptimizationResult(unittest.TestCase):
    """Test OptimizationResult class functionality."""
    
    def test_critical_point_classification_1d(self):
        """Test critical point classification for 1D functions."""
        x = symbols('x')
        function = x**2  # Has minimum at x=0
        variables = [x]
        critical_points = [{x: 0}]
        gradient = [2*x]
        hessian = Matrix([[2]])
        
        result = OptimizationResult(
            function=function,
            variables=variables,
            critical_points=critical_points,
            gradient=gradient,
            hessian=hessian
        )
        
        classifications = result.classify_critical_points()
        self.assertEqual(classifications[0], "local minimum")
    
    def test_critical_point_classification_2d(self):
        """Test critical point classification for 2D functions."""
        x, y = symbols('x y')
        
        # Test minimum
        function_min = x**2 + y**2
        variables = [x, y]
        critical_points = [{x: 0, y: 0}]
        gradient = [2*x, 2*y]
        hessian = Matrix([[2, 0], [0, 2]])
        
        result_min = OptimizationResult(
            function=function_min,
            variables=variables,
            critical_points=critical_points,
            gradient=gradient,
            hessian=hessian
        )
        
        classifications_min = result_min.classify_critical_points()
        self.assertEqual(classifications_min[0], "local minimum")
        
        # Test saddle point
        function_saddle = x**2 - y**2
        hessian_saddle = Matrix([[2, 0], [0, -2]])
        
        result_saddle = OptimizationResult(
            function=function_saddle,
            variables=variables,
            critical_points=critical_points,
            gradient=[2*x, -2*y],
            hessian=hessian_saddle
        )
        
        classifications_saddle = result_saddle.classify_critical_points()
        self.assertEqual(classifications_saddle[0], "saddle point")
    
    def test_ml_context_adaptation(self):
        """Test that ML context adapts to different optimization scenarios."""
        x, y = symbols('x y')
        
        # Function with minimum
        function = x**2 + y**2
        variables = [x, y]
        critical_points = [{x: 0, y: 0}]
        gradient = [2*x, 2*y]
        hessian = Matrix([[2, 0], [0, 2]])
        
        result = OptimizationResult(
            function=function,
            variables=variables,
            critical_points=critical_points,
            gradient=gradient,
            hessian=hessian
        )
        
        ml_context = result.get_ml_optimization_context()
        
        # Should include minima context
        self.assertIn('minima', ml_context)
        self.assertIn('global minimum', ml_context['minima'])
        
        # Should not include saddle point context for this function
        self.assertNotIn('saddle_points', ml_context)


if __name__ == '__main__':
    unittest.main()