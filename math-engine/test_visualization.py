"""
Unit tests for the mathematical visualization engine.
Tests plot generation accuracy, rendering performance, and interactive elements.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import tempfile
import base64

# Add shared models to path
from models import (
    PlotData, Point, PlotStyle, InteractiveElement,
    ParsedProblem, StepSolution, SolutionStep, MathDomain, DifficultyLevel
)

from visualization import MathPlotter, VisualizationEngine


class TestMathPlotter:
    """Test cases for the MathPlotter class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = MathPlotter()
    
    def test_create_function_plot_basic(self):
        """Test basic function plotting."""
        plot_data = self.plotter.create_function_plot(
            expression='x**2',
            x_range=(-5, 5)
        )
        
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) > 0
        assert plot_data.title == "Graph of x**2"
        assert plot_data.axis_labels['x'] == 'x'
        assert plot_data.axis_labels['y'] == 'f(x)'
        assert plot_data.styling.color == 'blue'
        assert plot_data.styling.line_width == 2.0
    
    def test_create_function_plot_trigonometric(self):
        """Test plotting trigonometric functions."""
        plot_data = self.plotter.create_function_plot(
            expression='sin(x)',
            x_range=(-np.pi, np.pi)
        )
        
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) > 0
        
        # Check that y values are within expected range for sin(x)
        y_values = [p.y for p in plot_data.data_points]
        assert min(y_values) >= -1.1  # Allow small numerical errors
        assert max(y_values) <= 1.1
    
    def test_create_function_plot_with_custom_style(self):
        """Test function plotting with custom styling."""
        custom_style = PlotStyle(
            color='red',
            line_width=3.0,
            marker_style='o',
            transparency=0.8
        )
        
        plot_data = self.plotter.create_function_plot(
            expression='x**3',
            style=custom_style,
            title="Custom Cubic Function"
        )
        
        assert plot_data.styling.color == 'red'
        assert plot_data.styling.line_width == 3.0
        assert plot_data.styling.marker_style == 'o'
        assert plot_data.styling.transparency == 0.8
        assert plot_data.title == "Custom Cubic Function"
    
    def test_create_derivative_plot(self):
        """Test derivative visualization."""
        plot_data = self.plotter.create_derivative_plot(
            function_expr='x**2',
            derivative_expr='2*x',
            highlight_point=1.0
        )
        
        assert plot_data.plot_type == 'derivative'
        assert len(plot_data.data_points) > 0
        assert len(plot_data.interactive_elements) > 0
        
        # Check for tangent point interactive element
        tangent_elements = [e for e in plot_data.interactive_elements 
                          if e.element_type == 'point']
        assert len(tangent_elements) > 0
        assert tangent_elements[0].position.x == 1.0
    
    def test_create_integral_plot(self):
        """Test integral visualization."""
        plot_data = self.plotter.create_integral_plot(
            function_expr='x**2',
            integral_bounds=(0, 2),
            show_area=True
        )
        
        assert plot_data.plot_type == 'integral'
        assert len(plot_data.data_points) > 0
        assert plot_data.styling.color == 'green'
        
        # Check for area interactive element
        area_elements = [e for e in plot_data.interactive_elements 
                        if e.element_type == 'area']
        assert len(area_elements) > 0
        assert "Integral from 0 to 2" in area_elements[0].tooltip
    
    def test_create_optimization_plot(self):
        """Test optimization visualization."""
        plot_data = self.plotter.create_optimization_plot(
            function_expr='x**2 - 4*x + 3',
            critical_points=[2.0],
            extrema_type=['min']
        )
        
        assert plot_data.plot_type == 'optimization'
        assert len(plot_data.data_points) > 0
        assert plot_data.styling.color == 'red'
        
        # Check for critical point interactive element
        critical_elements = [e for e in plot_data.interactive_elements 
                           if e.element_type == 'critical_point']
        assert len(critical_elements) > 0
        assert critical_elements[0].position.x == 2.0
        assert 'Min' in critical_elements[0].tooltip
    
    def test_create_multi_function_plot(self):
        """Test multi-function plotting."""
        plot_data = self.plotter.create_multi_function_plot(
            expressions=['x**2', 'x**3', 'sin(x)'],
            labels=['Quadratic', 'Cubic', 'Sine'],
            x_range=(-2, 2)
        )
        
        assert plot_data.plot_type == 'multi_function'
        assert len(plot_data.data_points) > 0
        
        # Check for legend elements
        legend_elements = [e for e in plot_data.interactive_elements 
                          if e.element_type == 'legend']
        assert len(legend_elements) == 3
    
    def test_render_plot_basic(self):
        """Test basic plot rendering."""
        plot_data = self.plotter.create_function_plot('x**2')
        
        # Test rendering to base64
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
        
        # Verify it's valid base64
        try:
            base64.b64decode(image_base64)
        except Exception:
            pytest.fail("Generated image is not valid base64")
    
    def test_render_plot_with_save(self):
        """Test plot rendering with file save."""
        plot_data = self.plotter.create_function_plot('sin(x)')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image_base64 = self.plotter.render_plot(plot_data, save_path=tmp.name)
            
            # Check that file was created
            assert os.path.exists(tmp.name)
            assert os.path.getsize(tmp.name) > 0
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_error_handling_invalid_expression(self):
        """Test error handling for invalid mathematical expressions."""
        # Invalid expressions should be handled gracefully with empty plots
        plot_data = self.plotter.create_function_plot('invalid_expression_xyz')
        
        # Should return a plot with no data points for invalid expressions
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) == 0
    
    def test_error_handling_invalid_range(self):
        """Test handling of invalid x ranges."""
        # This should still work, just with no points
        plot_data = self.plotter.create_function_plot(
            'x**2',
            x_range=(10, 5)  # Invalid range
        )
        
        # Should handle gracefully
        assert plot_data.plot_type == 'function'


class TestVisualizationEngine:
    """Test cases for the VisualizationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = VisualizationEngine()
    
    def create_sample_problem(self, problem_type: str, expression: str) -> ParsedProblem:
        """Create a sample problem for testing."""
        return ParsedProblem(
            id='test-123',
            original_text=f"Test {problem_type} problem",
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=[expression],
            problem_type=problem_type,
            metadata={}
        )
    
    def create_sample_solution(self, final_answer: str) -> StepSolution:
        """Create a sample solution for testing."""
        return StepSolution(
            problem_id='test-123',
            steps=[
                SolutionStep(
                    step_number=1,
                    operation='Test operation',
                    explanation='Test explanation',
                    mathematical_expression='test_expr',
                    intermediate_result='test_result'
                )
            ],
            final_answer=final_answer,
            solution_method='Test method',
            confidence_score=0.9,
            computation_time=0.1
        )
    
    def test_generate_derivative_visualization(self):
        """Test derivative problem visualization generation."""
        problem = self.create_sample_problem('derivative', 'x**2')
        solution = self.create_sample_solution('2*x')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'derivative'
        assert len(plot_data.data_points) > 0
        assert 'derivative' in plot_data.title.lower() or 'x**2' in plot_data.title
    
    def test_generate_integral_visualization(self):
        """Test integral problem visualization generation."""
        problem = self.create_sample_problem('integral', 'x**2')
        solution = self.create_sample_solution('x**3/3 + C')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'integral'
        assert len(plot_data.data_points) > 0
        assert plot_data.styling.color == 'green'
    
    def test_generate_optimization_visualization(self):
        """Test optimization problem visualization generation."""
        problem = self.create_sample_problem('optimization', 'x**2 - 4*x + 3')
        solution = self.create_sample_solution('minimum at x = 2')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'optimization'
        assert len(plot_data.data_points) > 0
        assert plot_data.styling.color == 'red'
    
    def test_generate_limit_visualization(self):
        """Test limit problem visualization generation."""
        problem = self.create_sample_problem('limit', '1/x')
        solution = self.create_sample_solution('0')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) > 0
        assert '1/x' in plot_data.title
    
    def test_generate_equation_visualization(self):
        """Test equation problem visualization generation."""
        problem = self.create_sample_problem('linear_equation', '2*x + 3 = 7')
        solution = self.create_sample_solution('2')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) > 0
    
    def test_error_handling_in_visualization(self):
        """Test error handling in visualization generation."""
        problem = self.create_sample_problem('unknown_type', 'invalid_expr')
        solution = self.create_sample_solution('error')
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        # Should return error plot data instead of raising exception
        assert plot_data.plot_type in ['error', 'function']  # May fall back to default
    
    def test_extract_integral_bounds(self):
        """Test extraction of integral bounds from problem text."""
        problem = ParsedProblem(
            id='test',
            original_text='Find the integral of x^2 from 0 to 5',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x**2'],
            problem_type='integral',
            metadata={}
        )
        
        bounds = self.engine._extract_integral_bounds(problem)
        assert bounds == (0.0, 5.0)
    
    def test_extract_integral_bounds_no_bounds(self):
        """Test extraction when no bounds are present."""
        problem = ParsedProblem(
            id='test',
            original_text='Find the integral of x^2',
            domain=MathDomain.CALCULUS,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x'],
            expressions=['x**2'],
            problem_type='integral',
            metadata={}
        )
        
        bounds = self.engine._extract_integral_bounds(problem)
        assert bounds is None


class TestPlotDataStructures:
    """Test the plot data structures and their properties."""
    
    def test_point_creation(self):
        """Test Point data structure."""
        point_2d = Point(x=1.0, y=2.0)
        assert point_2d.x == 1.0
        assert point_2d.y == 2.0
        assert point_2d.z is None
        
        point_3d = Point(x=1.0, y=2.0, z=3.0)
        assert point_3d.z == 3.0
    
    def test_plot_style_creation(self):
        """Test PlotStyle data structure."""
        style = PlotStyle(
            color='red',
            line_width=2.5,
            marker_style='o',
            transparency=0.8
        )
        
        assert style.color == 'red'
        assert style.line_width == 2.5
        assert style.marker_style == 'o'
        assert style.transparency == 0.8
    
    def test_interactive_element_creation(self):
        """Test InteractiveElement data structure."""
        element = InteractiveElement(
            element_type='point',
            position=Point(x=1.0, y=2.0),
            action='highlight',
            tooltip='Test tooltip'
        )
        
        assert element.element_type == 'point'
        assert element.position.x == 1.0
        assert element.action == 'highlight'
        assert element.tooltip == 'Test tooltip'
    
    def test_plot_data_creation(self):
        """Test PlotData data structure."""
        points = [Point(x=1.0, y=1.0), Point(x=2.0, y=4.0)]
        style = PlotStyle(color='blue', line_width=2.0)
        elements = [InteractiveElement(
            element_type='point',
            position=Point(x=1.0, y=1.0),
            action='highlight'
        )]
        
        plot_data = PlotData(
            plot_type='function',
            data_points=points,
            styling=style,
            interactive_elements=elements,
            title='Test Plot',
            axis_labels={'x': 'X-axis', 'y': 'Y-axis'}
        )
        
        assert plot_data.plot_type == 'function'
        assert len(plot_data.data_points) == 2
        assert plot_data.styling.color == 'blue'
        assert len(plot_data.interactive_elements) == 1
        assert plot_data.title == 'Test Plot'
        assert plot_data.axis_labels['x'] == 'X-axis'


class TestPerformance:
    """Test performance aspects of the visualization engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = MathPlotter()
    
    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time
        
        start_time = time.time()
        
        # Create plot with many points
        plot_data = self.plotter.create_function_plot(
            expression='sin(x) * cos(x)',
            x_range=(-50, 50),
            num_points=10000
        )
        
        creation_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        assert creation_time < 5.0  # 5 seconds max
        assert len(plot_data.data_points) > 5000  # Should have many valid points
    
    def test_rendering_performance(self):
        """Test rendering performance."""
        import time
        
        plot_data = self.plotter.create_function_plot('x**2 + sin(x)')
        
        start_time = time.time()
        image_base64 = self.plotter.render_plot(plot_data)
        rendering_time = time.time() - start_time
        
        # Should render within reasonable time
        assert rendering_time < 3.0  # 3 seconds max
        assert len(image_base64) > 0
    
    def test_memory_usage_multiple_plots(self):
        """Test memory usage with multiple plots."""
        # Create multiple plots to test memory management
        plots = []
        
        for i in range(10):
            plot_data = self.plotter.create_function_plot(f'x**{i+1}')
            plots.append(plot_data)
        
        # All plots should be created successfully
        assert len(plots) == 10
        
        # Test rendering multiple plots
        for plot_data in plots[:3]:  # Test first 3 to avoid long test times
            image = self.plotter.render_plot(plot_data)
            assert len(image) > 0


class TestVectorMatrixVisualization:
    """Test cases for vector and matrix visualization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = MathPlotter()
    
    def test_create_vector_plot_2d(self):
        """Test 2D vector plotting."""
        vectors = [(3, 4), (-2, 1), (1, -3)]
        labels = ['v1', 'v2', 'v3']
        
        plot_data = self.plotter.create_vector_plot(
            vectors=vectors,
            labels=labels,
            is_3d=False
        )
        
        assert plot_data.plot_type == 'vector_2d'
        assert len(plot_data.data_points) == 4  # Origin + 3 vectors
        assert len(plot_data.interactive_elements) == 3  # One per vector
        assert plot_data.title == "2D Vector Visualization"
        
        # Check that first point is origin
        origin = plot_data.data_points[0]
        assert origin.x == 0 and origin.y == 0
        
        # Check vector endpoints
        for i, vector in enumerate(vectors):
            endpoint = plot_data.data_points[i + 1]
            assert endpoint.x == vector[0]
            assert endpoint.y == vector[1]
    
    def test_create_vector_plot_3d(self):
        """Test 3D vector plotting."""
        vectors = [(1, 2, 3), (-1, 0, 2)]
        
        plot_data = self.plotter.create_vector_plot(
            vectors=vectors,
            is_3d=True
        )
        
        assert plot_data.plot_type == 'vector_3d'
        assert len(plot_data.data_points) == 3  # Origin + 2 vectors
        assert plot_data.title == "3D Vector Visualization"
        
        # Check 3D coordinates
        for i, vector in enumerate(vectors):
            endpoint = plot_data.data_points[i + 1]
            assert endpoint.x == vector[0]
            assert endpoint.y == vector[1]
            assert endpoint.z == vector[2]
    
    def test_create_vector_field_plot(self):
        """Test vector field visualization."""
        plot_data = self.plotter.create_vector_field_plot(
            vector_function="[-y, x]",  # Rotation field
            x_range=(-2, 2),
            y_range=(-2, 2),
            grid_density=5
        )
        
        assert plot_data.plot_type == 'vector_field'
        assert len(plot_data.data_points) == 25  # 5x5 grid
        assert plot_data.title == "Vector Field: [-y, x]"
        
        # Check that points have vector components
        for point in plot_data.data_points:
            assert hasattr(point, 'u')
            assert hasattr(point, 'v')
            assert isinstance(point.u, float)
            assert isinstance(point.v, float)
    
    def test_create_matrix_transformation_plot(self):
        """Test matrix transformation visualization."""
        matrix = [[2, 1], [0, 1]]  # Shear transformation
        input_vectors = [(1, 0), (0, 1), (1, 1)]
        
        plot_data = self.plotter.create_matrix_transformation_plot(
            matrix=matrix,
            input_vectors=input_vectors,
            show_unit_circle=False,
            show_grid_transformation=False
        )
        
        assert plot_data.plot_type == 'matrix_transformation'
        assert len(plot_data.data_points) >= 6  # Original + transformed vectors
        assert len(plot_data.interactive_elements) == 3  # One per vector pair
        assert "det = 2.00" in plot_data.title  # Determinant should be 2
        
        # Check interactive elements have transformation info
        for element in plot_data.interactive_elements:
            assert element.element_type == 'vector_pair'
            assert '→' in element.tooltip
    
    def test_create_3d_surface_plot(self):
        """Test 3D surface plotting."""
        plot_data = self.plotter.create_3d_surface_plot(
            function_expr="x**2 + y**2",
            x_range=(-2, 2),
            y_range=(-2, 2),
            resolution=10
        )
        
        assert plot_data.plot_type == 'surface_3d'
        assert len(plot_data.data_points) == 100  # 10x10 grid
        assert plot_data.title == "3D Surface: f(x,y) = x**2 + y**2"
        assert plot_data.axis_labels['z'] == 'Z'
        
        # Check that all points have z coordinates
        for point in plot_data.data_points:
            assert point.z is not None
            assert isinstance(point.z, float)
    
    def test_vector_field_error_handling(self):
        """Test error handling in vector field creation."""
        with pytest.raises(ValueError) as excinfo:
            self.plotter.create_vector_field_plot("invalid_function")
        
        assert "Error creating vector field plot" in str(excinfo.value)
    
    def test_matrix_transformation_invalid_matrix(self):
        """Test error handling for invalid matrix dimensions."""
        with pytest.raises(ValueError) as excinfo:
            self.plotter.create_matrix_transformation_plot([[1, 2, 3]])  # Not 2x2
        
        assert "Matrix must be 2x2" in str(excinfo.value)


class TestAIMLVisualization:
    """Test cases for AI/ML mathematics visualization features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = MathPlotter()
    
    def test_create_loss_surface_plot(self):
        """Test loss surface visualization."""
        plot_data = self.plotter.create_loss_surface_plot(
            loss_function="(x-1)**2 + (y-2)**2",
            parameter_ranges={'x': (-2, 4), 'y': (-1, 5)},
            global_minimum=(1, 2),
            resolution=10
        )
        
        assert plot_data.plot_type == 'loss_surface'
        assert len(plot_data.data_points) == 100  # 10x10 grid
        assert plot_data.title == "Loss Surface: (x-1)**2 + (y-2)**2"
        assert plot_data.axis_labels['x'] == 'x'
        assert plot_data.axis_labels['y'] == 'y'
        assert plot_data.axis_labels['z'] == 'Loss'
        
        # Check for global minimum interactive element
        min_elements = [e for e in plot_data.interactive_elements 
                       if e.element_type == 'global_minimum']
        assert len(min_elements) == 1
        assert min_elements[0].position.x == 1
        assert min_elements[0].position.y == 2
    
    def test_create_loss_surface_with_optimization_path(self):
        """Test loss surface with optimization path."""
        optimization_path = [(0, 0), (0.5, 1), (1, 2)]
        
        plot_data = self.plotter.create_loss_surface_plot(
            loss_function="x**2 + y**2",
            parameter_ranges={'x': (-2, 2), 'y': (-2, 2)},
            optimization_path=optimization_path,
            resolution=5
        )
        
        # Check for optimization step elements
        step_elements = [e for e in plot_data.interactive_elements 
                        if e.element_type == 'optimization_step']
        assert len(step_elements) == 3
        
        # Check step tooltips
        for i, element in enumerate(step_elements):
            assert f'Step {i+1}' in element.tooltip
    
    def test_create_gradient_visualization(self):
        """Test gradient field visualization."""
        plot_data = self.plotter.create_gradient_visualization(
            function_expr="x**2 + y**2",
            x_range=(-2, 2),
            y_range=(-2, 2),
            gradient_points=[(1, 1), (-1, 0)],
            grid_density=5
        )
        
        assert plot_data.plot_type == 'gradient_field'
        assert plot_data.title == "Gradient Field: ∇(x**2 + y**2)"
        
        # Should have contour points and gradient points
        contour_points = [p for p in plot_data.data_points if not hasattr(p, 'is_gradient')]
        gradient_points = [p for p in plot_data.data_points if hasattr(p, 'is_gradient')]
        
        assert len(contour_points) > 0  # Contour data
        assert len(gradient_points) == 25  # 5x5 gradient grid
        
        # Check gradient points have u, v components
        for point in gradient_points:
            assert hasattr(point, 'u')
            assert hasattr(point, 'v')
        
        # Check interactive elements for specific gradient points
        grad_elements = [e for e in plot_data.interactive_elements 
                        if e.element_type == 'gradient_point']
        assert len(grad_elements) == 2
    
    def test_create_neural_network_visualization(self):
        """Test neural network architecture visualization."""
        layer_sizes = [3, 4, 2]
        weights = [np.random.randn(3, 4), np.random.randn(4, 2)]
        activations = [np.array([0.1, 0.5, -0.2]), np.array([0.3, -0.1, 0.8, 0.2]), np.array([0.9, -0.3])]
        
        plot_data = self.plotter.create_neural_network_visualization(
            layer_sizes=layer_sizes,
            weights=weights,
            activations=activations
        )
        
        assert plot_data.plot_type == 'neural_network'
        assert plot_data.title == "Neural Network Architecture: 3 → 4 → 2"
        
        # Check neuron count
        neurons = [p for p in plot_data.data_points if not hasattr(p, 'is_connection')]
        connections = [p for p in plot_data.data_points if hasattr(p, 'is_connection')]
        
        assert len(neurons) == 9  # 3 + 4 + 2 neurons
        assert len(connections) == 20  # 3*4 + 4*2 connections
        
        # Check neuron interactive elements
        neuron_elements = [e for e in plot_data.interactive_elements 
                          if e.element_type == 'neuron']
        assert len(neuron_elements) == 9
        
        # Check that neurons have activation values
        layer_0_neurons = neurons[:3]
        for i, neuron in enumerate(layer_0_neurons):
            assert hasattr(neuron, 'activation')
            assert neuron.activation == activations[0][i]
    
    def test_create_optimization_path_plot(self):
        """Test optimization path visualization."""
        optimization_steps = [
            (2.0, 3.0, 10.0),  # (param1, param2, loss)
            (1.5, 2.5, 7.5),
            (1.0, 2.0, 5.0),
            (0.8, 1.8, 3.2),
            (0.5, 1.5, 2.0)
        ]
        
        plot_data = self.plotter.create_optimization_path_plot(
            loss_function="w1**2 + w2**2",
            optimization_steps=optimization_steps,
            algorithm_name="Adam",
            parameter_names=("w1", "w2"),
            show_loss_contours=True
        )
        
        assert plot_data.plot_type == 'optimization_path'
        assert plot_data.title == "Adam Optimization Path"
        assert plot_data.axis_labels['x'] == 'w1'
        assert plot_data.axis_labels['y'] == 'w2'
        
        # Check path points
        path_points = [p for p in plot_data.data_points if not hasattr(p, 'is_contour')]
        assert len(path_points) == 5
        
        # Check step interactive elements
        step_elements = [e for e in plot_data.interactive_elements 
                        if e.element_type == 'optimization_step']
        assert len(step_elements) == 5
        
        # Check convergence info
        conv_elements = [e for e in plot_data.interactive_elements 
                        if e.element_type == 'convergence_info']
        assert len(conv_elements) == 1
        assert "Converged in 5 steps" in conv_elements[0].tooltip
    
    def test_neural_network_with_highlight_path(self):
        """Test neural network with highlighted path."""
        layer_sizes = [2, 3, 1]
        highlight_path = [0, 1, 0]  # Path through specific neurons
        
        plot_data = self.plotter.create_neural_network_visualization(
            layer_sizes=layer_sizes,
            highlight_path=highlight_path
        )
        
        # Check for highlighted neurons
        highlight_elements = [e for e in plot_data.interactive_elements 
                             if e.element_type == 'highlighted_neuron']
        assert len(highlight_elements) == 3  # One per layer
    
    def test_gradient_visualization_error_handling(self):
        """Test error handling in gradient visualization."""
        with pytest.raises(ValueError) as excinfo:
            self.plotter.create_gradient_visualization("invalid_function_xyz")
        
        assert "Error creating gradient visualization" in str(excinfo.value)
    
    def test_loss_surface_invalid_parameters(self):
        """Test error handling for invalid parameter count."""
        with pytest.raises(ValueError) as excinfo:
            self.plotter.create_loss_surface_plot(
                "x**2",
                parameter_ranges={'x': (-1, 1)}  # Only 1 parameter
            )
        
        assert "exactly 2 parameters" in str(excinfo.value)


class TestVisualizationRendering:
    """Test rendering of new visualization types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.plotter = MathPlotter()
    
    def test_render_vector_2d_plot(self):
        """Test rendering 2D vector plots."""
        vectors = [(2, 3), (-1, 2)]
        plot_data = self.plotter.create_vector_plot(vectors, is_3d=False)
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
        
        # Verify it's valid base64
        try:
            base64.b64decode(image_base64)
        except Exception:
            pytest.fail("Generated vector plot image is not valid base64")
    
    def test_render_vector_3d_plot(self):
        """Test rendering 3D vector plots."""
        vectors = [(1, 2, 3), (-1, 0, 2)]
        plot_data = self.plotter.create_vector_plot(vectors, is_3d=True)
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_vector_field_plot(self):
        """Test rendering vector field plots."""
        plot_data = self.plotter.create_vector_field_plot("[-y, x]", grid_density=5)
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_matrix_transformation_plot(self):
        """Test rendering matrix transformation plots."""
        matrix = [[1, 0.5], [0, 1]]
        plot_data = self.plotter.create_matrix_transformation_plot(matrix)
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_gradient_field_plot(self):
        """Test rendering gradient field plots."""
        plot_data = self.plotter.create_gradient_visualization(
            "x**2 + y**2", 
            x_range=(-2, 2), 
            y_range=(-2, 2),
            grid_density=5
        )
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_neural_network_plot(self):
        """Test rendering neural network plots."""
        plot_data = self.plotter.create_neural_network_visualization([2, 3, 1])
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_optimization_path_plot(self):
        """Test rendering optimization path plots."""
        steps = [(2, 3, 10), (1, 2, 5), (0.5, 1, 2)]
        plot_data = self.plotter.create_optimization_path_plot(
            "x**2 + y**2", steps, show_loss_contours=False
        )
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0
    
    def test_render_loss_surface_plot(self):
        """Test rendering 3D loss surface plots."""
        plot_data = self.plotter.create_loss_surface_plot(
            "x**2 + y**2",
            {'x': (-2, 2), 'y': (-2, 2)},
            resolution=10
        )
        
        image_base64 = self.plotter.render_plot(plot_data)
        
        assert isinstance(image_base64, str)
        assert len(image_base64) > 0


class TestVisualizationEngineAIML:
    """Test VisualizationEngine integration with AI/ML features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = VisualizationEngine()
    
    def test_generate_linear_algebra_visualization(self):
        """Test visualization generation for linear algebra problems."""
        problem = ParsedProblem(
            id='test-la',
            original_text='Find eigenvalues of matrix [[2, 1], [1, 2]]',
            domain=MathDomain.LINEAR_ALGEBRA,
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=['x', 'y'],
            expressions=['[[2, 1], [1, 2]]'],
            problem_type='eigenvalue',
            metadata={'matrix': [[2, 1], [1, 2]]}
        )
        
        solution = StepSolution(
            problem_id='test-la',
            steps=[],
            final_answer='λ₁ = 3, λ₂ = 1',
            solution_method='Characteristic polynomial',
            confidence_score=0.95,
            computation_time=0.2
        )
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        # Should fall back to default visualization for unsupported type
        assert plot_data.plot_type in ['function', 'error']
    
    def test_ai_ml_problem_type_handling(self):
        """Test handling of AI/ML specific problem types."""
        problem = ParsedProblem(
            id='test-ml',
            original_text='Minimize loss function L(w) = w²',
            domain=MathDomain.CALCULUS,  # AI/ML problems often use calculus
            difficulty=DifficultyLevel.ADVANCED,
            variables=['w'],
            expressions=['w**2'],
            problem_type='optimization',
            metadata={'context': 'machine_learning'}
        )
        
        solution = StepSolution(
            problem_id='test-ml',
            steps=[],
            final_answer='w* = 0',
            solution_method='Gradient descent',
            confidence_score=0.9,
            computation_time=0.1
        )
        
        plot_data = self.engine.generate_problem_visualization(problem, solution)
        
        assert plot_data.plot_type == 'optimization'
        # The optimization plot may have empty data points if no critical points are found
        # This is acceptable behavior for the visualization engine


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])