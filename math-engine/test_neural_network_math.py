"""
Tests for neural network mathematics module.
Tests backpropagation accuracy, activation function analysis, and educational effectiveness.
"""

import unittest
import numpy as np
from sympy import symbols, exp, tanh
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from neural_network_math import (
    ActivationFunction, NeuralNetworkLayer, BackpropagationExplainer, 
    NeuralNetworkMathematics
)
from models import ParsedProblem, MathDomain, DifficultyLevel


class TestActivationFunction(unittest.TestCase):
    """Test activation function implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        x = symbols('x')
        sigmoid_expr = 1 / (1 + exp(-x))
        sigmoid_deriv = sigmoid_expr * (1 - sigmoid_expr)
        self.sigmoid = ActivationFunction(
            'sigmoid', sigmoid_expr, sigmoid_deriv, (-10, 10), (0, 1)
        )
    
    def test_sigmoid_evaluation(self):
        """Test sigmoid function evaluation."""
        # Test at x = 0 (should be 0.5)
        result = self.sigmoid.evaluate(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.5, places=6)
        
        # Test at large positive value (should approach 1)
        result = self.sigmoid.evaluate(np.array([10.0]))
        self.assertAlmostEqual(result[0], 1.0, places=3)
        
        # Test at large negative value (should approach 0)
        result = self.sigmoid.evaluate(np.array([-10.0]))
        self.assertAlmostEqual(result[0], 0.0, places=3)
    
    def test_sigmoid_derivative(self):
        """Test sigmoid derivative evaluation."""
        # At x = 0, derivative should be 0.25
        result = self.sigmoid.evaluate_derivative(np.array([0.0]))
        self.assertAlmostEqual(result[0], 0.25, places=6)
        
        # Derivative should be positive for all inputs
        x_vals = np.linspace(-5, 5, 100)
        deriv_vals = self.sigmoid.evaluate_derivative(x_vals)
        self.assertTrue(np.all(deriv_vals > 0))
    
    def test_activation_properties(self):
        """Test activation function properties."""
        properties = self.sigmoid.get_properties()
        
        self.assertEqual(properties['name'], 'sigmoid')
        self.assertEqual(properties['domain'], (-10, 10))
        self.assertEqual(properties['range'], (0, 1))
        self.assertTrue(properties['monotonic'])
        self.assertTrue(properties['bounded'])
        self.assertFalse(properties['zero_centered'])
        self.assertTrue(properties['differentiable'])


class TestNeuralNetworkLayer(unittest.TestCase):
    """Test neural network layer implementation."""
    
    def setUp(self):
        """Set up test fixtures."""
        x = symbols('x')
        sigmoid_expr = 1 / (1 + exp(-x))
        sigmoid_deriv = sigmoid_expr * (1 - sigmoid_expr)
        self.sigmoid = ActivationFunction(
            'sigmoid', sigmoid_expr, sigmoid_deriv, (-10, 10), (0, 1)
        )
        
        # Create a simple 2->3 layer
        self.layer = NeuralNetworkLayer(2, 3, self.sigmoid)
    
    def test_layer_initialization(self):
        """Test layer initialization."""
        self.assertEqual(self.layer.input_size, 2)
        self.assertEqual(self.layer.output_size, 3)
        self.assertEqual(self.layer.weights.shape, (3, 2))
        self.assertEqual(self.layer.biases.shape, (3, 1))
    
    def test_forward_pass(self):
        """Test forward pass computation."""
        # Input: 2x1 vector
        x = np.array([[1.0], [2.0]])
        
        output = self.layer.forward(x)
        
        # Check output shape
        self.assertEqual(output.shape, (3, 1))
        
        # Check that intermediate values are stored
        self.assertIsNotNone(self.layer.z)
        self.assertIsNotNone(self.layer.a)
        self.assertIsNotNone(self.layer.input)
        
        # Check that output values are in sigmoid range (0, 1)
        self.assertTrue(np.all(output > 0))
        self.assertTrue(np.all(output < 1))
    
    def test_backward_pass(self):
        """Test backward pass computation."""
        # First do forward pass
        x = np.array([[1.0], [2.0]])
        output = self.layer.forward(x)
        
        # Simulate gradient from next layer
        dA = np.array([[0.1], [0.2], [0.3]])
        
        dA_prev, dW, db = self.layer.backward(dA)
        
        # Check gradient shapes
        self.assertEqual(dA_prev.shape, (2, 1))  # Same as input
        self.assertEqual(dW.shape, (3, 2))       # Same as weights
        self.assertEqual(db.shape, (3, 1))       # Same as biases
        
        # Check that gradients are reasonable (not NaN or infinite)
        self.assertTrue(np.all(np.isfinite(dA_prev)))
        self.assertTrue(np.all(np.isfinite(dW)))
        self.assertTrue(np.all(np.isfinite(db)))


class TestBackpropagationExplainer(unittest.TestCase):
    """Test backpropagation explanation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.explainer = BackpropagationExplainer()
    
    def test_activation_functions_creation(self):
        """Test creation of activation functions."""
        functions = self.explainer.activation_functions
        
        # Check that all expected functions are created
        expected_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
        for func_name in expected_functions:
            self.assertIn(func_name, functions)
            self.assertIsInstance(functions[func_name], ActivationFunction)
    
    def test_backpropagation_explanation(self):
        """Test step-by-step backpropagation explanation."""
        network_structure = [2, 3, 1]
        activation_names = ['sigmoid', 'sigmoid']
        
        # Create sample data
        input_data = np.array([[1.0], [2.0]])
        target_output = np.array([[0.5]])
        
        steps = self.explainer.explain_backpropagation_step_by_step(
            network_structure, activation_names, input_data, target_output
        )
        
        # Check that we have a reasonable number of steps
        self.assertGreaterEqual(len(steps), 10)
        
        # Check that all steps have required fields
        for step in steps:
            self.assertIsNotNone(step.step_number)
            self.assertIsNotNone(step.operation)
            self.assertIsNotNone(step.explanation)
            self.assertIsNotNone(step.mathematical_expression)
            self.assertIsNotNone(step.intermediate_result)
        
        # Check that key operations are present
        operations = [step.operation for step in steps]
        self.assertTrue(any('Initialize' in op for op in operations))
        self.assertTrue(any('Forward Pass' in op for op in operations))
        self.assertTrue(any('Compute Loss' in op for op in operations))
        self.assertTrue(any('Backward Pass' in op for op in operations))
        self.assertTrue(any('Update Weights' in op for op in operations))
    
    def test_activation_function_analysis(self):
        """Test activation function analysis."""
        analysis = self.explainer.analyze_activation_functions()
        
        # Check that all activation functions are analyzed
        expected_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
        for func_name in expected_functions:
            self.assertIn(func_name, analysis)
            
            func_analysis = analysis[func_name]
            self.assertIn('properties', func_analysis)
            self.assertIn('visualization', func_analysis)
            self.assertIn('ml_context', func_analysis)
            
            # Check visualization data
            viz = func_analysis['visualization']
            self.assertIn('x_values', viz)
            self.assertIn('function_values', viz)
            self.assertIn('derivative_values', viz)
            
            # Check that we have reasonable amount of data points
            self.assertEqual(len(viz['x_values']), 1000)
            self.assertEqual(len(viz['function_values']), 1000)
            self.assertEqual(len(viz['derivative_values']), 1000)
            
            # Check ML context
            ml_context = func_analysis['ml_context']
            self.assertIn('usage', ml_context)
            self.assertIn('advantages', ml_context)
            self.assertIn('disadvantages', ml_context)
            self.assertIn('gradient_flow', ml_context)
    
    def test_learning_rate_impact_demonstration(self):
        """Test learning rate impact demonstration."""
        network_structure = [2, 3, 1]
        learning_rates = [0.01, 0.1, 1.0]
        
        results = self.explainer.demonstrate_learning_rate_impact(
            network_structure, learning_rates, epochs=5
        )
        
        # Check that results are generated for all learning rates
        for lr in learning_rates:
            key = f'lr_{lr}'
            self.assertIn(key, results)
            
            result = results[key]
            self.assertEqual(result['learning_rate'], lr)
            self.assertIn('losses', result)
            self.assertIn('weight_changes', result)
            self.assertIn('final_loss', result)
            self.assertIn('convergence_analysis', result)
            
            # Check that we have the right number of loss values
            self.assertEqual(len(result['losses']), 5)  # 5 epochs
            self.assertEqual(len(result['weight_changes']), 5)
            
            # Check convergence analysis
            conv_analysis = result['convergence_analysis']
            self.assertIn('status', conv_analysis)
            self.assertIn('is_decreasing', conv_analysis)
            self.assertIn('is_oscillating', conv_analysis)
            self.assertIn('convergence_rate', conv_analysis)
            self.assertIn('final_loss', conv_analysis)
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        # Test decreasing losses (good convergence)
        decreasing_losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        analysis = self.explainer._analyze_convergence(decreasing_losses, 0.01)
        self.assertTrue(analysis['is_decreasing'])
        self.assertFalse(analysis['is_oscillating'])
        self.assertGreater(analysis['convergence_rate'], 0)
        
        # Test oscillating losses
        oscillating_losses = [1.0, 0.5, 1.2, 0.3, 1.1, 0.4]
        analysis = self.explainer._analyze_convergence(oscillating_losses, 0.5)
        self.assertTrue(analysis['is_oscillating'])
        
        # Test high learning rate
        analysis = self.explainer._analyze_convergence([1.0, 2.0, 4.0], 2.0)
        self.assertIn('Too high', analysis['status'])
        
        # Test low learning rate
        analysis = self.explainer._analyze_convergence([1.0, 0.999, 0.998], 1e-7)
        self.assertIn('Too low', analysis['status'])


class TestNeuralNetworkMathematics(unittest.TestCase):
    """Test main neural network mathematics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nn_math = NeuralNetworkMathematics()
    
    def test_backpropagation_problem_solving(self):
        """Test backpropagation problem solving integration."""
        # Create a mock backpropagation problem
        problem = ParsedProblem(
            id="test_backprop_1",
            original_text="Explain backpropagation for a 2-3-1 network with sigmoid activation",
            domain=MathDomain.AI_ML_MATH,
            problem_type="backpropagation",
            difficulty=DifficultyLevel.ADVANCED,
            variables=[],
            expressions=[],
            metadata={}
        )
        
        solution = self.nn_math.solve_backpropagation_problem(problem)
        
        # Check solution structure
        self.assertIsNotNone(solution.steps)
        self.assertGreater(len(solution.steps), 10)
        self.assertIn("backpropagation", solution.solution_method.lower())
        
        # Check that key concepts are covered
        step_operations = [step.operation for step in solution.steps]
        self.assertTrue(any('Initialize' in op for op in step_operations))
        self.assertTrue(any('Forward' in op for op in step_operations))
        self.assertTrue(any('Loss' in op for op in step_operations))
        self.assertTrue(any('Backward' in op for op in step_operations))
        self.assertTrue(any('Update' in op for op in step_operations))
        
        # Check final answer
        self.assertIn("network", solution.final_answer.lower())
        self.assertIn("activation", solution.final_answer.lower())
    
    def test_network_extraction_from_problem(self):
        """Test extraction of network parameters from problem text."""
        # Test network structure extraction
        problem1 = ParsedProblem(
            id="test_extract_1",
            original_text="Train a 3-4-2 network with ReLU activation",
            domain=MathDomain.AI_ML_MATH,
            problem_type="backpropagation",
            difficulty=DifficultyLevel.ADVANCED,
            variables=[],
            expressions=[],
            metadata={}
        )
        
        structure, activations, lr = self.nn_math._extract_network_from_problem(problem1)
        self.assertEqual(structure, [3, 4, 2])
        self.assertEqual(activations, ['relu', 'sigmoid'])
        
        # Test learning rate extraction
        problem2 = ParsedProblem(
            id="test_extract_2",
            original_text="Use learning rate 0.05 for training",
            domain=MathDomain.AI_ML_MATH,
            problem_type="backpropagation",
            difficulty=DifficultyLevel.ADVANCED,
            variables=[],
            expressions=[],
            metadata={}
        )
        
        structure, activations, lr = self.nn_math._extract_network_from_problem(problem2)
        self.assertEqual(lr, 0.05)
        
        # Test tanh activation extraction
        problem3 = ParsedProblem(
            id="test_extract_3",
            original_text="Network with tanh activation function",
            domain=MathDomain.AI_ML_MATH,
            problem_type="backpropagation",
            difficulty=DifficultyLevel.ADVANCED,
            variables=[],
            expressions=[],
            metadata={}
        )
        
        structure, activations, lr = self.nn_math._extract_network_from_problem(problem3)
        self.assertEqual(activations, ['tanh', 'sigmoid'])
    
    def test_activation_function_analysis_integration(self):
        """Test activation function analysis integration."""
        analysis = self.nn_math.get_activation_function_analysis()
        
        # Check that all expected functions are analyzed
        expected_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']
        for func_name in expected_functions:
            self.assertIn(func_name, analysis)
            
            func_analysis = analysis[func_name]
            
            # Check completeness of analysis
            self.assertIn('properties', func_analysis)
            self.assertIn('visualization', func_analysis)
            self.assertIn('ml_context', func_analysis)
            
            # Check that ML context is educational
            ml_context = func_analysis['ml_context']
            for key in ['usage', 'advantages', 'disadvantages', 'gradient_flow']:
                self.assertIn(key, ml_context)
                self.assertGreater(len(ml_context[key]), 20)  # Non-trivial explanation
    
    def test_learning_rate_effects_demonstration(self):
        """Test learning rate effects demonstration."""
        results = self.nn_math.demonstrate_learning_rate_effects()
        
        # Check that results are generated for default learning rates
        default_lrs = [0.001, 0.01, 0.1, 1.0]
        for lr in default_lrs:
            key = f'lr_{lr}'
            self.assertIn(key, results)
            
            result = results[key]
            self.assertEqual(result['learning_rate'], lr)
            
            # Check that training curves are generated
            self.assertIn('losses', result)
            self.assertIn('weight_changes', result)
            self.assertGreater(len(result['losses']), 5)
            
            # Check convergence analysis
            self.assertIn('convergence_analysis', result)
            conv_analysis = result['convergence_analysis']
            self.assertIn('status', conv_analysis)
            self.assertIsInstance(conv_analysis['status'], str)
    
    def test_custom_network_learning_rate_demo(self):
        """Test learning rate demonstration with custom network."""
        custom_structure = [3, 5, 2]
        custom_lrs = [0.01, 0.1]
        
        results = self.nn_math.demonstrate_learning_rate_effects(
            custom_structure, custom_lrs
        )
        
        # Check that results are generated for custom parameters
        for lr in custom_lrs:
            key = f'lr_{lr}'
            self.assertIn(key, results)
            
            result = results[key]
            self.assertEqual(result['learning_rate'], lr)
            
            # Check that the network structure affects the results
            self.assertIn('losses', result)
            self.assertIn('weight_changes', result)


class TestEducationalEffectiveness(unittest.TestCase):
    """Test educational effectiveness of neural network mathematics."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.nn_math = NeuralNetworkMathematics()
        self.explainer = BackpropagationExplainer()
    
    def test_step_by_step_clarity(self):
        """Test that step-by-step explanations are clear and educational."""
        network_structure = [2, 3, 1]
        activation_names = ['sigmoid', 'sigmoid']
        input_data = np.array([[1.0], [2.0]])
        target_output = np.array([[0.5]])
        
        steps = self.explainer.explain_backpropagation_step_by_step(
            network_structure, activation_names, input_data, target_output
        )
        
        # Check that explanations are educational
        for step in steps:
            # Each step should have a meaningful explanation
            self.assertGreater(len(step.explanation), 20)
            
            # Mathematical expressions should be present
            self.assertGreater(len(step.mathematical_expression), 5)
            
            # Results should be informative
            self.assertGreater(len(step.intermediate_result), 10)
        
        # Check that the sequence makes logical sense
        operations = [step.operation for step in steps]
        
        # Initialize should come before forward pass
        init_idx = next(i for i, op in enumerate(operations) if 'Initialize' in op)
        forward_idx = next(i for i, op in enumerate(operations) if 'Forward Pass' in op)
        self.assertLess(init_idx, forward_idx)
        
        # Forward pass should come before loss computation
        loss_idx = next(i for i, op in enumerate(operations) if 'Loss' in op)
        self.assertLess(forward_idx, loss_idx)
        
        # Loss should come before backward pass
        backward_idx = next(i for i, op in enumerate(operations) if 'Backward Pass' in op)
        self.assertLess(loss_idx, backward_idx)
    
    def test_activation_function_educational_content(self):
        """Test that activation function analysis is educational."""
        analysis = self.explainer.analyze_activation_functions()
        
        for func_name, func_analysis in analysis.items():
            ml_context = func_analysis['ml_context']
            
            # Check that each context element is educational
            for context_key, context_value in ml_context.items():
                self.assertIsInstance(context_value, str)
                self.assertGreater(len(context_value), 30)  # Substantial explanation
                
                # Check that context is relevant to ML
                ml_keywords = ['neural', 'network', 'gradient', 'learning', 'training', 'neuron']
                has_ml_keyword = any(keyword in context_value.lower() for keyword in ml_keywords)
                self.assertTrue(has_ml_keyword, f"Context for {func_name}.{context_key} lacks ML relevance")
    
    def test_learning_rate_educational_insights(self):
        """Test that learning rate demonstration provides educational insights."""
        results = self.nn_math.demonstrate_learning_rate_effects()
        
        # Check that different learning rates produce different outcomes
        loss_patterns = {}
        for key, result in results.items():
            lr = result['learning_rate']
            final_loss = result['final_loss']
            convergence_status = result['convergence_analysis']['status']
            
            loss_patterns[lr] = (final_loss, convergence_status)
        
        # Very low learning rate should be slow
        low_lr_status = loss_patterns[0.001][1]
        self.assertIn('slow', low_lr_status.lower())
        
        # High learning rate should have different behavior than low learning rate
        high_lr_status = loss_patterns[1.0][1]
        # Either it should have issues OR be faster than low learning rate
        self.assertTrue(
            'high' in high_lr_status.lower() or 
            'oscillat' in high_lr_status.lower() or 
            'diverg' in high_lr_status.lower() or
            'good' in high_lr_status.lower() or
            'steady' in high_lr_status.lower()
        )
        
        # Check that convergence analysis provides actionable insights
        for result in results.values():
            conv_analysis = result['convergence_analysis']
            status = conv_analysis['status']
            
            # Status should be descriptive and actionable
            self.assertGreater(len(status), 10)
            
            # Should provide guidance
            guidance_keywords = ['reduce', 'increase', 'consider', 'good', 'slow', 'fast', 'high', 'low']
            has_guidance = any(keyword in status.lower() for keyword in guidance_keywords)
            self.assertTrue(has_guidance, f"Status '{status}' lacks actionable guidance")


if __name__ == '__main__':
    unittest.main()