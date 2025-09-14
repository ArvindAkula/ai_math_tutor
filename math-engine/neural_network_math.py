"""
Neural network mathematics module for educational purposes.
Implements backpropagation step-by-step explanation, activation function visualization,
and weight update demonstrations with learning rate impact analysis.
"""

import numpy as np
import sympy as sp
from sympy import symbols, exp, tanh, log, diff, lambdify, Matrix
from typing import List, Dict, Any, Optional, Tuple, Callable
import matplotlib.pyplot as plt
import sys
import os

# Add shared models to path
from models import (
    ParsedProblem, StepSolution, SolutionStep, ValidationResult,
    MathDomain, DifficultyLevel, ComputationError
)


class ActivationFunction:
    """Represents an activation function with its properties."""
    
    def __init__(self, name: str, expression: sp.Expr, derivative: sp.Expr, 
                 domain: Tuple[float, float] = (-5, 5), 
                 range_vals: Tuple[float, float] = None):
        self.name = name
        self.expression = expression
        self.derivative = derivative
        self.domain = domain
        self.range_vals = range_vals
        
        # Create numerical functions
        x = symbols('x')
        self.func = lambdify(x, expression, 'numpy')
        self.deriv_func = lambdify(x, derivative, 'numpy')
    
    def evaluate(self, x_vals: np.ndarray) -> np.ndarray:
        """Evaluate activation function at given points."""
        return self.func(x_vals)
    
    def evaluate_derivative(self, x_vals: np.ndarray) -> np.ndarray:
        """Evaluate derivative at given points."""
        return self.deriv_func(x_vals)
    
    def get_properties(self) -> Dict[str, Any]:
        """Get mathematical properties of the activation function."""
        properties = {
            'name': self.name,
            'expression': str(self.expression),
            'derivative': str(self.derivative),
            'domain': self.domain,
            'range': self.range_vals,
            'monotonic': self._is_monotonic(),
            'bounded': self.range_vals is not None,
            'zero_centered': self._is_zero_centered(),
            'differentiable': True  # All our functions are differentiable
        }
        return properties
    
    def _is_monotonic(self) -> bool:
        """Check if function is monotonic."""
        # For common activation functions
        if self.name in ['sigmoid', 'tanh', 'relu', 'leaky_relu']:
            return True
        return False
    
    def _is_zero_centered(self) -> bool:
        """Check if function is zero-centered."""
        return self.name in ['tanh']


class NeuralNetworkLayer:
    """Represents a layer in a neural network."""
    
    def __init__(self, input_size: int, output_size: int, 
                 activation: ActivationFunction, weights: np.ndarray = None, 
                 biases: np.ndarray = None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Initialize weights and biases
        if weights is not None:
            self.weights = weights
        else:
            # Xavier initialization
            self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / input_size)
        
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.zeros((output_size, 1))
        
        # Store intermediate values for backpropagation
        self.z = None  # Linear combination (before activation)
        self.a = None  # Activation output
        self.input = None  # Input to this layer
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.input = x
        self.z = np.dot(self.weights, x) + self.biases
        self.a = self.activation.evaluate(self.z)
        return self.a
    
    def backward(self, dA: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Backward pass through the layer."""
        m = self.input.shape[1]  # Number of examples
        
        # Compute dZ (derivative w.r.t. linear combination)
        dZ = dA * self.activation.evaluate_derivative(self.z)
        
        # Compute gradients
        dW = (1/m) * np.dot(dZ, self.input.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.weights.T, dZ)
        
        return dA_prev, dW, db


class BackpropagationExplainer:
    """Explains backpropagation step by step."""
    
    def __init__(self):
        """Initialize backpropagation explainer."""
        self.activation_functions = self._create_activation_functions()
    
    def _create_activation_functions(self) -> Dict[str, ActivationFunction]:
        """Create common activation functions."""
        x = symbols('x')
        
        functions = {}
        
        # Sigmoid
        sigmoid_expr = 1 / (1 + exp(-x))
        sigmoid_deriv = sigmoid_expr * (1 - sigmoid_expr)
        functions['sigmoid'] = ActivationFunction(
            'sigmoid', sigmoid_expr, sigmoid_deriv, (-10, 10), (0, 1)
        )
        
        # Tanh
        tanh_expr = tanh(x)
        tanh_deriv = 1 - tanh_expr**2
        functions['tanh'] = ActivationFunction(
            'tanh', tanh_expr, tanh_deriv, (-5, 5), (-1, 1)
        )
        
        # ReLU
        relu_expr = sp.Max(0, x)
        relu_deriv = sp.Piecewise((0, x < 0), (1, x >= 0))
        functions['relu'] = ActivationFunction(
            'relu', relu_expr, relu_deriv, (-5, 5), (0, sp.oo)
        )
        
        # Leaky ReLU
        alpha = 0.01
        leaky_relu_expr = sp.Max(alpha * x, x)
        leaky_relu_deriv = sp.Piecewise((alpha, x < 0), (1, x >= 0))
        functions['leaky_relu'] = ActivationFunction(
            'leaky_relu', leaky_relu_expr, leaky_relu_deriv, (-5, 5), (-sp.oo, sp.oo)
        )
        
        return functions
    
    def explain_backpropagation_step_by_step(self, 
                                           network_structure: List[int],
                                           activation_names: List[str],
                                           input_data: np.ndarray,
                                           target_output: np.ndarray,
                                           learning_rate: float = 0.01) -> List[SolutionStep]:
        """
        Explain backpropagation algorithm step by step.
        
        Args:
            network_structure: List of layer sizes [input, hidden1, hidden2, ..., output]
            activation_names: List of activation function names for each layer
            input_data: Input data for the network
            target_output: Target output for training
            learning_rate: Learning rate for weight updates
            
        Returns:
            List of solution steps explaining backpropagation
        """
        steps = []
        step_num = 1
        
        # Step 1: Initialize network
        layers = []
        for i in range(len(network_structure) - 1):
            activation = self.activation_functions[activation_names[i]]
            layer = NeuralNetworkLayer(
                network_structure[i], 
                network_structure[i + 1], 
                activation
            )
            layers.append(layer)
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Initialize Neural Network",
            explanation=f"Create network with structure {network_structure} and activations {activation_names}",
            mathematical_expression=f"Network: {' → '.join(map(str, network_structure))}",
            intermediate_result=f"Initialized {len(layers)} layers with random weights"
        ))
        step_num += 1
        
        # Step 2: Forward pass
        current_input = input_data
        layer_outputs = [current_input]
        
        for i, layer in enumerate(layers):
            output = layer.forward(current_input)
            layer_outputs.append(output)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation=f"Forward Pass - Layer {i+1}",
                explanation=f"Compute z = W·a + b, then apply {layer.activation.name} activation",
                mathematical_expression=f"z{i+1} = W{i+1}·a{i} + b{i+1}, a{i+1} = {layer.activation.name}(z{i+1})",
                intermediate_result=f"Layer {i+1} output shape: {output.shape}"
            ))
            step_num += 1
            current_input = output
        
        # Step 3: Compute loss
        final_output = layer_outputs[-1]
        loss = 0.5 * np.mean((final_output - target_output)**2)  # MSE loss
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Compute Loss",
            explanation="Calculate Mean Squared Error between prediction and target",
            mathematical_expression="L = (1/2m) Σ(ŷ - y)²",
            intermediate_result=f"Loss: {loss:.6f}"
        ))
        step_num += 1
        
        # Step 4: Backward pass
        # Start with loss gradient
        dA = final_output - target_output  # Gradient of MSE loss
        
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Initialize Backpropagation",
            explanation="Start with gradient of loss w.r.t. final output",
            mathematical_expression="∂L/∂a^(L) = ŷ - y",
            intermediate_result=f"Initial gradient shape: {dA.shape}"
        ))
        step_num += 1
        
        # Backpropagate through layers
        gradients = []
        for i in reversed(range(len(layers))):
            layer = layers[i]
            dA_prev, dW, db = layer.backward(dA)
            gradients.insert(0, (dW, db))
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation=f"Backward Pass - Layer {i+1}",
                explanation=f"Compute gradients using chain rule and {layer.activation.name} derivative",
                mathematical_expression=f"∂L/∂W{i+1} = (∂L/∂z{i+1}) · a{i}ᵀ, ∂L/∂z{i+1} = (∂L/∂a{i+1}) ⊙ σ'(z{i+1})",
                intermediate_result=f"Weight gradient shape: {dW.shape}, Bias gradient shape: {db.shape}"
            ))
            step_num += 1
            
            dA = dA_prev
        
        # Step 5: Update weights
        for i, (layer, (dW, db)) in enumerate(zip(layers, gradients)):
            old_weights = layer.weights.copy()
            old_biases = layer.biases.copy()
            
            layer.weights -= learning_rate * dW
            layer.biases -= learning_rate * db
            
            weight_change = np.linalg.norm(layer.weights - old_weights)
            bias_change = np.linalg.norm(layer.biases - old_biases)
            
            steps.append(SolutionStep(
                step_number=step_num,
                operation=f"Update Weights - Layer {i+1}",
                explanation=f"Apply gradient descent with learning rate {learning_rate}",
                mathematical_expression=f"W{i+1} ← W{i+1} - α·∂L/∂W{i+1}, b{i+1} ← b{i+1} - α·∂L/∂b{i+1}",
                intermediate_result=f"Weight change: {weight_change:.6f}, Bias change: {bias_change:.6f}"
            ))
            step_num += 1
        
        # Step 6: Educational insights
        steps.append(SolutionStep(
            step_number=step_num,
            operation="Backpropagation Insights",
            explanation="Key concepts in backpropagation algorithm",
            mathematical_expression="Chain Rule: ∂L/∂w = (∂L/∂a) · (∂a/∂z) · (∂z/∂w)",
            intermediate_result="Gradients flow backward, weights updated to minimize loss"
        ))
        
        return steps
    
    def analyze_activation_functions(self) -> Dict[str, Dict[str, Any]]:
        """Analyze properties of different activation functions."""
        analysis = {}
        
        for name, activation in self.activation_functions.items():
            # Generate visualization data
            x_vals = np.linspace(activation.domain[0], activation.domain[1], 1000)
            y_vals = activation.evaluate(x_vals)
            dy_vals = activation.evaluate_derivative(x_vals)
            
            # Analyze properties
            properties = activation.get_properties()
            
            # Add visualization data
            analysis[name] = {
                'properties': properties,
                'visualization': {
                    'x_values': x_vals.tolist(),
                    'function_values': y_vals.tolist(),
                    'derivative_values': dy_vals.tolist()
                },
                'ml_context': self._get_activation_ml_context(name)
            }
        
        return analysis
    
    def _get_activation_ml_context(self, activation_name: str) -> Dict[str, str]:
        """Get ML context for activation functions."""
        contexts = {
            'sigmoid': {
                'usage': "Historically used in neural network hidden layers, now mainly for binary classification output",
                'advantages': "Smooth, differentiable, outputs in (0,1) range suitable for neural networks",
                'disadvantages': "Vanishing gradient problem in deep networks, not zero-centered, computationally expensive",
                'gradient_flow': "Gradients become very small for large |x|, causing vanishing gradients in deep neural networks"
            },
            'tanh': {
                'usage': "Better than sigmoid for neural network hidden layers, zero-centered for better training",
                'advantages': "Zero-centered, stronger gradients than sigmoid in neural network training",
                'disadvantages': "Still suffers from vanishing gradient problem in deep neural networks",
                'gradient_flow': "Better gradient flow than sigmoid but still vanishes for large |x| in deep networks"
            },
            'relu': {
                'usage': "Most popular activation for hidden layers in deep neural networks",
                'advantages': "Computationally efficient, no vanishing gradient for positive inputs in neural networks",
                'disadvantages': "Dead neurons problem (neurons can stop learning during training)",
                'gradient_flow': "Perfect gradient flow for positive inputs, zero for negative in neural network training"
            },
            'leaky_relu': {
                'usage': "Addresses dead neuron problem of ReLU in neural network training",
                'advantages': "Prevents dead neurons, allows small gradient for negative inputs during neural network learning",
                'disadvantages': "Introduces hyperparameter (leak coefficient) that needs tuning in neural networks",
                'gradient_flow': "Small but non-zero gradients for negative inputs help neural network training"
            }
        }
        
        return contexts.get(activation_name, {})
    
    def demonstrate_learning_rate_impact(self, 
                                       network_structure: List[int],
                                       learning_rates: List[float],
                                       epochs: int = 10) -> Dict[str, Any]:
        """
        Demonstrate the impact of different learning rates on training.
        
        Args:
            network_structure: Network architecture
            learning_rates: List of learning rates to compare
            epochs: Number of training epochs
            
        Returns:
            Dictionary containing training curves and analysis
        """
        # Generate simple training data
        np.random.seed(42)
        X = np.random.randn(network_structure[0], 100)
        y = np.random.randn(network_structure[-1], 100)
        
        results = {}
        
        for lr in learning_rates:
            # Initialize network
            layers = []
            for i in range(len(network_structure) - 1):
                activation = self.activation_functions['sigmoid']  # Use sigmoid for consistency
                layer = NeuralNetworkLayer(
                    network_structure[i], 
                    network_structure[i + 1], 
                    activation
                )
                layers.append(layer)
            
            # Training loop
            losses = []
            weight_changes = []
            
            for epoch in range(epochs):
                # Forward pass
                current_input = X
                for layer in layers:
                    current_input = layer.forward(current_input)
                
                # Compute loss
                loss = 0.5 * np.mean((current_input - y)**2)
                losses.append(loss)
                
                # Backward pass
                dA = current_input - y
                total_weight_change = 0
                
                for i in reversed(range(len(layers))):
                    layer = layers[i]
                    dA_prev, dW, db = layer.backward(dA)
                    
                    # Update weights
                    old_weights = layer.weights.copy()
                    layer.weights -= lr * dW
                    layer.biases -= lr * db
                    
                    total_weight_change += np.linalg.norm(layer.weights - old_weights)
                    dA = dA_prev
                
                weight_changes.append(total_weight_change)
            
            results[f'lr_{lr}'] = {
                'learning_rate': lr,
                'losses': losses,
                'weight_changes': weight_changes,
                'final_loss': losses[-1],
                'convergence_analysis': self._analyze_convergence(losses, lr)
            }
        
        return results
    
    def _analyze_convergence(self, losses: List[float], learning_rate: float) -> Dict[str, Any]:
        """Analyze convergence properties of training."""
        losses = np.array(losses)
        
        # Check if loss is decreasing
        is_decreasing = np.all(np.diff(losses) <= 0.01)  # Allow small fluctuations
        
        # Check for oscillations
        diff = np.diff(losses)
        sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
        is_oscillating = sign_changes > len(losses) * 0.3
        
        # Check convergence rate
        if len(losses) > 1:
            convergence_rate = (losses[0] - losses[-1]) / len(losses)
        else:
            convergence_rate = 0
        
        # Determine status
        if learning_rate > 1.0:
            status = "Too high - likely diverging"
        elif learning_rate < 1e-6:
            status = "Too low - very slow convergence"
        elif is_oscillating:
            status = "Oscillating - consider reducing learning rate"
        elif is_decreasing and convergence_rate > 0.001:
            status = "Good - steady convergence"
        elif is_decreasing:
            status = "Slow but stable convergence"
        else:
            status = "Poor convergence"
        
        return {
            'status': status,
            'is_decreasing': is_decreasing,
            'is_oscillating': is_oscillating,
            'convergence_rate': convergence_rate,
            'final_loss': losses[-1]
        }


class NeuralNetworkMathematics:
    """Main neural network mathematics module."""
    
    def __init__(self):
        """Initialize neural network mathematics module."""
        self.backprop_explainer = BackpropagationExplainer()
    
    def solve_backpropagation_problem(self, problem: ParsedProblem) -> StepSolution:
        """
        Solve backpropagation problems with step-by-step explanation.
        
        Args:
            problem: Parsed backpropagation problem
            
        Returns:
            Complete step-by-step solution
        """
        try:
            # Extract network parameters from problem
            network_structure, activation_names, learning_rate = self._extract_network_from_problem(problem)
            
            # Generate sample data for demonstration
            np.random.seed(42)
            input_data = np.random.randn(network_structure[0], 1)
            target_output = np.random.randn(network_structure[-1], 1)
            
            # Perform step-by-step backpropagation
            steps = self.backprop_explainer.explain_backpropagation_step_by_step(
                network_structure, activation_names, input_data, target_output, learning_rate
            )
            
            # Add final educational summary
            final_step = SolutionStep(
                step_number=len(steps) + 1,
                operation="Neural Network Learning Summary",
                explanation="Backpropagation enables neural networks to learn by adjusting weights based on errors",
                mathematical_expression="Learning = Forward Pass + Loss Computation + Backward Pass + Weight Update",
                intermediate_result="Network weights updated to minimize prediction error"
            )
            steps.append(final_step)
            
            final_answer = f"Backpropagation completed for {len(network_structure)-1}-layer network with {activation_names} activations"
            
            return StepSolution(
                problem_id=problem.id,
                steps=steps,
                final_answer=final_answer,
                solution_method="Backpropagation step-by-step explanation",
                confidence_score=0.95,
                computation_time=0.0
            )
            
        except Exception as e:
            raise ComputationError(f"Failed to solve backpropagation problem: {str(e)}")
    
    def _extract_network_from_problem(self, problem: ParsedProblem) -> Tuple[List[int], List[str], float]:
        """Extract network structure from problem."""
        # Default network structure
        network_structure = [2, 3, 1]  # Simple 2-3-1 network
        activation_names = ['sigmoid', 'sigmoid']
        learning_rate = 0.01
        
        # Try to extract from problem text
        import re
        
        # Look for network structure patterns
        structure_patterns = [
            r'(\d+)-(\d+)-(\d+)\s+network',
            r'network.*?(\d+)-(\d+)-(\d+)',
            r'(\d+)\s*-\s*(\d+)\s*-\s*(\d+)'
        ]
        
        for pattern in structure_patterns:
            structure_match = re.search(pattern, problem.original_text.lower())
            if structure_match:
                network_structure = [int(structure_match.group(1)), 
                                   int(structure_match.group(2)), 
                                   int(structure_match.group(3))]
                break
        
        # Look for activation functions
        if 'relu' in problem.original_text.lower():
            activation_names = ['relu', 'sigmoid']
        elif 'tanh' in problem.original_text.lower():
            activation_names = ['tanh', 'sigmoid']
        
        # Look for learning rate
        lr_match = re.search(r'learning.rate.*?(\d+\.?\d*)', problem.original_text.lower())
        if lr_match:
            learning_rate = float(lr_match.group(1))
        
        return network_structure, activation_names, learning_rate
    
    def get_activation_function_analysis(self) -> Dict[str, Any]:
        """Get comprehensive analysis of activation functions."""
        return self.backprop_explainer.analyze_activation_functions()
    
    def demonstrate_learning_rate_effects(self, 
                                        network_structure: List[int] = [2, 3, 1],
                                        learning_rates: List[float] = [0.001, 0.01, 0.1, 1.0]) -> Dict[str, Any]:
        """Demonstrate effects of different learning rates."""
        return self.backprop_explainer.demonstrate_learning_rate_impact(
            network_structure, learning_rates
        )