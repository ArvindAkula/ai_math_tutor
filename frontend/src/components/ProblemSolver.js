import React, { useState } from 'react';
import {
  Typography,
  TextField,
  Button,
  Paper,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Stack,
  Grid,
  Card,
  CardContent,
  Divider,
  useTheme,
  useMediaQuery
} from '@mui/material';
import MathRenderer from './MathRenderer';
import InteractiveVisualization from './InteractiveVisualization';
import MobileInput from './MobileInput';
import { useGuestLimitations } from '../contexts/GuestLimitationContext';
import { useAuth } from '../contexts/AuthContext';

function ProblemSolver() {
  const [problem, setProblem] = useState('');
  const [solution, setSolution] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  
  // Guest limitation hooks
  const { trackUsage, showPrompt } = useGuestLimitations();
  const { isAuthenticated } = useAuth();

  const handleSolve = async () => {
    if (!problem.trim()) {
      setError('Please enter a mathematical problem');
      return;
    }

    // Track usage for guest users and check limits
    if (!isAuthenticated) {
      trackUsage('PROBLEM_SOLVER');
      
      // Check if limit is exceeded and show prompt
      const limitExceeded = showPrompt('PROBLEM_SOLVER');
      if (limitExceeded) {
        // The prompt will be shown by the context, but we can still continue
        // This allows users to see the prompt but still use the feature
      }
    }

    setLoading(true);
    setError('');
    setSolution(null);

    try {
      // For now, use mock solutions until the API is fully implemented
      // This provides working examples for the demo
      const mockSolution = generateMockSolution(problem.trim());
      
      // Simulate API delay for realistic experience
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setSolution(mockSolution);
      setLoading(false);
      
      // TODO: Replace with actual API calls once endpoints are fully implemented
      /*
      // Step 1: Parse the problem
      const parseResponse = await fetch('http://localhost:8001/parse-problem', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
          problem_text: problem.trim()
        })
      });

      if (!parseResponse.ok) {
        throw new Error(`Parse failed: ${parseResponse.status}`);
      }

      const parseResult = await parseResponse.json();

      // Step 2: Solve the parsed problem
      const solveResponse = await fetch('http://localhost:8001/solve-step-by-step', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(parseResult)
      });

      if (!solveResponse.ok) {
        throw new Error(`Solve failed: ${solveResponse.status}`);
      }

      const solveResult = await solveResponse.json();

      // Step 3: Generate visualization if applicable
      let visualizationData = null;
      try {
        const vizResponse = await fetch('http://localhost:8001/generate-visualization', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            problem: parseResult,
            viz_type: 'auto'
          })
        });

        if (vizResponse.ok) {
          const vizResult = await vizResponse.json();
          visualizationData = vizResult.plot_data;
        }
      } catch (vizError) {
        console.warn('Visualization generation failed:', vizError);
      }

      setSolution({
        ...solveResult,
        visualization_data: visualizationData
      });
      */
      
    } catch (err) {
      console.error('Problem solving error:', err);
      setError(`Failed to solve the problem: ${err.message}`);
      setLoading(false);
    }
  };

  const generateMockSolution = (problemText) => {
    const lowerProblem = problemText.toLowerCase();
    
    // Basic arithmetic (like 5+2=? or 10-3=?)
    const basicArithmeticMatch = problemText.match(/^(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*\??\s*$/);
    if (basicArithmeticMatch) {
      const num1 = parseInt(basicArithmeticMatch[1]);
      const operator = basicArithmeticMatch[2];
      const num2 = parseInt(basicArithmeticMatch[3]);
      let result, operationName, operationSymbol;
      
      switch (operator) {
        case '+':
          result = num1 + num2;
          operationName = 'Addition';
          operationSymbol = '+';
          break;
        case '-':
          result = num1 - num2;
          operationName = 'Subtraction';
          operationSymbol = '-';
          break;
        case '*':
          result = num1 * num2;
          operationName = 'Multiplication';
          operationSymbol = '\\times';
          break;
        case '/':
          result = num1 / num2;
          operationName = 'Division';
          operationSymbol = '\\div';
          break;
        default:
          result = 0;
          operationName = 'Unknown';
          operationSymbol = operator;
      }
      
      return {
        steps: [
          {
            step_number: 1,
            operation: `Identify the ${operationName.toLowerCase()} problem`,
            explanation: `This is a basic ${operationName.toLowerCase()} problem`,
            mathematical_expression: `${num1} ${operationSymbol} ${num2} = ?`,
            intermediate_result: `${operationName} of ${num1} and ${num2}`
          },
          {
            step_number: 2,
            operation: `Perform the ${operationName.toLowerCase()}`,
            explanation: `Calculate ${num1} ${operator} ${num2}`,
            mathematical_expression: `${num1} ${operationSymbol} ${num2} = ${result}`,
            intermediate_result: `Result: ${result}`
          }
        ],
        final_answer: `${result}`,
        solution_method: `Basic ${operationName}`,
        confidence_score: 1.0
      };
    }

    // Simple equations with variables (like 5+y=30, x+7=15)
    const simpleVariableMatch = problemText.match(/^(\d+)\s*([+\-])\s*([a-z])\s*=\s*(\d+)\s*$/);
    if (simpleVariableMatch) {
      const num1 = parseInt(simpleVariableMatch[1]);
      const operator = simpleVariableMatch[2];
      const variable = simpleVariableMatch[3];
      const result = parseInt(simpleVariableMatch[4]);
      
      let solution, operationName;
      if (operator === '+') {
        solution = result - num1;
        operationName = 'subtraction';
      } else {
        solution = result + num1;
        operationName = 'addition';
      }
      
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the simple equation",
            explanation: `This is a simple equation with one variable: ${variable}`,
            mathematical_expression: problemText,
            intermediate_result: `Equation: ${num1} ${operator} ${variable} = ${result}`
          },
          {
            step_number: 2,
            operation: `Isolate ${variable} using ${operationName}`,
            explanation: `${operator === '+' ? 'Subtract' : 'Add'} ${num1} ${operator === '+' ? 'from' : 'to'} both sides`,
            mathematical_expression: `${variable} = ${result} ${operator === '+' ? '-' : '+'} ${num1}`,
            intermediate_result: `${variable} = ${solution}`
          },
          {
            step_number: 3,
            operation: "Verify the solution",
            explanation: "Check by substituting back into the original equation",
            mathematical_expression: `${num1} ${operator} ${solution} = ${operator === '+' ? num1 + solution : num1 - solution} = ${result} \\checkmark`,
            intermediate_result: "Solution verified"
          }
        ],
        final_answer: `${variable} = ${solution}`,
        solution_method: "Simple Linear Equation",
        confidence_score: 1.0,
        visualization_data: generateSimpleEquationPlot(variable, solution)
      };
    }

    // Variable first equations (like y+5=30, x-3=12)
    const variableFirstMatch = problemText.match(/^([a-z])\s*([+\-])\s*(\d+)\s*=\s*(\d+)\s*$/);
    if (variableFirstMatch) {
      const variable = variableFirstMatch[1];
      const operator = variableFirstMatch[2];
      const num1 = parseInt(variableFirstMatch[3]);
      const result = parseInt(variableFirstMatch[4]);
      
      let solution;
      if (operator === '+') {
        solution = result - num1;
      } else {
        solution = result + num1;
      }
      
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the simple equation",
            explanation: `This is a simple equation with variable ${variable} first`,
            mathematical_expression: problemText,
            intermediate_result: `Equation: ${variable} ${operator} ${num1} = ${result}`
          },
          {
            step_number: 2,
            operation: `Isolate ${variable}`,
            explanation: `${operator === '+' ? 'Subtract' : 'Add'} ${num1} ${operator === '+' ? 'from' : 'to'} both sides`,
            mathematical_expression: `${variable} = ${result} ${operator === '+' ? '-' : '+'} ${num1}`,
            intermediate_result: `${variable} = ${solution}`
          },
          {
            step_number: 3,
            operation: "Verify the solution",
            explanation: "Check by substituting back into the original equation",
            mathematical_expression: `${solution} ${operator} ${num1} = ${operator === '+' ? solution + num1 : solution - num1} = ${result} \\checkmark`,
            intermediate_result: "Solution verified"
          }
        ],
        final_answer: `${variable} = ${solution}`,
        solution_method: "Simple Linear Equation",
        confidence_score: 1.0,
        visualization_data: generateSimpleEquationPlot(variable, solution)
      };
    }
    
    // Helper function to detect equation types
    const isLinearEquation = (text) => {
      // Check for single variable linear equations like ax + b = c
      const linearPattern = /^\s*\d*x\s*[+\-]\s*\d+\s*=\s*\d+\s*$/;
      const generalLinearPattern = /^\s*\d*x\s*[+\-]?\s*\d*\s*=\s*\d+\s*$/;
      return linearPattern.test(text) || generalLinearPattern.test(text) || 
             (text.includes('x') && text.includes('=') && !text.includes('x^2') && !text.includes('y'));
    };

    const isSystemOfEquations = (text) => {
      // Check for system of equations with x and y
      return (text.includes('x') && text.includes('y') && text.includes('=')) ||
             text.includes(',') && text.includes('x') && text.includes('y');
    };

    const parseLinearEquation = (text) => {
      // Extract coefficients from linear equation like "5x + y = 30" or "2x + 3 = 7"
      const match = text.match(/(\d*)x\s*([+\-])\s*(\d+)\s*=\s*(\d+)/);
      if (match) {
        const a = match[1] ? parseInt(match[1]) : 1;
        const sign = match[2];
        const b = parseInt(match[3]);
        const c = parseInt(match[4]);
        return { a, b: sign === '+' ? b : -b, c };
      }
      return null;
    };

    // Linear equations (enhanced pattern matching)
    if (isLinearEquation(problemText) && !isSystemOfEquations(problemText)) {
      const coeffs = parseLinearEquation(problemText);
      if (coeffs) {
        const { a, b, c } = coeffs;
        const solution = (c - b) / a;
        return {
          steps: [
            {
              step_number: 1,
              operation: "Identify the linear equation",
              explanation: "This is a linear equation in the form ax + b = c",
              mathematical_expression: problemText,
              intermediate_result: `a = ${a}, b = ${b}, c = ${c}`
            },
            {
              step_number: 2,
              operation: `${b >= 0 ? 'Subtract' : 'Add'} ${Math.abs(b)} ${b >= 0 ? 'from' : 'to'} both sides`,
              explanation: `Isolate the term with x by ${b >= 0 ? 'subtracting' : 'adding'} ${Math.abs(b)} ${b >= 0 ? 'from' : 'to'} both sides`,
              mathematical_expression: `${a}x ${b >= 0 ? '+' : ''} ${b} ${b >= 0 ? '- ' + b : '+ ' + Math.abs(b)} = ${c} ${b >= 0 ? '- ' + b : '+ ' + Math.abs(b)}`,
              intermediate_result: `${a}x = ${c - b}`
            },
            {
              step_number: 3,
              operation: `Divide both sides by ${a}`,
              explanation: `Solve for x by dividing both sides by the coefficient of x`,
              mathematical_expression: `\\frac{${a}x}{${a}} = \\frac{${c - b}}{${a}}`,
              intermediate_result: `x = ${solution}`
            },
            {
              step_number: 4,
              operation: "Verify the solution",
              explanation: "Check our answer by substituting back into the original equation",
              mathematical_expression: `${a}(${solution}) ${b >= 0 ? '+' : ''} ${b} = ${a * solution + b} = ${c} \\checkmark`,
              intermediate_result: "Solution verified"
            }
          ],
          final_answer: `x = ${solution}`,
          solution_method: "Linear Equation Solving",
          confidence_score: 1.0,
          visualization_data: generateDynamicLinearPlot(a, b, c, solution)
        };
      }
    }
    
    // System of equations (enhanced)
    else if (isSystemOfEquations(problemText)) {
      // Handle single equation with x and y (like "5x+y=30")
      if (!problemText.includes(',') && problemText.includes('x') && problemText.includes('y')) {
        // Parse equation like "5x+y=30"
        const match = problemText.match(/(\d*)x\s*([+\-])\s*(\d*)y\s*=\s*(\d+)/);
        if (match) {
          const a = match[1] ? parseInt(match[1]) : 1;
          const sign = match[2];
          const b = match[3] ? parseInt(match[3]) : 1;
          const c = parseInt(match[4]);
          const bCoeff = sign === '+' ? b : -b;
          
          return {
            steps: [
              {
                step_number: 1,
                operation: "Identify the linear equation with two variables",
                explanation: "This is a linear equation in two variables: ax + by = c",
                mathematical_expression: problemText,
                intermediate_result: `a = ${a}, b = ${bCoeff}, c = ${c}`
              },
              {
                step_number: 2,
                operation: "Solve for y in terms of x",
                explanation: "Rearrange the equation to express y as a function of x",
                mathematical_expression: `${bCoeff}y = ${c} - ${a}x`,
                intermediate_result: `y = \\frac{${c} - ${a}x}{${bCoeff}}`
              },
              {
                step_number: 3,
                operation: "Simplify the equation",
                explanation: "Express y in slope-intercept form",
                mathematical_expression: `y = ${-a/bCoeff}x + ${c/bCoeff}`,
                intermediate_result: `Linear function: y = ${-a/bCoeff}x + ${c/bCoeff}`
              },
              {
                step_number: 4,
                operation: "Identify key characteristics",
                explanation: "This represents a line with infinite solutions (x, y) pairs",
                mathematical_expression: `\\text{Slope: } m = ${-a/bCoeff}, \\text{ y-intercept: } b = ${c/bCoeff}`,
                intermediate_result: "Line equation determined"
              }
            ],
            final_answer: `y = ${-a/bCoeff}x + ${c/bCoeff}`,
            solution_method: "Linear Equation in Two Variables",
            confidence_score: 1.0,
            visualization_data: generateDynamicLinearPlot(a, bCoeff, c, null, true)
          };
        }
      }
      
      // Handle system of two equations
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the system of equations",
            explanation: "We have two linear equations with two unknowns",
            mathematical_expression: "\\begin{cases} 2x + y = 5 \\\\ x - y = 1 \\end{cases}",
            intermediate_result: "System identified"
          },
          {
            step_number: 2,
            operation: "Use elimination method",
            explanation: "Add the equations to eliminate y",
            mathematical_expression: "(2x + y) + (x - y) = 5 + 1",
            intermediate_result: "3x = 6"
          },
          {
            step_number: 3,
            operation: "Solve for x",
            explanation: "Divide both sides by 3",
            mathematical_expression: "x = \\frac{6}{3} = 2",
            intermediate_result: "x = 2"
          },
          {
            step_number: 4,
            operation: "Substitute to find y",
            explanation: "Use x = 2 in the first equation",
            mathematical_expression: "2(2) + y = 5 \\Rightarrow 4 + y = 5 \\Rightarrow y = 1",
            intermediate_result: "y = 1"
          }
        ],
        final_answer: "x = 2, y = 1",
        solution_method: "Elimination Method",
        confidence_score: 0.99,
        visualization_data: generateSystemPlot()
      };
    }
    
    // Quadratic equations
    else if (lowerProblem.includes('x^2') || lowerProblem.includes('quadratic')) {
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the quadratic equation",
            explanation: "This is a quadratic equation in standard form ax² + bx + c = 0",
            mathematical_expression: "x^2 - 5x + 6 = 0",
            intermediate_result: "a = 1, b = -5, c = 6"
          },
          {
            step_number: 2,
            operation: "Try factoring first",
            explanation: "Look for two numbers that multiply to 6 and add to -5",
            mathematical_expression: "x^2 - 5x + 6 = (x - 2)(x - 3) = 0",
            intermediate_result: "Factors: -2 and -3"
          },
          {
            step_number: 3,
            operation: "Apply zero product property",
            explanation: "If the product of factors equals zero, then at least one factor must be zero",
            mathematical_expression: "x - 2 = 0 \\text{ or } x - 3 = 0",
            intermediate_result: "Two separate equations"
          },
          {
            step_number: 4,
            operation: "Solve each equation",
            explanation: "Solve each linear equation separately",
            mathematical_expression: "x = 2 \\text{ or } x = 3",
            intermediate_result: "Two solutions found"
          }
        ],
        final_answer: "x = 2 \\text{ or } x = 3",
        solution_method: "Factoring Method",
        confidence_score: 0.98,
        visualization_data: generateQuadraticPlot()
      };
    }
    
    // Derivatives
    else if (lowerProblem.includes('d/dx') || lowerProblem.includes('derivative')) {
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the function to differentiate",
            explanation: "We need to find the derivative of the given function",
            mathematical_expression: "f(x) = x^2 + 3x",
            intermediate_result: "Function: polynomial of degree 2"
          },
          {
            step_number: 2,
            operation: "Apply power rule to x² term",
            explanation: "Use the power rule: d/dx(x^n) = nx^(n-1)",
            mathematical_expression: "\\frac{d}{dx}(x^2) = 2x^{2-1} = 2x",
            intermediate_result: "First term derivative: 2x"
          },
          {
            step_number: 3,
            operation: "Apply power rule to 3x term",
            explanation: "The derivative of ax is simply a",
            mathematical_expression: "\\frac{d}{dx}(3x) = 3x^{1-1} = 3",
            intermediate_result: "Second term derivative: 3"
          },
          {
            step_number: 4,
            operation: "Combine the results",
            explanation: "Add the derivatives of individual terms",
            mathematical_expression: "f'(x) = 2x + 3",
            intermediate_result: "Final derivative obtained"
          }
        ],
        final_answer: "f'(x) = 2x + 3",
        solution_method: "Power Rule Differentiation",
        confidence_score: 0.99,
        visualization_data: generateDerivativePlot()
      };
    }
    
    // Integrals
    else if (lowerProblem.includes('∫') || lowerProblem.includes('integral')) {
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the function to integrate",
            explanation: "We need to find the antiderivative of the given function",
            mathematical_expression: "\\int (2x + 1) dx",
            intermediate_result: "Integrand: 2x + 1"
          },
          {
            step_number: 2,
            operation: "Apply power rule for integration",
            explanation: "∫x^n dx = x^(n+1)/(n+1) + C",
            mathematical_expression: "\\int 2x dx = 2 \\cdot \\frac{x^{1+1}}{1+1} = \\frac{2x^2}{2} = x^2",
            intermediate_result: "First term: x²"
          },
          {
            step_number: 3,
            operation: "Integrate the constant term",
            explanation: "The integral of a constant a is ax",
            mathematical_expression: "\\int 1 dx = x",
            intermediate_result: "Second term: x"
          },
          {
            step_number: 4,
            operation: "Combine results and add constant",
            explanation: "Add the constant of integration C",
            mathematical_expression: "\\int (2x + 1) dx = x^2 + x + C",
            intermediate_result: "Complete antiderivative"
          }
        ],
        final_answer: "x^2 + x + C",
        solution_method: "Power Rule Integration",
        confidence_score: 0.99,
        visualization_data: generateIntegralPlot()
      };
    }
    
    // System of equations
    else if (lowerProblem.includes('system') || (lowerProblem.includes('x') && lowerProblem.includes('y') && lowerProblem.includes('='))) {
      return {
        steps: [
          {
            step_number: 1,
            operation: "Identify the system of equations",
            explanation: "We have two linear equations with two unknowns",
            mathematical_expression: "\\begin{cases} 2x + y = 5 \\\\ x - y = 1 \\end{cases}",
            intermediate_result: "System identified"
          },
          {
            step_number: 2,
            operation: "Use elimination method",
            explanation: "Add the equations to eliminate y",
            mathematical_expression: "(2x + y) + (x - y) = 5 + 1",
            intermediate_result: "3x = 6"
          },
          {
            step_number: 3,
            operation: "Solve for x",
            explanation: "Divide both sides by 3",
            mathematical_expression: "x = \\frac{6}{3} = 2",
            intermediate_result: "x = 2"
          },
          {
            step_number: 4,
            operation: "Substitute to find y",
            explanation: "Use x = 2 in the first equation",
            mathematical_expression: "2(2) + y = 5 \\Rightarrow 4 + y = 5 \\Rightarrow y = 1",
            intermediate_result: "y = 1"
          }
        ],
        final_answer: "x = 2, y = 1",
        solution_method: "Elimination Method",
        confidence_score: 0.99,
        visualization_data: generateSystemPlot()
      };
    }
    
    // Default case
    else {
      return {
        steps: [
          {
            step_number: 1,
            operation: "Analyze the problem",
            explanation: "Examining the mathematical expression to determine the best solution approach",
            mathematical_expression: problemText,
            intermediate_result: "Problem structure analyzed"
          },
          {
            step_number: 2,
            operation: "Apply mathematical principles",
            explanation: "Using appropriate mathematical rules and techniques for this type of problem",
            mathematical_expression: "\\text{Solution approach determined}",
            intermediate_result: "Method selected"
          }
        ],
        final_answer: "\\text{Try one of the example problems for a complete solution!}",
        solution_method: "General Problem Analysis",
        confidence_score: 0.85
      };
    }
  };

  const generateQuadraticPlot = () => {
    const x = [];
    const y = [];
    for (let i = -1; i <= 6; i += 0.1) {
      x.push(i);
      y.push(i * i - 5 * i + 6);
    }
    return [{
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      name: 'f(x) = x² - 5x + 6',
      line: { color: '#1976d2', width: 3 }
    }, {
      x: [2, 3],
      y: [0, 0],
      type: 'scatter',
      mode: 'markers',
      name: 'Roots',
      marker: { color: 'red', size: 10 }
    }];
  };

  const generateLinearPlot = () => {
    const x = [];
    const y = [];
    for (let i = -1; i <= 5; i += 0.1) {
      x.push(i);
      y.push(2 * i + 3);
    }
    return [{
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      name: 'y = 2x + 3',
      line: { color: '#1976d2', width: 3 }
    }, {
      x: [2],
      y: [7],
      type: 'scatter',
      mode: 'markers',
      name: 'Solution (2, 7)',
      marker: { color: 'red', size: 12 }
    }, {
      x: [-1, 5],
      y: [7, 7],
      type: 'scatter',
      mode: 'lines',
      name: 'y = 7',
      line: { color: 'green', width: 2, dash: 'dash' }
    }];
  };

  const generateDerivativePlot = () => {
    const x = [];
    const y_original = [];
    const y_derivative = [];
    for (let i = -5; i <= 5; i += 0.1) {
      x.push(i);
      y_original.push(i * i + 3 * i);
      y_derivative.push(2 * i + 3);
    }
    return [{
      x: x,
      y: y_original,
      type: 'scatter',
      mode: 'lines',
      name: 'f(x) = x² + 3x',
      line: { color: '#1976d2', width: 2 }
    }, {
      x: x,
      y: y_derivative,
      type: 'scatter',
      mode: 'lines',
      name: "f'(x) = 2x + 3",
      line: { color: '#dc004e', width: 2, dash: 'dash' }
    }];
  };

  const generateIntegralPlot = () => {
    const x = [];
    const y_integrand = [];
    const y_antiderivative = [];
    for (let i = -3; i <= 3; i += 0.1) {
      x.push(i);
      y_integrand.push(2 * i + 1);
      y_antiderivative.push(i * i + i); // Without constant C for visualization
    }
    return [{
      x: x,
      y: y_integrand,
      type: 'scatter',
      mode: 'lines',
      name: 'f(x) = 2x + 1',
      line: { color: '#1976d2', width: 2 }
    }, {
      x: x,
      y: y_antiderivative,
      type: 'scatter',
      mode: 'lines',
      name: 'F(x) = x² + x',
      line: { color: '#dc004e', width: 2, dash: 'dash' }
    }];
  };

  const generateDynamicLinearPlot = (a, b, c, solution, isTwoVariable = false) => {
    const x = [];
    const y = [];
    
    if (isTwoVariable) {
      // For equations like 5x + y = 30, plot y = (c - ax) / b
      for (let i = -2; i <= 10; i += 0.1) {
        x.push(i);
        y.push((c - a * i) / b);
      }
      return [{
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: `${a}x + ${b}y = ${c}`,
        line: { color: '#1976d2', width: 3 }
      }];
    } else {
      // For single variable equations like 2x + 3 = 7
      for (let i = -1; i <= 5; i += 0.1) {
        x.push(i);
        y.push(a * i + b);
      }
      return [{
        x: x,
        y: y,
        type: 'scatter',
        mode: 'lines',
        name: `y = ${a}x + ${b}`,
        line: { color: '#1976d2', width: 3 }
      }, {
        x: [solution],
        y: [c],
        type: 'scatter',
        mode: 'markers',
        name: `Solution (${solution}, ${c})`,
        marker: { color: 'red', size: 12 }
      }, {
        x: [-1, 5],
        y: [c, c],
        type: 'scatter',
        mode: 'lines',
        name: `y = ${c}`,
        line: { color: 'green', width: 2, dash: 'dash' }
      }];
    }
  };

  const generateSimpleEquationPlot = (variable, solution) => {
    // For simple equations like 5+y=30, create a basic visualization
    const x = [];
    const y = [];
    for (let i = solution - 10; i <= solution + 10; i += 0.5) {
      x.push(i);
      y.push(i); // Simple y = x line for reference
    }
    return [{
      x: x,
      y: y,
      type: 'scatter',
      mode: 'lines',
      name: `${variable} values`,
      line: { color: '#1976d2', width: 2, dash: 'dot' }
    }, {
      x: [solution],
      y: [solution],
      type: 'scatter',
      mode: 'markers',
      name: `Solution: ${variable} = ${solution}`,
      marker: { color: 'red', size: 15, symbol: 'star' }
    }];
  };

  const generateSystemPlot = () => {
    const x = [];
    const y1 = []; // 2x + y = 5 => y = 5 - 2x
    const y2 = []; // x - y = 1 => y = x - 1
    for (let i = -1; i <= 4; i += 0.1) {
      x.push(i);
      y1.push(5 - 2 * i);
      y2.push(i - 1);
    }
    return [{
      x: x,
      y: y1,
      type: 'scatter',
      mode: 'lines',
      name: '2x + y = 5',
      line: { color: '#1976d2', width: 2 }
    }, {
      x: x,
      y: y2,
      type: 'scatter',
      mode: 'lines',
      name: 'x - y = 1',
      line: { color: '#dc004e', width: 2 }
    }, {
      x: [2],
      y: [1],
      type: 'scatter',
      mode: 'markers',
      name: 'Solution (2, 1)',
      marker: { color: 'green', size: 12 }
    }];
  };

  const exampleProblems = [
    "2x + 3 = 7",
    "x^2 - 5x + 6 = 0",
    "d/dx(x^2 + 3x)",
    "∫(2x + 1)dx",
    "2x + y = 5, x - y = 1"
  ];

  return (
    <Box>
      <Typography variant="h2" component="h1" gutterBottom>
        Problem Solver
      </Typography>
      
      <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
        Enter any mathematical problem and get step-by-step solutions with detailed explanations.
      </Typography>

      <Paper sx={{ p: 3, mb: 3 }}>
        {isMobile ? (
          <MobileInput
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            onSubmit={handleSolve}
            placeholder="e.g., 2x + 3 = 7 or d/dx(x^2 + 3x)"
          />
        ) : (
          <TextField
            fullWidth
            multiline
            rows={3}
            label="Enter your mathematical problem"
            placeholder="e.g., 2x + 3 = 7 or d/dx(x^2 + 3x)"
            value={problem}
            onChange={(e) => setProblem(e.target.value)}
            sx={{ mb: 2 }}
          />
        )}
        
        <Button
          variant="contained"
          onClick={handleSolve}
          disabled={loading}
          size="large"
          fullWidth={isMobile}
          sx={{ mb: 2 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Solve Problem'}
        </Button>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
      </Paper>

      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Try these examples:
        </Typography>
        <Stack direction="row" spacing={1} flexWrap="wrap">
          {exampleProblems.map((example, index) => (
            <Chip
              key={index}
              label={example}
              onClick={() => setProblem(example)}
              variant="outlined"
              sx={{ mb: 1 }}
            />
          ))}
        </Stack>
      </Box>

      {solution && (
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Solution
          </Typography>
          
          <Typography variant="h6" gutterBottom>
            Method: {solution.solution_method}
          </Typography>
          
          <Grid container spacing={2} sx={{ mb: 3 }}>
            <Grid item xs={12} md={solution.visualization_data ? 6 : 12}>
              {solution.steps.map((step, index) => (
                <Card key={index} sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h6" color="primary" gutterBottom>
                      Step {step.step_number}: {step.operation}
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 2 }}>
                      {step.explanation}
                    </Typography>
                    
                    {step.mathematical_expression && (
                      <Box sx={{ my: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
                        <MathRenderer 
                          math={step.mathematical_expression} 
                          block={true}
                        />
                      </Box>
                    )}
                    
                    <Typography variant="body2" color="text.secondary">
                      <strong>Result:</strong> {step.intermediate_result}
                    </Typography>
                  </CardContent>
                </Card>
              ))}
            </Grid>
            
            {solution.visualization_data && (
              <Grid item xs={12} md={6}>
                <InteractiveVisualization 
                  plotData={solution.visualization_data}
                  title="Problem Visualization"
                  interactive={false}
                />
              </Grid>
            )}
          </Grid>
          
          <Alert severity="success" sx={{ mb: 2 }}>
            <Typography variant="h6" component="div">
              Final Answer: <MathRenderer math={solution.final_answer} />
            </Typography>
          </Alert>
          
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Confidence: {(solution.confidence_score * 100).toFixed(1)}%
          </Typography>
        </Paper>
      )}

      <Box sx={{ mt: 4, p: 2, bgcolor: 'info.light', borderRadius: 1 }}>
        <Typography variant="body2" color="info.contrastText">
          <strong>Note:</strong> This is a preview interface. Full mathematical computation 
          capabilities will be implemented in the upcoming development tasks.
        </Typography>
      </Box>
    </Box>
  );
}

export default ProblemSolver;