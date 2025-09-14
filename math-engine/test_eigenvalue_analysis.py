"""
Tests for eigenvalue and eigenvector analysis tools with AI/ML context.
Tests numerical accuracy and educational effectiveness.
"""

import unittest
import numpy as np
from sympy import Matrix, I, sqrt, Rational
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

from aiml_mathematics import AIMLMathematics, EigenAnalysisResult
from models import ParsedProblem, MathDomain, DifficultyLevel


class TestEigenvalueAnalysis(unittest.TestCase):
    """Test eigenvalue and eigenvector analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.aiml_math = AIMLMathematics()
    
    def test_simple_2x2_eigenanalysis(self):
        """Test eigenanalysis of a simple 2x2 matrix."""
        # Test matrix: [[1, 2], [3, 4]]
        matrix = Matrix([[1, 2], [3, 4]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # Check that we have the expected number of steps
        self.assertGreaterEqual(len(steps), 5)
        
        # Check that eigenvalues are computed correctly
        # For [[1,2],[3,4]], eigenvalues should be (5±√33)/2
        expected_eigenvals = [(5 + sqrt(33))/2, (5 - sqrt(33))/2]
        
        # Convert to float for comparison
        computed_eigenvals = [complex(ev.evalf()) for ev in result.eigenvalues]
        expected_eigenvals_float = [complex(ev.evalf()) for ev in expected_eigenvals]
        
        # Sort both lists for comparison
        computed_eigenvals.sort(key=lambda x: x.real)
        expected_eigenvals_float.sort(key=lambda x: x.real)
        
        for computed, expected in zip(computed_eigenvals, expected_eigenvals_float):
            self.assertAlmostEqual(computed.real, expected.real, places=10)
            self.assertAlmostEqual(computed.imag, expected.imag, places=10)
        
        # Check that eigenvectors are computed
        self.assertGreater(len(result.eigenvectors), 0)
        
        # Check ML interpretation
        ml_interp = result.get_ml_interpretation()
        self.assertIn('stability', ml_interp)
        self.assertIn('dynamics', ml_interp)
        self.assertIn('diagonalization', ml_interp)
    
    def test_diagonal_matrix_eigenanalysis(self):
        """Test eigenanalysis of a diagonal matrix."""
        # Diagonal matrix: [[3, 0], [0, -2]]
        matrix = Matrix([[3, 0], [0, -2]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # For diagonal matrix, eigenvalues should be the diagonal elements
        expected_eigenvals = [3, -2]
        computed_eigenvals = [complex(ev.evalf()) for ev in result.eigenvalues]
        
        # Sort for comparison
        expected_eigenvals.sort()
        computed_eigenvals.sort(key=lambda x: x.real)
        
        for computed, expected in zip(computed_eigenvals, expected_eigenvals):
            self.assertAlmostEqual(computed.real, expected, places=10)
            self.assertAlmostEqual(computed.imag, 0, places=10)
        
        # Check that matrix is diagonalizable
        self.assertTrue(result.is_diagonalizable())
    
    def test_complex_eigenvalues(self):
        """Test matrix with complex eigenvalues."""
        # Rotation matrix: [[0, -1], [1, 0]] (90-degree rotation)
        matrix = Matrix([[0, -1], [1, 0]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # Eigenvalues should be ±i
        computed_eigenvals = [complex(ev.evalf()) for ev in result.eigenvalues]
        
        # Check that we have complex eigenvalues
        has_complex = any(abs(ev.imag) > 1e-10 for ev in computed_eigenvals)
        self.assertTrue(has_complex)
        
        # Check ML interpretation mentions oscillatory behavior
        ml_interp = result.get_ml_interpretation()
        self.assertIn('oscillatory', ml_interp['dynamics'].lower())
    
    def test_repeated_eigenvalues(self):
        """Test matrix with repeated eigenvalues."""
        # Matrix with repeated eigenvalue: [[2, 1], [0, 2]]
        matrix = Matrix([[2, 1], [0, 2]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # Both eigenvalues should be 2
        computed_eigenvals = [complex(ev.evalf()) for ev in result.eigenvalues]
        
        for ev in computed_eigenvals:
            self.assertAlmostEqual(ev.real, 2, places=10)
            self.assertAlmostEqual(ev.imag, 0, places=10)
        
        # Check multiplicities
        self.assertEqual(len(result.algebraic_multiplicities), 1)  # One unique eigenvalue
        self.assertEqual(result.algebraic_multiplicities[0], 2)    # With multiplicity 2
    
    def test_3x3_matrix_eigenanalysis(self):
        """Test eigenanalysis of a 3x3 matrix."""
        # 3x3 matrix: [[1, 0, 0], [0, 2, 1], [0, 0, 2]]
        matrix = Matrix([[1, 0, 0], [0, 2, 1], [0, 0, 2]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # Should have eigenvalues 1, 2, 2
        computed_eigenvals = [complex(ev.evalf()) for ev in result.eigenvalues]
        computed_eigenvals.sort(key=lambda x: x.real)
        
        expected = [1, 2, 2]
        for computed, expected_val in zip(computed_eigenvals, expected):
            self.assertAlmostEqual(computed.real, expected_val, places=10)
            self.assertAlmostEqual(computed.imag, 0, places=10)
    
    def test_eigenspace_visualization_2x2(self):
        """Test eigenspace visualization for 2x2 matrices."""
        matrix = Matrix([[3, 1], [0, 2]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        viz_data = self.aiml_math.generate_eigenspace_visualization(result)
        
        # Check visualization data structure
        self.assertIn('matrix', viz_data)
        self.assertIn('eigenvalues', viz_data)
        self.assertIn('eigenvectors', viz_data)
        self.assertIn('eigenlines', viz_data)
        self.assertIn('transformation_demo', viz_data)
        
        # Check that eigenlines are generated
        self.assertGreater(len(viz_data['eigenlines']), 0)
        
        # Check transformation demo
        self.assertIn('original_circle', viz_data['transformation_demo'])
        self.assertIn('transformed_shape', viz_data['transformation_demo'])
    
    def test_eigenspace_visualization_3x3_error(self):
        """Test that 3x3 matrices return appropriate error for visualization."""
        matrix = Matrix([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        viz_data = self.aiml_math.generate_eigenspace_visualization(result)
        
        # Should return error for non-2x2 matrices
        self.assertIn('error', viz_data)
    
    def test_ml_applications_explanations(self):
        """Test AI/ML context explanations."""
        matrix = Matrix([[2, 1], [1, 2]])
        
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        explanations = self.aiml_math.explain_ml_applications(result)
        
        # Check that all expected ML applications are covered
        expected_keys = ['PCA', 'Stability', 'Spectral_Methods', 'Optimization', 'Dimensionality_Reduction']
        
        for key in expected_keys:
            self.assertIn(key, explanations)
            self.assertIsInstance(explanations[key], str)
            self.assertGreater(len(explanations[key].strip()), 50)  # Non-trivial explanation
    
    def test_stability_analysis(self):
        """Test stability analysis for different eigenvalue magnitudes."""
        # Stable matrix (eigenvalues < 1)
        stable_matrix = Matrix([[Rational(1,2), 0], [0, Rational(1,3)]])
        steps, stable_result = self.aiml_math.analyze_eigenvalues_step_by_step(stable_matrix)
        stable_interp = stable_result.get_ml_interpretation()
        self.assertIn('stable', stable_interp['stability'].lower())
        
        # Unstable matrix (eigenvalues > 1)
        unstable_matrix = Matrix([[2, 0], [0, 3]])
        steps, unstable_result = self.aiml_math.analyze_eigenvalues_step_by_step(unstable_matrix)
        unstable_interp = unstable_result.get_ml_interpretation()
        self.assertIn('unstable', unstable_interp['stability'].lower())
    
    def test_solve_eigenvalue_problem_integration(self):
        """Test integration with problem solving framework."""
        # Create a mock problem
        problem = ParsedProblem(
            id="test_eigen_1",
            original_text="Find eigenvalues and eigenvectors of [[1,2],[3,4]]",
            domain=MathDomain.LINEAR_ALGEBRA,
            problem_type="eigenvalue_analysis",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=[],
            expressions=["[[1,2],[3,4]]"],
            metadata={}
        )
        
        solution = self.aiml_math.solve_eigenvalue_problem(problem)
        
        # Check solution structure
        self.assertIsNotNone(solution.steps)
        self.assertGreater(len(solution.steps), 5)
        self.assertIn("eigenvalue", solution.solution_method.lower())
        self.assertIn("ml", solution.solution_method.lower())
        
        # Check that ML context is included
        ml_step_found = False
        for step in solution.steps:
            if "ml" in step.operation.lower() or "machine learning" in step.operation.lower():
                ml_step_found = True
                break
        self.assertTrue(ml_step_found)
    
    def test_answer_validation(self):
        """Test eigenvalue answer validation."""
        matrix = Matrix([[1, 2], [3, 4]])
        steps, result = self.aiml_math.analyze_eigenvalues_step_by_step(matrix)
        
        # Create mock problem
        problem = ParsedProblem(
            id="test_validation",
            original_text="Find eigenvalues of [[1,2],[3,4]]",
            domain=MathDomain.LINEAR_ALGEBRA,
            problem_type="eigenvalue_analysis",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=[],
            expressions=["[[1,2],[3,4]]"],
            metadata={}
        )
        
        # Test correct answer
        correct_eigenvals = result.eigenvalues
        correct_answer = f"{correct_eigenvals[0]}, {correct_eigenvals[1]}"
        
        validation = self.aiml_math.validate_eigenvalue_answer(problem, correct_answer, result)
        self.assertTrue(validation.is_correct)
        self.assertEqual(validation.partial_credit, 1.0)
        
        # Test incorrect answer
        wrong_answer = "1, 2"
        validation_wrong = self.aiml_math.validate_eigenvalue_answer(problem, wrong_answer, result)
        self.assertFalse(validation_wrong.is_correct)
        self.assertLess(validation_wrong.partial_credit, 1.0)
    
    def test_matrix_extraction_from_problem(self):
        """Test matrix extraction from different problem formats."""
        # Test list format
        problem1 = ParsedProblem(
            id="test_extract_1",
            original_text="Find eigenvalues of [[1,2],[3,4]]",
            domain=MathDomain.LINEAR_ALGEBRA,
            problem_type="eigenvalue_analysis",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=[],
            expressions=["[[1,2],[3,4]]"],
            metadata={}
        )
        
        matrix1 = self.aiml_math._extract_matrix_from_problem(problem1)
        expected_matrix1 = Matrix([[1, 2], [3, 4]])
        self.assertEqual(matrix1, expected_matrix1)
        
        # Test when no matrix found (should return default)
        problem2 = ParsedProblem(
            id="test_extract_2",
            original_text="Some text without matrix",
            domain=MathDomain.LINEAR_ALGEBRA,
            problem_type="eigenvalue_analysis",
            difficulty=DifficultyLevel.INTERMEDIATE,
            variables=[],
            expressions=[],
            metadata={}
        )
        
        matrix2 = self.aiml_math._extract_matrix_from_problem(problem2)
        # Should return default 2x2 matrix
        self.assertEqual(matrix2.rows, 2)
        self.assertEqual(matrix2.cols, 2)
    
    def test_eigenvalue_parsing_from_answer(self):
        """Test parsing eigenvalues from user text answers."""
        # Test real eigenvalues
        answer1 = "λ₁ = 3, λ₂ = -1"
        parsed1 = self.aiml_math._parse_eigenvalues_from_answer(answer1)
        expected1 = [3+0j, -1+0j]
        
        self.assertEqual(len(parsed1), 2)
        for parsed, expected in zip(sorted(parsed1, key=lambda x: x.real), sorted(expected1, key=lambda x: x.real)):
            self.assertAlmostEqual(parsed.real, expected.real, places=6)
            self.assertAlmostEqual(parsed.imag, expected.imag, places=6)
        
        # Test complex eigenvalues
        answer2 = "1+2i, 1-2i"
        parsed2 = self.aiml_math._parse_eigenvalues_from_answer(answer2)
        expected2 = [1+2j, 1-2j]
        
        self.assertEqual(len(parsed2), 2)
        # Sort by real part, then imaginary part
        parsed2_sorted = sorted(parsed2, key=lambda x: (x.real, x.imag))
        expected2_sorted = sorted(expected2, key=lambda x: (x.real, x.imag))
        
        for parsed, expected in zip(parsed2_sorted, expected2_sorted):
            self.assertAlmostEqual(parsed.real, expected.real, places=6)
            self.assertAlmostEqual(parsed.imag, expected.imag, places=6)
    
    def test_eigenvalue_set_comparison(self):
        """Test eigenvalue set comparison with tolerance."""
        set1 = [1+0j, 2+0j, 3+0j]
        set2 = [1.0000001+0j, 2.0000001+0j, 3.0000001+0j]
        
        # Should be equal within tolerance
        self.assertTrue(self.aiml_math._compare_eigenvalue_sets(set1, set2, 1e-6))
        
        # Should not be equal with stricter tolerance
        self.assertFalse(self.aiml_math._compare_eigenvalue_sets(set1, set2, 1e-8))
        
        # Different lengths should not be equal
        set3 = [1+0j, 2+0j]
        self.assertFalse(self.aiml_math._compare_eigenvalue_sets(set1, set3, 1e-6))
    
    def test_partial_credit_calculation(self):
        """Test partial credit calculation for eigenvalue problems."""
        correct_eigenvals = [1+0j, 2+0j, 3+0j]
        
        # All correct
        user_answer1 = [1+0j, 2+0j, 3+0j]
        credit1 = self.aiml_math._calculate_eigenvalue_partial_credit(user_answer1, correct_eigenvals)
        self.assertEqual(credit1, 1.0)
        
        # Partially correct
        user_answer2 = [1+0j, 2+0j, 999+0j]  # 2 out of 3 correct
        credit2 = self.aiml_math._calculate_eigenvalue_partial_credit(user_answer2, correct_eigenvals)
        self.assertAlmostEqual(credit2, 2.0/3.0, places=6)
        
        # None correct
        user_answer3 = [999+0j, 888+0j, 777+0j]
        credit3 = self.aiml_math._calculate_eigenvalue_partial_credit(user_answer3, correct_eigenvals)
        self.assertEqual(credit3, 0.0)
        
        # Empty answer
        user_answer4 = []
        credit4 = self.aiml_math._calculate_eigenvalue_partial_credit(user_answer4, correct_eigenvals)
        self.assertEqual(credit4, 0.0)


class TestEigenAnalysisResult(unittest.TestCase):
    """Test EigenAnalysisResult class functionality."""
    
    def test_diagonalizability_check(self):
        """Test diagonalizability checking."""
        matrix = Matrix([[1, 2], [3, 4]])
        eigenvals = [(5 + sqrt(17))/2, (5 - sqrt(17))/2]
        eigenvecs = [Matrix([1, (1 + sqrt(17))/4]), Matrix([1, (1 - sqrt(17))/4])]
        
        result = EigenAnalysisResult(
            matrix=matrix,
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            characteristic_polynomial=(5 - sqrt(17))/2,  # Simplified for test
            geometric_multiplicities=[1, 1],
            algebraic_multiplicities=[1, 1]
        )
        
        # Should be diagonalizable (geometric multiplicities sum to matrix size)
        self.assertTrue(result.is_diagonalizable())
    
    def test_dominant_eigenvalue(self):
        """Test dominant eigenvalue identification."""
        matrix = Matrix([[3, 1], [0, 2]])
        eigenvals = [3, 2]
        
        result = EigenAnalysisResult(
            matrix=matrix,
            eigenvalues=eigenvals,
            eigenvectors=[],
            characteristic_polynomial=None,
            geometric_multiplicities=[1, 1],
            algebraic_multiplicities=[1, 1]
        )
        
        dominant = result.get_dominant_eigenvalue()
        self.assertEqual(dominant, 3)
    
    def test_ml_interpretation_stability(self):
        """Test ML interpretation for different stability scenarios."""
        # Stable system
        stable_matrix = Matrix([[0.5, 0], [0, 0.3]])
        stable_result = EigenAnalysisResult(
            matrix=stable_matrix,
            eigenvalues=[0.5, 0.3],
            eigenvectors=[],
            characteristic_polynomial=None,
            geometric_multiplicities=[1, 1],
            algebraic_multiplicities=[1, 1]
        )
        
        stable_interp = stable_result.get_ml_interpretation()
        self.assertIn('stable', stable_interp['stability'].lower())
        
        # Unstable system
        unstable_matrix = Matrix([[2, 0], [0, 1.5]])
        unstable_result = EigenAnalysisResult(
            matrix=unstable_matrix,
            eigenvalues=[2, 1.5],
            eigenvectors=[],
            characteristic_polynomial=None,
            geometric_multiplicities=[1, 1],
            algebraic_multiplicities=[1, 1]
        )
        
        unstable_interp = unstable_result.get_ml_interpretation()
        self.assertIn('unstable', unstable_interp['stability'].lower())


if __name__ == '__main__':
    unittest.main()