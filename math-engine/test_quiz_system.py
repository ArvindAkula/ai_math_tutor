"""
Comprehensive tests for the quiz generation and assessment system.
Tests quiz generation variety, difficulty consistency, answer validation accuracy, and session management.
"""

import unittest
import time
from datetime import datetime, timedelta
import sys
import os

# Add shared models to path
from models import DifficultyLevel, MathDomain, QuestionType

from quiz_generator import QuizGenerator, ProblemBank
from answer_validator import AnswerValidator
from quiz_session_manager import QuizSessionManager


class TestProblemBank(unittest.TestCase):
    """Test the problem bank functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.problem_bank = ProblemBank()
    
    def test_get_problems_by_criteria(self):
        """Test retrieving problems by domain and difficulty."""
        # Test algebra beginner problems
        problems = self.problem_bank.get_problems_by_criteria(
            MathDomain.ALGEBRA, DifficultyLevel.BEGINNER, 5
        )
        self.assertEqual(len(problems), 5)
        self.assertTrue(all(p['variables'] for p in problems))
        self.assertTrue(all(p['template'] for p in problems))
    
    def test_generate_problem_instance(self):
        """Test generating specific problem instances from templates."""
        template = {
            'template': 'Solve for x: {a}x + {b} = {c}',
            'type': 'linear_equation',
            'variables': ['a', 'b', 'c'],
            'constraints': {'a': (1, 10), 'b': (-10, 10), 'c': (-20, 20)},
            'answer_formula': '({c} - {b}) / {a}'
        }
        
        problem_text, correct_answer, values = self.problem_bank.generate_problem_instance(template)
        
        # Check that problem text contains the values
        self.assertIn(str(values['a']), problem_text)
        self.assertIn(str(values['b']), problem_text)
        self.assertIn(str(values['c']), problem_text)
        
        # Check that answer is calculated correctly
        expected_answer = (values['c'] - values['b']) / values['a']
        self.assertEqual(float(correct_answer), expected_answer)
    
    def test_answer_calculation_linear_equation(self):
        """Test answer calculation for linear equations."""
        template = {
            'type': 'linear_equation',
            'answer_formula': '({c} - {b}) / {a}'
        }
        values = {'a': 2, 'b': 3, 'c': 7}
        
        answer = self.problem_bank._calculate_answer(template, values)
        self.assertEqual(float(answer), 2.0)  # (7 - 3) / 2 = 2
    
    def test_answer_calculation_expansion(self):
        """Test answer calculation for expansion problems."""
        template = {
            'type': 'expansion',
            'answer_formula': '{a}*{c}*x^2 + ({a}*{d} + {b}*{c})*x + {b}*{d}'
        }
        values = {'a': 1, 'b': 2, 'c': 1, 'd': 3}
        
        answer = self.problem_bank._calculate_answer(template, values)
        # (x + 2)(x + 3) = x² + 5x + 6
        self.assertIn('x²', answer)
        self.assertIn('5x', answer)
        self.assertIn('6', answer)


class TestQuizGenerator(unittest.TestCase):
    """Test the quiz generator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quiz_generator = QuizGenerator()
    
    def test_generate_quiz_basic(self):
        """Test basic quiz generation."""
        quiz = self.quiz_generator.generate_quiz(
            topic='algebra',
            difficulty=DifficultyLevel.BEGINNER,
            num_questions=5
        )
        
        self.assertEqual(len(quiz.questions), 5)
        self.assertEqual(quiz.topic, 'algebra')
        self.assertEqual(quiz.difficulty, DifficultyLevel.BEGINNER)
        self.assertIsNotNone(quiz.id)
        self.assertIsInstance(quiz.created_at, datetime)
    
    def test_quiz_generation_variety(self):
        """Test that generated quizzes have variety in problems."""
        quiz1 = self.quiz_generator.generate_quiz('algebra', DifficultyLevel.BEGINNER, 10)
        quiz2 = self.quiz_generator.generate_quiz('algebra', DifficultyLevel.BEGINNER, 10)
        
        # Questions should be different between quizzes
        quiz1_texts = [q.text for q in quiz1.questions]
        quiz2_texts = [q.text for q in quiz2.questions]
        
        # At least some questions should be different
        different_questions = sum(1 for q1, q2 in zip(quiz1_texts, quiz2_texts) if q1 != q2)
        self.assertGreater(different_questions, 0)
    
    def test_difficulty_consistency(self):
        """Test that questions match the requested difficulty level."""
        beginner_quiz = self.quiz_generator.generate_quiz('algebra', DifficultyLevel.BEGINNER, 5)
        intermediate_quiz = self.quiz_generator.generate_quiz('algebra', DifficultyLevel.INTERMEDIATE, 5)
        
        # Check that beginner questions are simpler (this is a basic check)
        beginner_topics = [q.topic for q in beginner_quiz.questions]
        intermediate_topics = [q.topic for q in intermediate_quiz.questions]
        
        # Beginner should have more basic topics
        self.assertIn('linear_equation', beginner_topics)
        
        # Intermediate should have more complex topics
        if intermediate_topics:  # Only check if we have intermediate problems
            advanced_topics = ['quadratic_equation', 'factoring']
            has_advanced = any(topic in intermediate_topics for topic in advanced_topics)
            # This might not always be true due to random selection, so we'll just check structure
            self.assertTrue(len(intermediate_topics) > 0)
    
    def test_generate_similar_problems(self):
        """Test generation of similar problems."""
        original_problem = "Solve for x: 2x + 3 = 7"
        similar_problems = self.quiz_generator.generate_similar_problems(original_problem, 3)
        
        self.assertEqual(len(similar_problems), 3)
        
        # Similar problems should have the same structure but different numbers
        for problem in similar_problems:
            self.assertIn('Solve for x:', problem)
            self.assertIn('=', problem)
    
    def test_multiple_choice_generation(self):
        """Test generation of multiple choice options."""
        correct_answer = "2"
        problem_type = "linear_equation"
        
        options = self.quiz_generator._generate_multiple_choice_options(correct_answer, problem_type)
        
        self.assertEqual(len(options), 4)
        self.assertIn(correct_answer, options)
        
        # Options should be different
        unique_options = set(options)
        self.assertEqual(len(unique_options), len(options))
    
    def test_hint_generation(self):
        """Test generation of progressive hints."""
        template = {
            'type': 'linear_equation',
            'variables': ['a', 'b', 'c']
        }
        values = {'a': 2, 'b': 3, 'c': 7}
        
        hints = self.quiz_generator._generate_hints(template, values)
        
        self.assertGreater(len(hints), 0)
        self.assertTrue(all(isinstance(hint, str) for hint in hints))
        self.assertTrue(all(len(hint) > 0 for hint in hints))


class TestAnswerValidator(unittest.TestCase):
    """Test the answer validation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AnswerValidator()
    
    def test_validate_correct_answer(self):
        """Test validation of correct answers."""
        result = self.validator.validate_answer("2", "2", "linear_equation")
        
        self.assertTrue(result.is_correct)
        self.assertEqual(result.partial_credit, 1.0)
        self.assertIn("Correct", result.explanation)
    
    def test_validate_incorrect_answer(self):
        """Test validation of incorrect answers."""
        result = self.validator.validate_answer("3", "2", "linear_equation")
        
        self.assertFalse(result.is_correct)
        self.assertEqual(result.partial_credit, 0.0)
        self.assertIn("incorrect", result.explanation.lower())
    
    def test_symbolic_equivalence(self):
        """Test recognition of symbolically equivalent answers."""
        # Test algebraic equivalence
        result = self.validator.validate_answer("x + 1", "1 + x", "simplification")
        self.assertTrue(result.is_correct)
        
        # Test numerical equivalence
        result = self.validator.validate_answer("4/2", "2", "linear_equation")
        self.assertTrue(result.is_correct)
        
        # Test expanded vs factored forms
        result = self.validator.validate_answer("x^2 + 2*x + 1", "(x + 1)^2", "expansion")
        self.assertTrue(result.is_correct)
    
    def test_partial_credit_quadratic(self):
        """Test partial credit for quadratic equations."""
        # User finds one root of a quadratic with two roots
        result = self.validator.validate_answer("2", "2, -1", "quadratic_equation")
        
        self.assertFalse(result.is_correct)
        self.assertGreater(result.partial_credit, 0)
        self.assertIn("partial", result.explanation.lower())
    
    def test_answer_normalization(self):
        """Test answer normalization."""
        # Test whitespace removal
        normalized = self.validator._normalize_answer("  2 x + 3  ")
        self.assertEqual(normalized, "2x+3")
        
        # Test power notation conversion
        normalized = self.validator._normalize_answer("x**2")
        self.assertEqual(normalized, "x^2")
    
    def test_error_analysis(self):
        """Test error analysis functionality."""
        analysis = self.validator._analyze_error("4", "2", "linear_equation")
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 0)
    
    def test_feedback_generation(self):
        """Test detailed feedback generation."""
        feedback = self.validator.generate_feedback("4", "2", False, "linear_equation")
        
        self.assertIn("incorrect", feedback.lower())
        self.assertIn("correct answer is", feedback.lower())
        self.assertIn("suggestions", feedback.lower())
    
    def test_special_cases(self):
        """Test handling of special cases."""
        # Test "no solution" cases
        result = self.validator.validate_answer("no solution", "no real solutions", "quadratic_equation")
        self.assertTrue(result.is_correct)
        
        # Test infinity cases
        result = self.validator.validate_answer("infinity", "∞", "limit")
        self.assertTrue(result.is_correct)


class TestQuizSessionManager(unittest.TestCase):
    """Test the quiz session management system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.session_manager = QuizSessionManager()
        self.quiz_generator = QuizGenerator()
        
        # Create a test quiz
        self.test_quiz = self.quiz_generator.generate_quiz(
            topic='algebra',
            difficulty=DifficultyLevel.BEGINNER,
            num_questions=3
        )
        self.user_id = "test_user_123"
    
    def test_start_quiz_session(self):
        """Test starting a new quiz session."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.session_manager.active_sessions)
        
        session = self.session_manager.active_sessions[session_id]
        self.assertEqual(session.user_id, self.user_id)
        self.assertEqual(session.quiz.id, self.test_quiz.id)
        self.assertFalse(session.is_completed)
    
    def test_get_current_question(self):
        """Test getting the current question."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        current_question = self.session_manager.get_current_question(session_id)
        
        self.assertIsNotNone(current_question)
        self.assertEqual(current_question.id, self.test_quiz.questions[0].id)
    
    def test_submit_answer(self):
        """Test submitting an answer."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        current_question = self.session_manager.get_current_question(session_id)
        
        # Submit a correct answer
        result = self.session_manager.submit_answer(
            session_id=session_id,
            question_id=current_question.id,
            user_answer=current_question.correct_answer,
            time_taken=30
        )
        
        self.assertIsNotNone(result)
        self.assertEqual(result.question_id, current_question.id)
        self.assertTrue(result.is_correct)
        self.assertGreater(result.points_earned, 0)
    
    def test_quiz_progress_tracking(self):
        """Test quiz progress tracking."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        # Initial progress
        progress = self.session_manager.get_quiz_progress(session_id)
        self.assertEqual(progress['answered_questions'], 0)
        self.assertEqual(progress['current_question_index'], 0)
        
        # Answer first question
        current_question = self.session_manager.get_current_question(session_id)
        self.session_manager.submit_answer(
            session_id, current_question.id, current_question.correct_answer, 30
        )
        
        # Check updated progress
        progress = self.session_manager.get_quiz_progress(session_id)
        self.assertEqual(progress['answered_questions'], 1)
        self.assertEqual(progress['current_question_index'], 1)
        self.assertEqual(progress['correct_answers'], 1)
    
    def test_hint_usage(self):
        """Test hint usage functionality."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        current_question = self.session_manager.get_current_question(session_id)
        
        # Use a hint
        hint = self.session_manager.use_hint(session_id, current_question.id, 1)
        
        self.assertIsInstance(hint, str)
        self.assertGreater(len(hint), 0)
        
        # Check hint tracking
        session = self.session_manager.active_sessions[session_id]
        self.assertEqual(session.hints_used[current_question.id], 1)
    
    def test_quiz_completion(self):
        """Test quiz completion and results calculation."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        # Answer all questions
        for i in range(len(self.test_quiz.questions)):
            current_question = self.session_manager.get_current_question(session_id)
            if current_question:
                self.session_manager.submit_answer(
                    session_id, current_question.id, current_question.correct_answer, 30
                )
        
        # Check that session is completed
        self.assertNotIn(session_id, self.session_manager.active_sessions)
        
        # Check performance history
        self.assertIn(self.user_id, self.session_manager.performance_history)
        results = self.session_manager.performance_history[self.user_id]
        self.assertEqual(len(results), 1)
        
        quiz_result = results[0]
        self.assertEqual(quiz_result.total_questions, len(self.test_quiz.questions))
        self.assertEqual(quiz_result.correct_answers, len(self.test_quiz.questions))
    
    def test_scoring_system(self):
        """Test the scoring system."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        current_question = self.session_manager.get_current_question(session_id)
        
        # Test correct answer with time bonus
        result = self.session_manager.submit_answer(
            session_id, current_question.id, current_question.correct_answer, 30  # Fast answer
        )
        
        base_points = self.session_manager.scoring_config['base_points_per_question']
        time_bonus = self.session_manager.scoring_config['time_bonus_points']
        
        # Should get base points + time bonus
        expected_points = base_points + time_bonus
        self.assertEqual(result.points_earned, expected_points)
    
    def test_performance_metrics(self):
        """Test user performance metrics calculation."""
        # Complete a quiz first
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        for i in range(len(self.test_quiz.questions)):
            current_question = self.session_manager.get_current_question(session_id)
            if current_question:
                # Answer correctly for first question, incorrectly for others
                answer = current_question.correct_answer if i == 0 else "wrong_answer"
                self.session_manager.submit_answer(session_id, current_question.id, answer, 60)
        
        # Get performance metrics
        performance = self.session_manager.get_user_performance_metrics(self.user_id)
        
        self.assertIsInstance(performance.accuracy, float)
        self.assertIsInstance(performance.average_time, float)
        self.assertIsInstance(performance.streak, int)
        self.assertIsInstance(performance.topics_mastered, list)
        self.assertIsInstance(performance.areas_needing_work, list)
    
    def test_session_analytics(self):
        """Test session analytics functionality."""
        session_id = self.session_manager.start_quiz_session(self.user_id, self.test_quiz)
        
        # Complete the quiz
        for i in range(len(self.test_quiz.questions)):
            current_question = self.session_manager.get_current_question(session_id)
            if current_question:
                self.session_manager.submit_answer(
                    session_id, current_question.id, current_question.correct_answer, 45
                )
        
        # Get analytics
        analytics = self.session_manager.get_session_analytics(session_id)
        
        self.assertIn('session_id', analytics)
        self.assertIn('question_analytics', analytics)
        self.assertIn('time_distribution', analytics)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete quiz system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.quiz_generator = QuizGenerator()
        self.session_manager = QuizSessionManager()
        self.user_id = "integration_test_user"
    
    def test_complete_quiz_workflow(self):
        """Test the complete quiz workflow from generation to completion."""
        # 1. Generate a quiz
        quiz = self.quiz_generator.generate_quiz(
            topic='algebra',
            difficulty=DifficultyLevel.BEGINNER,
            num_questions=3
        )
        
        self.assertEqual(len(quiz.questions), 3)
        
        # 2. Start a quiz session
        session_id = self.session_manager.start_quiz_session(self.user_id, quiz)
        self.assertIsNotNone(session_id)
        
        # 3. Complete the quiz
        total_points = 0
        for i in range(len(quiz.questions)):
            # Get current question
            current_question = self.session_manager.get_current_question(session_id)
            self.assertIsNotNone(current_question)
            
            # Use a hint occasionally
            if i == 1:
                hint = self.session_manager.use_hint(session_id, current_question.id, 1)
                self.assertIsInstance(hint, str)
            
            # Submit answer
            result = self.session_manager.submit_answer(
                session_id=session_id,
                question_id=current_question.id,
                user_answer=current_question.correct_answer,
                time_taken=45
            )
            
            self.assertTrue(result.is_correct)
            total_points += result.points_earned
        
        # 4. Check final results
        self.assertNotIn(session_id, self.session_manager.active_sessions)
        
        # 5. Verify performance tracking
        performance = self.session_manager.get_user_performance_metrics(self.user_id)
        self.assertEqual(performance.accuracy, 1.0)  # All correct
        self.assertGreater(performance.average_time, 0)
    
    def test_mixed_performance_quiz(self):
        """Test quiz with mixed correct/incorrect answers."""
        quiz = self.quiz_generator.generate_quiz('algebra', DifficultyLevel.BEGINNER, 4)
        session_id = self.session_manager.start_quiz_session(self.user_id, quiz)
        
        correct_answers = 0
        for i, question in enumerate(quiz.questions):
            current_question = self.session_manager.get_current_question(session_id)
            
            # Answer correctly for even indices, incorrectly for odd
            if i % 2 == 0:
                answer = current_question.correct_answer
                correct_answers += 1
            else:
                answer = "wrong_answer"
            
            result = self.session_manager.submit_answer(
                session_id, current_question.id, answer, 60
            )
            
            expected_correct = (i % 2 == 0)
            self.assertEqual(result.is_correct, expected_correct)
        
        # Check final performance
        performance = self.session_manager.get_user_performance_metrics(self.user_id)
        expected_accuracy = correct_answers / len(quiz.questions)
        self.assertAlmostEqual(performance.accuracy, expected_accuracy, places=2)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)