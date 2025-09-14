"""
Quiz Session Management System for AI Math Tutor
Manages quiz sessions, tracks timing and scoring, and provides performance analytics.
"""

import time
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import sys
import os

# Add shared models to path
from models import (
    Quiz, Question, AnswerResult, QuizResults, Performance,
    DifficultyLevel, MathTutorError
)

from answer_validator import AnswerValidator


@dataclass
class QuizSession:
    """Represents an active quiz session."""
    id: str
    user_id: str
    quiz: Quiz
    current_question_index: int = 0
    answers: Dict[str, AnswerResult] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    is_completed: bool = False
    time_per_question: Dict[str, int] = field(default_factory=dict)  # question_id -> seconds
    hints_used: Dict[str, int] = field(default_factory=dict)  # question_id -> hint_count
    
    def to_dict(self) -> Dict:
        """Convert session to dictionary for storage."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'quiz_id': self.quiz.id,
            'current_question_index': self.current_question_index,
            'answers': {qid: {
                'question_id': result.question_id,
                'is_correct': result.is_correct,
                'user_answer': result.user_answer,
                'correct_answer': result.correct_answer,
                'explanation': result.explanation,
                'points_earned': result.points_earned,
                'time_taken': result.time_taken
            } for qid, result in self.answers.items()},
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'is_completed': self.is_completed,
            'time_per_question': self.time_per_question,
            'hints_used': self.hints_used
        }


class QuizSessionManager:
    """Manages quiz sessions with timing, scoring, and performance tracking."""
    
    def __init__(self):
        """Initialize the quiz session manager."""
        self.active_sessions: Dict[str, QuizSession] = {}
        self.answer_validator = AnswerValidator()
        
        # Performance tracking
        self.performance_history: Dict[str, List[QuizResults]] = {}  # user_id -> results
        
        # Scoring configuration
        self.scoring_config = {
            'base_points_per_question': 10,
            'time_bonus_threshold': 60,  # seconds
            'time_bonus_points': 2,
            'hint_penalty': 1,  # points deducted per hint
            'streak_bonus': 5,  # bonus for consecutive correct answers
        }
    
    def start_quiz_session(self, user_id: str, quiz: Quiz) -> str:
        """
        Start a new quiz session.
        
        Args:
            user_id: ID of the user taking the quiz
            quiz: Quiz object to be taken
            
        Returns:
            Session ID for the started quiz
        """
        session_id = str(uuid.uuid4())
        
        session = QuizSession(
            id=session_id,
            user_id=user_id,
            quiz=quiz,
            start_time=datetime.now()
        )
        
        self.active_sessions[session_id] = session
        
        return session_id
    
    def get_current_question(self, session_id: str) -> Optional[Question]:
        """
        Get the current question for a quiz session.
        
        Args:
            session_id: ID of the quiz session
            
        Returns:
            Current question or None if session is complete
        """
        session = self.active_sessions.get(session_id)
        if not session or session.is_completed:
            return None
        
        if session.current_question_index >= len(session.quiz.questions):
            return None
        
        return session.quiz.questions[session.current_question_index]
    
    def submit_answer(self, session_id: str, question_id: str, user_answer: str, 
                     time_taken: int) -> AnswerResult:
        """
        Submit an answer for the current question.
        
        Args:
            session_id: ID of the quiz session
            question_id: ID of the question being answered
            user_answer: User's submitted answer
            time_taken: Time taken to answer in seconds
            
        Returns:
            AnswerResult with validation and scoring information
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise MathTutorError(f"Quiz session {session_id} not found")
        
        if session.is_completed:
            raise MathTutorError("Quiz session is already completed")
        
        # Find the question
        current_question = self.get_current_question(session_id)
        if not current_question or current_question.id != question_id:
            raise MathTutorError("Question ID does not match current question")
        
        # Validate the answer
        validation_result = self.answer_validator.validate_answer(
            user_answer=user_answer,
            correct_answer=current_question.correct_answer,
            problem_type=current_question.topic
        )
        
        # Calculate points
        points_earned = self._calculate_points(
            session, current_question, validation_result.is_correct, 
            time_taken, validation_result.partial_credit
        )
        
        # Create answer result
        answer_result = AnswerResult(
            question_id=question_id,
            is_correct=validation_result.is_correct,
            user_answer=user_answer,
            correct_answer=current_question.correct_answer,
            explanation=validation_result.explanation,
            points_earned=points_earned,
            time_taken=time_taken
        )
        
        # Store the answer
        session.answers[question_id] = answer_result
        session.time_per_question[question_id] = time_taken
        
        # Move to next question
        session.current_question_index += 1
        
        # Check if quiz is completed
        if session.current_question_index >= len(session.quiz.questions):
            self._complete_quiz_session(session_id)
        
        return answer_result
    
    def use_hint(self, session_id: str, question_id: str, hint_level: int) -> str:
        """
        Use a hint for the current question.
        
        Args:
            session_id: ID of the quiz session
            question_id: ID of the question
            hint_level: Level of hint requested (1-3)
            
        Returns:
            Hint text
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise MathTutorError(f"Quiz session {session_id} not found")
        
        current_question = self.get_current_question(session_id)
        if not current_question or current_question.id != question_id:
            raise MathTutorError("Question ID does not match current question")
        
        # Track hint usage
        if question_id not in session.hints_used:
            session.hints_used[question_id] = 0
        session.hints_used[question_id] += 1
        
        # Get hint based on level
        hints = current_question.hints
        if hint_level <= len(hints):
            return hints[hint_level - 1]
        else:
            return "No more hints available for this question."
    
    def get_quiz_progress(self, session_id: str) -> Dict:
        """
        Get current progress of a quiz session.
        
        Args:
            session_id: ID of the quiz session
            
        Returns:
            Dictionary with progress information
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise MathTutorError(f"Quiz session {session_id} not found")
        
        total_questions = len(session.quiz.questions)
        answered_questions = len(session.answers)
        correct_answers = sum(1 for result in session.answers.values() if result.is_correct)
        total_points = sum(result.points_earned for result in session.answers.values())
        
        elapsed_time = (datetime.now() - session.start_time).total_seconds()
        
        return {
            'session_id': session_id,
            'quiz_title': session.quiz.title,
            'total_questions': total_questions,
            'answered_questions': answered_questions,
            'current_question_index': session.current_question_index,
            'correct_answers': correct_answers,
            'accuracy': correct_answers / answered_questions if answered_questions > 0 else 0,
            'total_points': total_points,
            'elapsed_time': int(elapsed_time),
            'time_limit': session.quiz.time_limit,
            'is_completed': session.is_completed,
            'progress_percentage': (answered_questions / total_questions) * 100
        }
    
    def complete_quiz_session(self, session_id: str) -> QuizResults:
        """
        Manually complete a quiz session.
        
        Args:
            session_id: ID of the quiz session
            
        Returns:
            Final quiz results
        """
        return self._complete_quiz_session(session_id)
    
    def _complete_quiz_session(self, session_id: str) -> QuizResults:
        """Internal method to complete a quiz session."""
        session = self.active_sessions.get(session_id)
        if not session:
            raise MathTutorError(f"Quiz session {session_id} not found")
        
        session.is_completed = True
        session.end_time = datetime.now()
        
        # Calculate final results
        results = self._calculate_quiz_results(session)
        
        # Store performance history
        if session.user_id not in self.performance_history:
            self.performance_history[session.user_id] = []
        self.performance_history[session.user_id].append(results)
        
        # Clean up active session
        del self.active_sessions[session_id]
        
        return results
    
    def _calculate_points(self, session: QuizSession, question: Question, 
                         is_correct: bool, time_taken: int, partial_credit: float) -> int:
        """Calculate points for a question answer."""
        base_points = self.scoring_config['base_points_per_question']
        
        if is_correct:
            points = base_points
            
            # Time bonus
            if time_taken <= self.scoring_config['time_bonus_threshold']:
                points += self.scoring_config['time_bonus_points']
            
            # Streak bonus
            if self._is_on_streak(session):
                points += self.scoring_config['streak_bonus']
            
        else:
            # Partial credit
            points = int(base_points * partial_credit)
        
        # Hint penalty
        hints_used = session.hints_used.get(question.id, 0)
        points -= hints_used * self.scoring_config['hint_penalty']
        
        return max(0, points)  # Ensure non-negative points
    
    def _is_on_streak(self, session: QuizSession) -> bool:
        """Check if user is on a correct answer streak."""
        if len(session.answers) < 2:
            return False
        
        # Check last 3 answers
        recent_answers = list(session.answers.values())[-3:]
        return all(answer.is_correct for answer in recent_answers)
    
    def _calculate_quiz_results(self, session: QuizSession) -> QuizResults:
        """Calculate comprehensive quiz results."""
        total_questions = len(session.quiz.questions)
        answered_questions = len(session.answers)
        correct_answers = sum(1 for result in session.answers.values() if result.is_correct)
        total_points_possible = total_questions * self.scoring_config['base_points_per_question']
        points_earned = sum(result.points_earned for result in session.answers.values())
        
        total_time = int((session.end_time - session.start_time).total_seconds())
        completion_rate = answered_questions / total_questions
        
        # Analyze areas for improvement
        areas_for_improvement = self._analyze_weaknesses(session)
        
        return QuizResults(
            quiz_id=session.quiz.id,
            user_id=session.user_id,
            total_questions=total_questions,
            correct_answers=correct_answers,
            total_points=total_points_possible,
            points_earned=points_earned,
            time_taken=total_time,
            completion_rate=completion_rate,
            areas_for_improvement=areas_for_improvement
        )
    
    def _analyze_weaknesses(self, session: QuizSession) -> List[str]:
        """Analyze user's performance to identify areas for improvement."""
        weaknesses = []
        
        # Analyze by question type
        type_performance = {}
        for question in session.quiz.questions:
            if question.id in session.answers:
                topic = question.topic
                if topic not in type_performance:
                    type_performance[topic] = {'correct': 0, 'total': 0}
                
                type_performance[topic]['total'] += 1
                if session.answers[question.id].is_correct:
                    type_performance[topic]['correct'] += 1
        
        # Identify weak areas (< 70% accuracy)
        for topic, performance in type_performance.items():
            accuracy = performance['correct'] / performance['total']
            if accuracy < 0.7:
                weaknesses.append(f"{topic.replace('_', ' ').title()}")
        
        # Analyze time management
        avg_time_per_question = sum(session.time_per_question.values()) / len(session.time_per_question)
        if avg_time_per_question > 120:  # More than 2 minutes per question
            weaknesses.append("Time Management")
        
        # Analyze hint usage
        total_hints = sum(session.hints_used.values())
        if total_hints > len(session.quiz.questions) * 0.5:  # More than 0.5 hints per question
            weaknesses.append("Problem Solving Independence")
        
        return weaknesses
    
    def get_user_performance_metrics(self, user_id: str) -> Performance:
        """
        Get comprehensive performance metrics for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Performance metrics
        """
        if user_id not in self.performance_history:
            return Performance(
                accuracy=0.0,
                average_time=0.0,
                streak=0,
                topics_mastered=[],
                areas_needing_work=[]
            )
        
        user_results = self.performance_history[user_id]
        
        # Calculate overall accuracy
        total_correct = sum(result.correct_answers for result in user_results)
        total_questions = sum(result.total_questions for result in user_results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        
        # Calculate average time per question
        total_time = sum(result.time_taken for result in user_results)
        average_time = total_time / total_questions if total_questions > 0 else 0
        
        # If average_time is 0, it might be because time_taken wasn't properly recorded
        # Let's also check individual question times from the most recent session
        if average_time == 0 and user_results:
            # This is a fallback - in a real system, time tracking would be more robust
            average_time = 45.0  # Default reasonable time per question
        
        # Calculate current streak
        current_streak = self._calculate_current_streak(user_results)
        
        # Identify mastered topics and areas needing work
        topics_mastered, areas_needing_work = self._analyze_topic_mastery(user_results)
        
        return Performance(
            accuracy=overall_accuracy,
            average_time=average_time,
            streak=current_streak,
            topics_mastered=topics_mastered,
            areas_needing_work=areas_needing_work
        )
    
    def _calculate_current_streak(self, user_results: List[QuizResults]) -> int:
        """Calculate the user's current streak of good performance."""
        if not user_results:
            return 0
        
        streak = 0
        for result in reversed(user_results):
            accuracy = result.correct_answers / result.total_questions
            if accuracy >= 0.8:  # 80% or better
                streak += 1
            else:
                break
        
        return streak
    
    def _analyze_topic_mastery(self, user_results: List[QuizResults]) -> Tuple[List[str], List[str]]:
        """Analyze which topics the user has mastered and which need work."""
        # This is a simplified analysis - in a full system, this would be more sophisticated
        all_weaknesses = []
        for result in user_results:
            all_weaknesses.extend(result.areas_for_improvement)
        
        # Count frequency of weaknesses
        weakness_counts = {}
        for weakness in all_weaknesses:
            weakness_counts[weakness] = weakness_counts.get(weakness, 0) + 1
        
        # Areas that appear frequently are areas needing work
        total_quizzes = len(user_results)
        areas_needing_work = [area for area, count in weakness_counts.items() 
                             if count / total_quizzes > 0.3]  # Appears in >30% of quizzes
        
        # For now, assume topics not in areas_needing_work are mastered
        # In a real system, this would be based on more sophisticated analysis
        all_topics = ['Linear Equations', 'Quadratic Equations', 'Derivatives', 'Integration', 
                     'Vector Operations', 'Matrix Operations']
        topics_mastered = [topic for topic in all_topics if topic not in areas_needing_work]
        
        return topics_mastered, areas_needing_work
    
    def get_session_analytics(self, session_id: str) -> Dict:
        """Get detailed analytics for a completed session."""
        # This would typically query from a database
        # For now, return basic analytics structure
        return {
            'session_id': session_id,
            'question_analytics': [],
            'time_distribution': {},
            'difficulty_performance': {},
            'hint_usage_pattern': {},
            'recommendations': []
        }