"""
Core data models and interfaces for the AI Math Tutor system.
These models define the contract between services.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import uuid
from datetime import datetime


class MathDomain(Enum):
    """Mathematical domains supported by the system."""
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    STATISTICS = "statistics"
    AI_ML_MATH = "ai_ml_math"


class QuestionType(Enum):
    """Types of quiz questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    NUMERIC = "numeric"
    ALGEBRAIC = "algebraic"
    TRUE_FALSE = "true_false"


class DifficultyLevel(Enum):
    """Difficulty levels for problems and quizzes."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4


@dataclass
class ParsedProblem:
    """Represents a parsed mathematical problem."""
    id: str
    original_text: str
    domain: MathDomain
    difficulty: DifficultyLevel
    variables: List[str]
    expressions: List[str]
    problem_type: str
    metadata: Dict[str, Any]


@dataclass
class SolutionStep:
    """Represents a single step in a mathematical solution."""
    step_number: int
    operation: str
    explanation: str
    mathematical_expression: str
    intermediate_result: str
    reasoning: Optional[str] = None


@dataclass
class StepSolution:
    """Complete step-by-step solution to a mathematical problem."""
    problem_id: str
    steps: List[SolutionStep]
    final_answer: str
    solution_method: str
    confidence_score: float
    computation_time: float


@dataclass
class ValidationResult:
    """Result of validating a user's answer."""
    is_correct: bool
    user_answer: str
    correct_answer: str
    explanation: Optional[str] = None
    partial_credit: float = 0.0


@dataclass
class Explanation:
    """AI-generated explanation for mathematical concepts."""
    content: str
    complexity_level: str
    related_concepts: List[str]
    examples: List[str]
    confidence_score: float


@dataclass
class Hint:
    """Contextual hint for problem solving."""
    content: str
    hint_level: int  # 1 = subtle, 5 = very direct
    reveals_answer: bool
    related_concepts: List[str]


@dataclass
class Point:
    """2D or 3D point for visualizations."""
    x: float
    y: float
    z: Optional[float] = None


@dataclass
class PlotStyle:
    """Styling configuration for mathematical plots."""
    color: str
    line_width: float
    marker_style: Optional[str] = None
    transparency: float = 1.0


@dataclass
class InteractiveElement:
    """Interactive element in a visualization."""
    element_type: str
    position: Point
    action: str
    tooltip: Optional[str] = None


@dataclass
class PlotData:
    """Data structure for mathematical visualizations."""
    plot_type: str
    data_points: List[Point]
    styling: PlotStyle
    interactive_elements: List[InteractiveElement]
    title: str
    axis_labels: Dict[str, str]


@dataclass
class AnimationData:
    """Data for animated mathematical visualizations."""
    frames: List[PlotData]
    frame_duration: float
    loop: bool
    controls: List[str]


@dataclass
class Question:
    """Quiz question structure."""
    id: str
    text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    hints: List[str]
    difficulty: DifficultyLevel
    topic: str


@dataclass
class Quiz:
    """Complete quiz structure."""
    id: str
    title: str
    questions: List[Question]
    time_limit: Optional[int]  # seconds
    topic: str
    difficulty: DifficultyLevel
    created_at: datetime


@dataclass
class AnswerResult:
    """Result of submitting a quiz answer."""
    question_id: str
    is_correct: bool
    user_answer: str
    correct_answer: str
    explanation: str
    points_earned: int
    time_taken: int  # seconds


@dataclass
class QuizResults:
    """Complete quiz results."""
    quiz_id: str
    user_id: str
    total_questions: int
    correct_answers: int
    total_points: int
    points_earned: int
    time_taken: int  # seconds
    completion_rate: float
    areas_for_improvement: List[str]


@dataclass
class Performance:
    """User performance metrics."""
    accuracy: float
    average_time: float
    streak: int
    topics_mastered: List[str]
    areas_needing_work: List[str]


@dataclass
class UserPreferences:
    """User learning preferences."""
    preferred_explanation_level: str
    visual_learning: bool
    step_by_step_detail: str  # "minimal", "standard", "detailed"
    notification_settings: Dict[str, bool]


@dataclass
class ProgressMetrics:
    """Detailed progress tracking metrics."""
    total_problems_solved: int
    current_streak: int
    longest_streak: int
    average_accuracy: float
    time_spent_learning: int  # minutes
    topics_completed: List[str]
    skill_levels: Dict[str, int]


@dataclass
class UserProfile:
    """Complete user profile with learning data."""
    id: str
    username: str
    email: str
    skill_levels: Dict[str, int]
    learning_goals: List[str]
    preferences: UserPreferences
    progress_metrics: ProgressMetrics
    created_at: datetime
    updated_at: datetime


@dataclass
class Recommendation:
    """Learning recommendation for a user."""
    type: str  # "topic", "difficulty", "practice"
    content: str
    reason: str
    priority: int
    estimated_time: int  # minutes


@dataclass
class LearningPath:
    """Personalized learning path for a user."""
    user_id: str
    current_topic: str
    next_topics: List[str]
    recommendations: List[Recommendation]
    estimated_completion_time: int  # days
    progress_percentage: float


@dataclass
class MathContext:
    """Context for AI explanations and hints."""
    problem: ParsedProblem
    current_step: Optional[int]
    user_level: str
    previous_attempts: List[str]
    related_concepts: List[str]


# Error classes for better error handling
class MathTutorError(Exception):
    """Base exception for AI Math Tutor system."""
    pass


class ParseError(MathTutorError):
    """Error parsing mathematical expressions."""
    pass


class ComputationError(MathTutorError):
    """Error during mathematical computation."""
    pass


class ValidationError(MathTutorError):
    """Error validating user input or answers."""
    pass


class AIServiceError(MathTutorError):
    """Error with AI explanation services."""
    pass