"""
Core data models for the AI Math Tutor math engine.
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


@dataclass
class ParsedProblem:
    """Represents a parsed mathematical problem."""
    id: str
    original_text: str
    domain: MathDomain
    expression_tree: Dict[str, Any]
    variables: List[str]
    constants: List[str]
    metadata: Dict[str, Any]


@dataclass
class SolutionStep:
    """Represents a single step in a mathematical solution."""
    step_number: int
    operation: str
    explanation: str
    mathematical_expression: str
    intermediate_result: str
    confidence_score: float


@dataclass
class StepSolution:
    """Represents a complete step-by-step solution."""
    steps: List[SolutionStep]
    final_answer: str
    solution_method: str
    confidence_score: float
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Represents the result of answer validation."""
    is_correct: bool
    confidence_score: float
    feedback: str
    suggested_corrections: List[str]


@dataclass
class Explanation:
    """Represents an AI-generated explanation."""
    content: str
    complexity_level: str
    related_concepts: List[str]
    examples: List[str]


@dataclass
class Hint:
    """Represents a contextual hint."""
    content: str
    hint_level: int
    reveals_answer: bool


@dataclass
class PlotData:
    """Represents mathematical visualization data."""
    plot_type: str
    data_points: List[Tuple[float, float]]
    styling: Dict[str, Any]
    interactive_elements: List[Dict[str, Any]]


# Additional classes needed by the math engine
class DifficultyLevel(Enum):
    """Difficulty levels for problems."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class QuestionType(Enum):
    """Types of quiz questions."""
    MULTIPLE_CHOICE = "multiple_choice"
    FILL_IN_BLANK = "fill_in_blank"
    STEP_BY_STEP = "step_by_step"
    TRUE_FALSE = "true_false"


# Exception classes
class MathTutorError(Exception):
    """Base exception for math tutor errors."""
    pass


class ComputationError(MathTutorError):
    """Error during mathematical computation."""
    pass


class ParseError(MathTutorError):
    """Error during problem parsing."""
    pass


# Additional data classes
@dataclass
class Point:
    """Represents a 2D or 3D point."""
    x: float
    y: float
    z: Optional[float] = None


@dataclass
class PlotStyle:
    """Styling information for plots."""
    color: str
    line_width: float
    marker_style: str


@dataclass
class InteractiveElement:
    """Interactive element in a plot."""
    element_type: str
    properties: Dict[str, Any]


@dataclass
class AnimationData:
    """Data for animated visualizations."""
    frames: List[Dict[str, Any]]
    duration: float
    loop: bool


@dataclass
class Question:
    """Represents a quiz question."""
    id: str
    text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    hints: List[str]


@dataclass
class Quiz:
    """Represents a complete quiz."""
    id: str
    questions: List[Question]
    time_limit: Optional[int]
    topic: str
    difficulty: DifficultyLevel


@dataclass
class AnswerResult:
    """Result of answering a question."""
    is_correct: bool
    feedback: str
    explanation: str


@dataclass
class QuizResults:
    """Results of a completed quiz."""
    quiz_id: str
    score: float
    total_questions: int
    correct_answers: int
    time_taken: int


@dataclass
class Performance:
    """Performance metrics for a user."""
    accuracy: float
    speed: float
    consistency: float