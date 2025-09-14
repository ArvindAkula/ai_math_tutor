"""
Adaptive learning path algorithms for personalized math instruction.
Implements performance analysis, topic recommendations, and difficulty adjustment.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

from shared.models.core import (
    MathDomain, DifficultyLevel, LearningPath, Recommendation,
    UserProfile, Performance
)
from user_progress import UserProgressDatabase, SkillAssessment

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of learning recommendations."""
    TOPIC_INTRODUCTION = "topic_introduction"
    SKILL_REINFORCEMENT = "skill_reinforcement"
    DIFFICULTY_INCREASE = "difficulty_increase"
    DIFFICULTY_DECREASE = "difficulty_decrease"
    REVIEW_FUNDAMENTALS = "review_fundamentals"
    PRACTICE_WEAK_AREAS = "practice_weak_areas"
    ADVANCE_TO_NEW_TOPIC = "advance_to_new_topic"


@dataclass
class TopicDependency:
    """Represents prerequisite relationships between topics."""
    topic: str
    prerequisites: List[str]
    difficulty_level: int
    estimated_hours: float
    importance_weight: float


@dataclass
class LearningObjective:
    """Specific learning objective within a topic."""
    id: str
    topic: str
    title: str
    description: str
    difficulty: int
    prerequisites: List[str]
    skills_developed: List[str]
    estimated_time: int  # minutes


@dataclass
class PerformanceAnalysis:
    """Detailed analysis of user performance."""
    user_id: str
    overall_accuracy: float
    consistency_score: float
    learning_velocity: float
    strengths: List[str]
    weaknesses: List[str]
    recommended_difficulty: int
    confidence_level: float
    analysis_timestamp: datetime


class AdaptiveLearningEngine:
    """Core engine for adaptive learning path generation and management."""
    
    def __init__(self, progress_db: UserProgressDatabase):
        """Initialize the adaptive learning engine."""
        self.progress_db = progress_db
        self.topic_dependencies = self._initialize_topic_dependencies()
        self.learning_objectives = self._initialize_learning_objectives()
    
    def _initialize_topic_dependencies(self) -> Dict[str, TopicDependency]:
        """Initialize the topic dependency graph."""
        dependencies = {
            # Basic Algebra
            'basic_algebra': TopicDependency(
                topic='basic_algebra',
                prerequisites=[],
                difficulty_level=1,
                estimated_hours=8.0,
                importance_weight=1.0
            ),
            
            # Advanced Algebra
            'advanced_algebra': TopicDependency(
                topic='advanced_algebra',
                prerequisites=['basic_algebra'],
                difficulty_level=2,
                estimated_hours=12.0,
                importance_weight=0.9
            ),
            
            # Pre-Calculus
            'precalculus': TopicDependency(
                topic='precalculus',
                prerequisites=['advanced_algebra'],
                difficulty_level=2,
                estimated_hours=15.0,
                importance_weight=0.8
            ),
            
            # Calculus I
            'calculus_1': TopicDependency(
                topic='calculus_1',
                prerequisites=['precalculus'],
                difficulty_level=3,
                estimated_hours=20.0,
                importance_weight=0.9
            ),
            
            # Calculus II
            'calculus_2': TopicDependency(
                topic='calculus_2',
                prerequisites=['calculus_1'],
                difficulty_level=3,
                estimated_hours=18.0,
                importance_weight=0.8
            ),
            
            # Linear Algebra
            'linear_algebra': TopicDependency(
                topic='linear_algebra',
                prerequisites=['advanced_algebra'],
                difficulty_level=3,
                estimated_hours=16.0,
                importance_weight=0.9
            ),
            
            # Statistics
            'statistics': TopicDependency(
                topic='statistics',
                prerequisites=['basic_algebra'],
                difficulty_level=2,
                estimated_hours=14.0,
                importance_weight=0.7
            ),
            
            # AI/ML Mathematics
            'ai_ml_math': TopicDependency(
                topic='ai_ml_math',
                prerequisites=['linear_algebra', 'calculus_1', 'statistics'],
                difficulty_level=4,
                estimated_hours=25.0,
                importance_weight=1.0
            )
        }
        return dependencies
    
    def _initialize_learning_objectives(self) -> Dict[str, List[LearningObjective]]:
        """Initialize learning objectives for each topic."""
        objectives = {
            'basic_algebra': [
                LearningObjective(
                    id='ba_001',
                    topic='basic_algebra',
                    title='Linear Equations',
                    description='Solve linear equations in one variable',
                    difficulty=1,
                    prerequisites=[],
                    skills_developed=['equation_solving', 'algebraic_manipulation'],
                    estimated_time=60
                ),
                LearningObjective(
                    id='ba_002',
                    topic='basic_algebra',
                    title='Systems of Linear Equations',
                    description='Solve systems of linear equations',
                    difficulty=2,
                    prerequisites=['ba_001'],
                    skills_developed=['system_solving', 'substitution', 'elimination'],
                    estimated_time=90
                )
            ],
            
            'calculus_1': [
                LearningObjective(
                    id='c1_001',
                    topic='calculus_1',
                    title='Limits',
                    description='Understand and compute limits of functions',
                    difficulty=2,
                    prerequisites=[],
                    skills_developed=['limit_computation', 'continuity'],
                    estimated_time=120
                ),
                LearningObjective(
                    id='c1_002',
                    topic='calculus_1',
                    title='Derivatives',
                    description='Compute derivatives using various rules',
                    difficulty=3,
                    prerequisites=['c1_001'],
                    skills_developed=['differentiation', 'chain_rule', 'product_rule'],
                    estimated_time=150
                )
            ],
            
            'linear_algebra': [
                LearningObjective(
                    id='la_001',
                    topic='linear_algebra',
                    title='Vector Operations',
                    description='Perform basic vector operations',
                    difficulty=2,
                    prerequisites=[],
                    skills_developed=['vector_addition', 'scalar_multiplication', 'dot_product'],
                    estimated_time=90
                ),
                LearningObjective(
                    id='la_002',
                    topic='linear_algebra',
                    title='Matrix Operations',
                    description='Perform matrix operations and transformations',
                    difficulty=3,
                    prerequisites=['la_001'],
                    skills_developed=['matrix_multiplication', 'determinants', 'inverse'],
                    estimated_time=120
                )
            ]
        }
        return objectives
    
    def analyze_user_performance(self, user_id: str, days: int = 30) -> PerformanceAnalysis:
        """Analyze user's recent performance across all domains."""
        try:
            # Get user profile and analytics
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                raise ValueError(f"User profile not found: {user_id}")
            
            analytics = self.progress_db.get_learning_analytics(user_id, days)
            
            # Calculate overall accuracy
            total_attempts = sum(day.get('problems_attempted', 0) for day in analytics['daily_activity'])
            total_solved = sum(day.get('problems_solved', 0) for day in analytics['daily_activity'])
            overall_accuracy = total_solved / max(total_attempts, 1)
            
            # Calculate consistency score (lower variance = higher consistency)
            daily_accuracies = []
            for day in analytics['daily_activity']:
                attempted = day.get('problems_attempted', 0)
                solved = day.get('problems_solved', 0)
                if attempted > 0:
                    daily_accuracies.append(solved / attempted)
            
            if len(daily_accuracies) > 1:
                consistency_score = 1.0 - np.std(daily_accuracies)
            else:
                consistency_score = 0.5  # Neutral score for insufficient data
            
            # Calculate learning velocity (problems solved per day)
            learning_velocity = total_solved / max(days, 1)
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(analytics['domain_performance'])
            
            # Recommend difficulty level
            recommended_difficulty = self._calculate_recommended_difficulty(profile, analytics)
            
            # Calculate confidence level based on recent performance
            confidence_level = self._calculate_confidence_level(analytics)
            
            return PerformanceAnalysis(
                user_id=user_id,
                overall_accuracy=overall_accuracy,
                consistency_score=max(0.0, min(1.0, consistency_score)),
                learning_velocity=learning_velocity,
                strengths=strengths,
                weaknesses=weaknesses,
                recommended_difficulty=recommended_difficulty,
                confidence_level=confidence_level,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze user performance: {e}")
            # Return default analysis
            return PerformanceAnalysis(
                user_id=user_id,
                overall_accuracy=0.0,
                consistency_score=0.0,
                learning_velocity=0.0,
                strengths=[],
                weaknesses=['insufficient_data'],
                recommended_difficulty=1,
                confidence_level=0.0,
                analysis_timestamp=datetime.now()
            )
    
    def _identify_strengths_weaknesses(self, domain_performance: List[Dict]) -> Tuple[List[str], List[str]]:
        """Identify user's strengths and weaknesses based on domain performance."""
        strengths = []
        weaknesses = []
        
        for domain_data in domain_performance:
            domain = domain_data['domain']
            accuracy = domain_data['accuracy']
            attempts = domain_data['attempts']
            
            # Consider domain a strength if accuracy > 0.75 and sufficient attempts
            if accuracy > 0.75 and attempts >= 5:
                strengths.append(domain)
            # Consider domain a weakness if accuracy < 0.5 and sufficient attempts
            elif accuracy < 0.5 and attempts >= 3:
                weaknesses.append(domain)
        
        return strengths, weaknesses
    
    def _calculate_recommended_difficulty(self, profile: UserProfile, analytics: Dict) -> int:
        """Calculate recommended difficulty level based on performance."""
        difficulty_data = analytics.get('difficulty_progression', [])
        
        if not difficulty_data:
            return 1  # Default to beginner
        
        # Find the highest difficulty level with good performance (>70% accuracy)
        max_good_difficulty = 1
        for diff_data in difficulty_data:
            difficulty = diff_data['difficulty_level']
            accuracy = diff_data['accuracy']
            attempts = diff_data['attempts']
            
            if accuracy > 0.7 and attempts >= 3:
                max_good_difficulty = max(max_good_difficulty, difficulty)
        
        # Recommend one level higher if performing well, or same level if struggling
        current_avg_accuracy = sum(d['accuracy'] for d in difficulty_data) / len(difficulty_data)
        
        if current_avg_accuracy > 0.8:
            return min(4, max_good_difficulty + 1)  # Level up
        elif current_avg_accuracy < 0.5:
            return max(1, max_good_difficulty - 1)  # Level down
        else:
            return max_good_difficulty  # Stay at current level
    
    def _calculate_confidence_level(self, analytics: Dict) -> float:
        """Calculate user's confidence level based on recent performance trends."""
        daily_activity = analytics.get('daily_activity', [])
        
        if len(daily_activity) < 3:
            return 0.5  # Neutral confidence for insufficient data
        
        # Calculate trend in accuracy over recent days
        recent_accuracies = []
        for day in daily_activity[-7:]:  # Last 7 days
            attempted = day.get('problems_attempted', 0)
            solved = day.get('problems_solved', 0)
            if attempted > 0:
                recent_accuracies.append(solved / attempted)
        
        if len(recent_accuracies) < 2:
            return 0.5
        
        # Calculate trend (positive trend = increasing confidence)
        x = np.arange(len(recent_accuracies))
        trend = np.polyfit(x, recent_accuracies, 1)[0]  # Linear trend coefficient
        
        # Base confidence on average accuracy, adjusted by trend
        avg_accuracy = np.mean(recent_accuracies)
        confidence = avg_accuracy + (trend * 0.5)  # Trend adjustment
        
        return max(0.0, min(1.0, confidence))
    
    def generate_topic_recommendations(self, user_id: str, max_recommendations: int = 5) -> List[Recommendation]:
        """Generate personalized topic recommendations for the user."""
        try:
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                return []
            
            performance_analysis = self.analyze_user_performance(user_id)
            recommendations = []
            
            # Get current skill levels and completed topics
            skill_levels = profile.skill_levels
            completed_topics = profile.progress_metrics.topics_completed
            
            # 1. Recommend reinforcement for weak areas
            for weakness in performance_analysis.weaknesses:
                if weakness in skill_levels and skill_levels[weakness] > 1:
                    recommendations.append(Recommendation(
                        type=RecommendationType.PRACTICE_WEAK_AREAS.value,
                        content=f"Practice more {weakness} problems to strengthen understanding",
                        reason=f"Recent performance in {weakness} shows room for improvement",
                        priority=3,
                        estimated_time=60
                    ))
            
            # 2. Recommend next logical topics based on prerequisites
            available_topics = self._find_available_topics(skill_levels, completed_topics)
            for topic in available_topics[:2]:  # Top 2 available topics
                dependency = self.topic_dependencies[topic]
                recommendations.append(Recommendation(
                    type=RecommendationType.TOPIC_INTRODUCTION.value,
                    content=f"Start learning {topic.replace('_', ' ').title()}",
                    reason=f"You've mastered the prerequisites: {', '.join(dependency.prerequisites)}",
                    priority=2,
                    estimated_time=int(dependency.estimated_hours * 60)
                ))
            
            # 3. Recommend difficulty adjustments
            current_avg_difficulty = self._calculate_current_difficulty(skill_levels)
            recommended_difficulty = performance_analysis.recommended_difficulty
            
            if recommended_difficulty > current_avg_difficulty:
                recommendations.append(Recommendation(
                    type=RecommendationType.DIFFICULTY_INCREASE.value,
                    content=f"Try more challenging problems (Level {recommended_difficulty})",
                    reason="Your recent performance suggests you're ready for harder problems",
                    priority=2,
                    estimated_time=45
                ))
            elif recommended_difficulty < current_avg_difficulty:
                recommendations.append(Recommendation(
                    type=RecommendationType.DIFFICULTY_DECREASE.value,
                    content=f"Focus on Level {recommended_difficulty} problems to build confidence",
                    reason="Consider practicing easier problems to strengthen fundamentals",
                    priority=2,
                    estimated_time=30
                ))
            
            # 4. Recommend skill reinforcement for strong areas
            for strength in performance_analysis.strengths[:1]:  # Top strength
                recommendations.append(Recommendation(
                    type=RecommendationType.SKILL_REINFORCEMENT.value,
                    content=f"Continue practicing {strength} to maintain proficiency",
                    reason=f"You're performing well in {strength}",
                    priority=1,
                    estimated_time=30
                ))
            
            # Sort by priority (higher priority first) and limit results
            recommendations.sort(key=lambda x: x.priority, reverse=True)
            return recommendations[:max_recommendations]
            
        except Exception as e:
            logger.error(f"Failed to generate topic recommendations: {e}")
            return []
    
    def _find_available_topics(self, skill_levels: Dict[str, int], completed_topics: List[str]) -> List[str]:
        """Find topics that are available based on completed prerequisites."""
        available = []
        
        for topic, dependency in self.topic_dependencies.items():
            # Skip if already completed
            if topic in completed_topics:
                continue
            
            # Check if all prerequisites are met
            prerequisites_met = True
            for prereq in dependency.prerequisites:
                if prereq not in completed_topics and skill_levels.get(prereq, 0) < 2:
                    prerequisites_met = False
                    break
            
            if prerequisites_met:
                available.append(topic)
        
        # Sort by importance weight and difficulty
        available.sort(key=lambda t: (
            self.topic_dependencies[t].importance_weight,
            -self.topic_dependencies[t].difficulty_level
        ), reverse=True)
        
        return available
    
    def _calculate_current_difficulty(self, skill_levels: Dict[str, int]) -> float:
        """Calculate user's current average difficulty level."""
        if not skill_levels:
            return 1.0
        
        total_weighted_difficulty = 0
        total_weight = 0
        
        for domain, level in skill_levels.items():
            if domain in self.topic_dependencies:
                weight = self.topic_dependencies[domain].importance_weight
                total_weighted_difficulty += level * weight
                total_weight += weight
        
        return total_weighted_difficulty / max(total_weight, 1)
    
    def adjust_difficulty_dynamically(self, user_id: str, recent_performance: List[bool], 
                                    current_difficulty: int) -> int:
        """Dynamically adjust difficulty based on recent performance."""
        if len(recent_performance) < 3:
            return current_difficulty
        
        # Calculate recent accuracy
        recent_accuracy = sum(recent_performance) / len(recent_performance)
        
        # Calculate performance trend
        recent_trend = self._calculate_performance_trend(recent_performance)
        
        # Adjustment rules
        if recent_accuracy > 0.85 and recent_trend > 0:
            # Performing very well and improving - increase difficulty
            return min(4, current_difficulty + 1)
        elif recent_accuracy > 0.75 and len(recent_performance) >= 5:
            # Consistently good performance - slight increase
            return min(4, current_difficulty + 1) if sum(recent_performance[-5:]) >= 4 else current_difficulty
        elif recent_accuracy < 0.4 and recent_trend < 0:
            # Poor performance and declining - decrease difficulty
            return max(1, current_difficulty - 1)
        elif recent_accuracy < 0.5:
            # Struggling - consider decrease
            return max(1, current_difficulty - 1) if len(recent_performance) >= 5 else current_difficulty
        else:
            # Stable performance - maintain current difficulty
            return current_difficulty
    
    def _calculate_performance_trend(self, recent_performance: List[bool]) -> float:
        """Calculate trend in recent performance (-1 to 1)."""
        if len(recent_performance) < 3:
            return 0.0
        
        # Convert boolean to numeric and calculate trend
        numeric_performance = [1.0 if p else 0.0 for p in recent_performance]
        x = np.arange(len(numeric_performance))
        
        try:
            trend = np.polyfit(x, numeric_performance, 1)[0]
            return max(-1.0, min(1.0, trend * len(numeric_performance)))
        except:
            return 0.0
    
    def generate_learning_path(self, user_id: str, target_topics: List[str] = None) -> LearningPath:
        """Generate a complete personalized learning path for the user."""
        try:
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                raise ValueError(f"User profile not found: {user_id}")
            
            performance_analysis = self.analyze_user_performance(user_id)
            
            # Determine target topics
            if not target_topics:
                # Use user's learning goals or default progression
                target_topics = profile.learning_goals or ['calculus_1', 'linear_algebra']
            
            # Find current topic based on recent activity
            current_topic = self._determine_current_topic(profile, performance_analysis)
            
            # Generate next topics in optimal order
            next_topics = self._plan_topic_sequence(profile, target_topics)
            
            # Generate recommendations
            recommendations = self.generate_topic_recommendations(user_id)
            
            # Estimate completion time
            estimated_completion_time = self._estimate_completion_time(next_topics, performance_analysis)
            
            # Calculate progress percentage
            progress_percentage = self._calculate_progress_percentage(profile, target_topics)
            
            return LearningPath(
                user_id=user_id,
                current_topic=current_topic,
                next_topics=next_topics,
                recommendations=recommendations,
                estimated_completion_time=estimated_completion_time,
                progress_percentage=progress_percentage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate learning path: {e}")
            # Return default learning path
            return LearningPath(
                user_id=user_id,
                current_topic='basic_algebra',
                next_topics=['advanced_algebra', 'precalculus'],
                recommendations=[],
                estimated_completion_time=30,
                progress_percentage=0.0
            )
    
    def _determine_current_topic(self, profile: UserProfile, analysis: PerformanceAnalysis) -> str:
        """Determine the user's current focus topic."""
        # Find the most advanced topic the user is actively working on
        skill_levels = profile.skill_levels
        
        # Look for topics with skill level 2-3 (actively learning)
        active_topics = [topic for topic, level in skill_levels.items() 
                        if 2 <= level <= 3 and topic in self.topic_dependencies]
        
        if active_topics:
            # Return the most advanced active topic
            return max(active_topics, key=lambda t: self.topic_dependencies[t].difficulty_level)
        
        # If no active topics, find the most advanced completed topic
        completed_topics = profile.progress_metrics.topics_completed
        if completed_topics:
            completed_in_deps = [t for t in completed_topics if t in self.topic_dependencies]
            if completed_in_deps:
                return max(completed_in_deps, key=lambda t: self.topic_dependencies[t].difficulty_level)
        
        # Default to basic algebra
        return 'basic_algebra'
    
    def _plan_topic_sequence(self, profile: UserProfile, target_topics: List[str]) -> List[str]:
        """Plan the optimal sequence of topics to reach targets."""
        completed = set(profile.progress_metrics.topics_completed)
        skill_levels = profile.skill_levels
        
        # Find all topics needed to reach targets (including prerequisites)
        needed_topics = set()
        for target in target_topics:
            needed_topics.update(self._get_all_prerequisites(target))
            needed_topics.add(target)
        
        # Remove already completed topics
        remaining_topics = needed_topics - completed
        
        # Remove topics where user already has high skill level
        remaining_topics = {t for t in remaining_topics 
                          if skill_levels.get(t, 0) < 4}
        
        # Sort by dependency order and difficulty
        sorted_topics = self._topological_sort(remaining_topics)
        
        return sorted_topics[:10]  # Limit to next 10 topics
    
    def _get_all_prerequisites(self, topic: str) -> List[str]:
        """Get all prerequisites for a topic (recursive)."""
        if topic not in self.topic_dependencies:
            return []
        
        all_prereqs = []
        direct_prereqs = self.topic_dependencies[topic].prerequisites
        
        for prereq in direct_prereqs:
            all_prereqs.append(prereq)
            all_prereqs.extend(self._get_all_prerequisites(prereq))
        
        return list(set(all_prereqs))  # Remove duplicates
    
    def _topological_sort(self, topics: set) -> List[str]:
        """Sort topics in dependency order using topological sort."""
        # Build adjacency list for topics in the set
        graph = {topic: [] for topic in topics}
        in_degree = {topic: 0 for topic in topics}
        
        for topic in topics:
            if topic in self.topic_dependencies:
                for prereq in self.topic_dependencies[topic].prerequisites:
                    if prereq in topics:
                        graph[prereq].append(topic)
                        in_degree[topic] += 1
        
        # Kahn's algorithm
        queue = [topic for topic in topics if in_degree[topic] == 0]
        result = []
        
        while queue:
            # Sort queue by difficulty and importance for consistent ordering
            queue.sort(key=lambda t: (
                self.topic_dependencies.get(t, TopicDependency('', [], 1, 1, 0)).difficulty_level,
                -self.topic_dependencies.get(t, TopicDependency('', [], 1, 1, 0)).importance_weight
            ))
            
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result
    
    def _estimate_completion_time(self, topics: List[str], analysis: PerformanceAnalysis) -> int:
        """Estimate completion time in days for the learning path."""
        total_hours = sum(
            self.topic_dependencies.get(topic, TopicDependency('', [], 1, 1, 0)).estimated_hours
            for topic in topics
        )
        
        # Adjust based on user's learning velocity and confidence
        velocity_factor = max(0.5, min(2.0, analysis.learning_velocity / 2.0))
        confidence_factor = max(0.7, min(1.3, analysis.confidence_level + 0.5))
        
        adjusted_hours = total_hours / (velocity_factor * confidence_factor)
        
        # Assume 1 hour of study per day on average
        return max(1, int(adjusted_hours))
    
    def _calculate_progress_percentage(self, profile: UserProfile, target_topics: List[str]) -> float:
        """Calculate progress percentage towards target topics."""
        if not target_topics:
            return 0.0
        
        completed_topics = set(profile.progress_metrics.topics_completed)
        skill_levels = profile.skill_levels
        
        total_progress = 0.0
        total_weight = 0.0
        
        for target in target_topics:
            # Get all topics needed for this target
            needed_topics = self._get_all_prerequisites(target) + [target]
            
            target_progress = 0.0
            target_weight = 0.0
            
            for topic in needed_topics:
                if topic in self.topic_dependencies:
                    weight = self.topic_dependencies[topic].importance_weight
                    
                    if topic in completed_topics:
                        topic_progress = 1.0
                    else:
                        # Progress based on skill level (0-4 scale)
                        topic_progress = skill_levels.get(topic, 0) / 4.0
                    
                    target_progress += topic_progress * weight
                    target_weight += weight
            
            if target_weight > 0:
                total_progress += target_progress / target_weight
                total_weight += 1.0
        
        return (total_progress / max(total_weight, 1)) * 100.0


# Utility functions for learning path optimization
def optimize_study_schedule(learning_path: LearningPath, available_hours_per_day: float = 1.0) -> Dict[str, Any]:
    """Optimize study schedule based on learning path and available time."""
    topics = learning_path.next_topics
    recommendations = learning_path.recommendations
    
    # Create daily schedule
    daily_schedule = []
    current_day = 0
    
    for topic in topics[:5]:  # Focus on next 5 topics
        # Estimate days needed for this topic (simplified)
        topic_hours = 8.0  # Default estimate
        days_needed = max(1, int(topic_hours / available_hours_per_day))
        
        daily_schedule.append({
            'day': current_day + 1,
            'topic': topic,
            'duration_days': days_needed,
            'focus_areas': [rec.content for rec in recommendations if topic in rec.content][:2]
        })
        
        current_day += days_needed
    
    return {
        'total_days': current_day,
        'daily_schedule': daily_schedule,
        'study_hours_per_day': available_hours_per_day,
        'completion_date': (datetime.now() + timedelta(days=current_day)).strftime('%Y-%m-%d')
    }


def calculate_learning_efficiency(user_id: str, progress_db: UserProgressDatabase, days: int = 30) -> float:
    """Calculate user's learning efficiency score (0-1)."""
    try:
        analytics = progress_db.get_learning_analytics(user_id, days)
        
        # Factors for efficiency calculation
        total_time = sum(day.get('avg_time', 0) * day.get('problems_attempted', 0) 
                        for day in analytics['daily_activity'])
        total_solved = sum(day.get('problems_solved', 0) for day in analytics['daily_activity'])
        
        if total_time == 0:
            return 0.0
        
        # Problems solved per minute
        efficiency = (total_solved / (total_time / 60)) * 10  # Scale factor
        
        return max(0.0, min(1.0, efficiency))
        
    except Exception as e:
        logger.error(f"Failed to calculate learning efficiency: {e}")
        return 0.0