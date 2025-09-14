"""
User progress tracking and personalization data models.
Implements database operations for user profiles, skill tracking, and learning analytics.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import psycopg2
from psycopg2.extras import RealDictCursor
import os

from shared.models.core import (
    UserProfile, UserPreferences, ProgressMetrics, MathDomain,
    DifficultyLevel, Performance, LearningPath, Recommendation
)

logger = logging.getLogger(__name__)


@dataclass
class SkillAssessment:
    """Assessment of user skill in a specific domain."""
    domain: str
    current_level: int  # 1-4 scale
    mastery_percentage: float  # 0.0-1.0
    recent_performance: float  # 0.0-1.0
    practice_count: int
    last_practiced: datetime
    time_spent: int  # minutes
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class LearningGoal:
    """User-defined learning goal with progress tracking."""
    id: str
    user_id: str
    title: str
    description: str
    target_domains: List[str]
    target_level: int
    deadline: Optional[datetime]
    progress_percentage: float
    created_at: datetime
    is_active: bool


@dataclass
class StudySession:
    """Individual study session tracking."""
    id: str
    user_id: str
    started_at: datetime
    ended_at: Optional[datetime]
    problems_attempted: int
    problems_solved: int
    domains_practiced: List[str]
    total_time: int  # minutes
    accuracy: float


class UserProgressDatabase:
    """Database interface for user progress and profile management."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize database connection."""
        self.connection_string = connection_string or os.getenv(
            'DATABASE_URL', 
            'postgresql://user:password@localhost:5432/math_tutor'
        )
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
            self.connection.autocommit = True
            logger.info("Connected to user progress database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _execute_query(self, query: str, params: Tuple = None) -> List[Dict]:
        """Execute a database query and return results."""
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                if cursor.description:
                    return [dict(row) for row in cursor.fetchall()]
                return []
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def create_user_profile(self, user_id: str, username: str, email: str) -> UserProfile:
        """Create a new user profile with default settings."""
        default_preferences = UserPreferences(
            preferred_explanation_level="standard",
            visual_learning=True,
            step_by_step_detail="standard",
            notification_settings={
                "quiz_reminders": True,
                "progress_updates": True,
                "achievement_notifications": True
            }
        )
        
        default_metrics = ProgressMetrics(
            total_problems_solved=0,
            current_streak=0,
            longest_streak=0,
            average_accuracy=0.0,
            time_spent_learning=0,
            topics_completed=[],
            skill_levels={domain.value: 1 for domain in MathDomain}
        )
        
        # Insert user profile into database
        query = """
        INSERT INTO user_profiles (
            user_id, skill_levels, learning_goals, preferences,
            total_problems_solved, current_streak, longest_streak,
            average_accuracy, time_spent_learning
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (user_id) DO UPDATE SET
            updated_at = NOW()
        """
        
        params = (
            user_id,
            json.dumps(default_metrics.skill_levels),
            [],
            json.dumps(asdict(default_preferences)),
            0, 0, 0, 0.0, 0
        )
        
        self._execute_query(query, params)
        
        # Initialize learning progress for each domain
        for domain in MathDomain:
            self._initialize_domain_progress(user_id, domain.value)
        
        return UserProfile(
            id=user_id,
            username=username,
            email=email,
            skill_levels=default_metrics.skill_levels,
            learning_goals=[],
            preferences=default_preferences,
            progress_metrics=default_metrics,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    def _initialize_domain_progress(self, user_id: str, domain: str):
        """Initialize learning progress for a specific domain."""
        query = """
        INSERT INTO learning_progress (user_id, topic, mastery_level, practice_count, total_time_spent)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (user_id, topic) DO NOTHING
        """
        self._execute_query(query, (user_id, domain, 0.0, 0, 0))
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve complete user profile."""
        query = """
        SELECT up.*, u.username, u.email, u.created_at
        FROM user_profiles up
        JOIN users u ON up.user_id = u.id
        WHERE up.user_id = %s
        """
        
        result = self._execute_query(query, (user_id,))
        if not result:
            return None
        
        profile_data = result[0]
        
        preferences = UserPreferences(**json.loads(profile_data['preferences']))
        
        metrics = ProgressMetrics(
            total_problems_solved=profile_data['total_problems_solved'],
            current_streak=profile_data['current_streak'],
            longest_streak=profile_data['longest_streak'],
            average_accuracy=float(profile_data['average_accuracy']),
            time_spent_learning=profile_data['time_spent_learning'],
            topics_completed=self._get_completed_topics(user_id),
            skill_levels=json.loads(profile_data['skill_levels'])
        )
        
        return UserProfile(
            id=user_id,
            username=profile_data['username'],
            email=profile_data['email'],
            skill_levels=json.loads(profile_data['skill_levels']),
            learning_goals=profile_data['learning_goals'] or [],
            preferences=preferences,
            progress_metrics=metrics,
            created_at=profile_data['created_at'],
            updated_at=profile_data['updated_at']
        )
    
    def update_skill_level(self, user_id: str, domain: str, new_level: int) -> bool:
        """Update user's skill level in a specific domain."""
        if not (1 <= new_level <= 4):
            raise ValueError("Skill level must be between 1 and 4")
        
        # Get current skill levels
        profile = self.get_user_profile(user_id)
        if not profile:
            return False
        
        # Update skill level
        profile.skill_levels[domain] = new_level
        
        # Update database
        query = """
        UPDATE user_profiles 
        SET skill_levels = %s, updated_at = NOW()
        WHERE user_id = %s
        """
        
        self._execute_query(query, (json.dumps(profile.skill_levels), user_id))
        return True
    
    def track_problem_attempt(self, user_id: str, domain: str, difficulty: int, 
                            is_correct: bool, time_taken: int, hints_used: int = 0) -> bool:
        """Track a problem attempt and update user progress."""
        try:
            # Update problem attempt statistics
            self._update_attempt_statistics(user_id, is_correct, time_taken)
            
            # Update domain-specific progress
            self._update_domain_progress(user_id, domain, is_correct, time_taken)
            
            # Update streak information
            self._update_streak(user_id, is_correct)
            
            # Assess if skill level should be updated
            self._assess_skill_level_change(user_id, domain, difficulty, is_correct)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to track problem attempt: {e}")
            return False
    
    def _update_attempt_statistics(self, user_id: str, is_correct: bool, time_taken: int):
        """Update overall attempt statistics."""
        query = """
        UPDATE user_profiles 
        SET 
            total_problems_solved = total_problems_solved + 1,
            average_accuracy = (
                (average_accuracy * total_problems_solved + %s) / 
                (total_problems_solved + 1)
            ),
            time_spent_learning = time_spent_learning + %s,
            updated_at = NOW()
        WHERE user_id = %s
        """
        
        accuracy_contribution = 1.0 if is_correct else 0.0
        time_minutes = max(1, time_taken // 60)  # Convert to minutes, minimum 1
        
        self._execute_query(query, (accuracy_contribution, time_minutes, user_id))
    
    def _update_domain_progress(self, user_id: str, domain: str, is_correct: bool, time_taken: int):
        """Update progress in a specific domain."""
        # Get current domain progress
        query = """
        SELECT mastery_level, practice_count, total_time_spent
        FROM learning_progress
        WHERE user_id = %s AND topic = %s
        """
        
        result = self._execute_query(query, (user_id, domain))
        if not result:
            self._initialize_domain_progress(user_id, domain)
            current_mastery = 0.0
            practice_count = 0
            total_time = 0
        else:
            current_mastery = float(result[0]['mastery_level'])
            practice_count = result[0]['practice_count']
            total_time = result[0]['total_time_spent']
        
        # Calculate new mastery level using exponential moving average
        learning_rate = 0.1
        performance_score = 1.0 if is_correct else 0.0
        new_mastery = current_mastery + learning_rate * (performance_score - current_mastery)
        new_mastery = max(0.0, min(1.0, new_mastery))  # Clamp to [0, 1]
        
        # Update domain progress
        update_query = """
        UPDATE learning_progress
        SET 
            mastery_level = %s,
            practice_count = %s,
            total_time_spent = %s,
            last_practiced = NOW()
        WHERE user_id = %s AND topic = %s
        """
        
        time_minutes = max(1, time_taken // 60)
        self._execute_query(update_query, (
            new_mastery, practice_count + 1, total_time + time_minutes, user_id, domain
        ))
    
    def _update_streak(self, user_id: str, is_correct: bool):
        """Update user's current and longest streak."""
        if is_correct:
            query = """
            UPDATE user_profiles
            SET 
                current_streak = current_streak + 1,
                longest_streak = GREATEST(longest_streak, current_streak + 1),
                updated_at = NOW()
            WHERE user_id = %s
            """
        else:
            query = """
            UPDATE user_profiles
            SET 
                current_streak = 0,
                updated_at = NOW()
            WHERE user_id = %s
            """
        
        self._execute_query(query, (user_id,))
    
    def _assess_skill_level_change(self, user_id: str, domain: str, difficulty: int, is_correct: bool):
        """Assess whether user's skill level should change based on performance."""
        # Get recent performance in this domain
        query = """
        SELECT 
            AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as recent_accuracy,
            COUNT(*) as attempt_count
        FROM problem_attempts pa
        JOIN problems p ON pa.problem_id = p.id
        WHERE pa.user_id = %s AND p.domain = %s 
        AND pa.attempt_timestamp > NOW() - INTERVAL '7 days'
        """
        
        result = self._execute_query(query, (user_id, domain))
        if not result or result[0]['attempt_count'] < 5:
            return  # Need at least 5 recent attempts
        
        recent_accuracy = float(result[0]['recent_accuracy'])
        current_level = self.get_user_profile(user_id).skill_levels.get(domain, 1)
        
        # Level up criteria: >80% accuracy on current level problems
        if recent_accuracy > 0.8 and current_level < 4:
            self.update_skill_level(user_id, domain, current_level + 1)
            logger.info(f"User {user_id} leveled up in {domain} to level {current_level + 1}")
        
        # Level down criteria: <40% accuracy on current level problems
        elif recent_accuracy < 0.4 and current_level > 1:
            self.update_skill_level(user_id, domain, current_level - 1)
            logger.info(f"User {user_id} leveled down in {domain} to level {current_level - 1}")
    
    def get_skill_assessment(self, user_id: str, domain: str) -> Optional[SkillAssessment]:
        """Get detailed skill assessment for a specific domain."""
        # Get domain progress
        query = """
        SELECT mastery_level, practice_count, total_time_spent, last_practiced
        FROM learning_progress
        WHERE user_id = %s AND topic = %s
        """
        
        result = self._execute_query(query, (user_id, domain))
        if not result:
            return None
        
        progress = result[0]
        
        # Get recent performance
        recent_query = """
        SELECT 
            AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) as accuracy,
            COUNT(*) as attempts
        FROM problem_attempts pa
        JOIN problems p ON pa.problem_id = p.id
        WHERE pa.user_id = %s AND p.domain = %s 
        AND pa.attempt_timestamp > NOW() - INTERVAL '14 days'
        """
        
        recent_result = self._execute_query(recent_query, (user_id, domain))
        recent_performance = float(recent_result[0]['accuracy']) if recent_result[0]['attempts'] > 0 else 0.0
        
        # Get current skill level
        profile = self.get_user_profile(user_id)
        current_level = profile.skill_levels.get(domain, 1)
        
        # Analyze strengths and weaknesses (simplified)
        strengths = []
        weaknesses = []
        
        if recent_performance > 0.7:
            strengths.append("Consistent problem solving")
        if progress['practice_count'] > 20:
            strengths.append("Regular practice")
        if recent_performance < 0.5:
            weaknesses.append("Accuracy needs improvement")
        if progress['practice_count'] < 5:
            weaknesses.append("Needs more practice")
        
        return SkillAssessment(
            domain=domain,
            current_level=current_level,
            mastery_percentage=float(progress['mastery_level']),
            recent_performance=recent_performance,
            practice_count=progress['practice_count'],
            last_practiced=progress['last_practiced'],
            time_spent=progress['total_time_spent'],
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _get_completed_topics(self, user_id: str) -> List[str]:
        """Get list of topics the user has completed (mastery > 0.8)."""
        query = """
        SELECT topic
        FROM learning_progress
        WHERE user_id = %s AND mastery_level > 0.8
        """
        
        result = self._execute_query(query, (user_id,))
        return [row['topic'] for row in result]
    
    def get_learning_analytics(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive learning analytics for a user."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Daily activity
        activity_query = """
        SELECT 
            DATE(attempt_timestamp) as date,
            COUNT(*) as problems_attempted,
            SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as problems_solved,
            AVG(time_taken) as avg_time
        FROM problem_attempts
        WHERE user_id = %s AND attempt_timestamp >= %s
        GROUP BY DATE(attempt_timestamp)
        ORDER BY date
        """
        
        activity_data = self._execute_query(activity_query, (user_id, start_date))
        
        # Domain performance
        domain_query = """
        SELECT 
            p.domain,
            COUNT(*) as attempts,
            AVG(CASE WHEN pa.is_correct THEN 1.0 ELSE 0.0 END) as accuracy,
            AVG(pa.time_taken) as avg_time
        FROM problem_attempts pa
        JOIN problems p ON pa.problem_id = p.id
        WHERE pa.user_id = %s AND pa.attempt_timestamp >= %s
        GROUP BY p.domain
        """
        
        domain_data = self._execute_query(domain_query, (user_id, start_date))
        
        # Difficulty progression
        difficulty_query = """
        SELECT 
            p.difficulty_level,
            COUNT(*) as attempts,
            AVG(CASE WHEN pa.is_correct THEN 1.0 ELSE 0.0 END) as accuracy
        FROM problem_attempts pa
        JOIN problems p ON pa.problem_id = p.id
        WHERE pa.user_id = %s AND pa.attempt_timestamp >= %s
        GROUP BY p.difficulty_level
        ORDER BY p.difficulty_level
        """
        
        difficulty_data = self._execute_query(difficulty_query, (user_id, start_date))
        
        return {
            'period_days': days,
            'daily_activity': activity_data,
            'domain_performance': domain_data,
            'difficulty_progression': difficulty_data,
            'generated_at': datetime.now().isoformat()
        }
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")


# Utility functions for progress analysis
def calculate_learning_velocity(user_id: str, db: UserProgressDatabase, days: int = 7) -> float:
    """Calculate user's learning velocity (problems solved per day)."""
    analytics = db.get_learning_analytics(user_id, days)
    daily_activity = analytics['daily_activity']
    
    if not daily_activity:
        return 0.0
    
    total_solved = sum(day['problems_solved'] for day in daily_activity)
    return total_solved / days


def identify_learning_patterns(user_id: str, db: UserProgressDatabase) -> Dict[str, Any]:
    """Identify patterns in user's learning behavior."""
    analytics = db.get_learning_analytics(user_id, 30)
    
    # Find peak performance times (simplified)
    domain_performance = analytics['domain_performance']
    best_domain = max(domain_performance, key=lambda x: x['accuracy']) if domain_performance else None
    worst_domain = min(domain_performance, key=lambda x: x['accuracy']) if domain_performance else None
    
    # Calculate consistency
    daily_activity = analytics['daily_activity']
    if len(daily_activity) > 1:
        accuracies = [day.get('problems_solved', 0) / max(day.get('problems_attempted', 1), 1) 
                     for day in daily_activity]
        consistency = 1.0 - (max(accuracies) - min(accuracies)) if accuracies else 0.0
    else:
        consistency = 0.0
    
    return {
        'best_domain': best_domain['domain'] if best_domain else None,
        'worst_domain': worst_domain['domain'] if worst_domain else None,
        'consistency_score': consistency,
        'learning_velocity': calculate_learning_velocity(user_id, db),
        'total_practice_days': len(daily_activity)
    }