"""
Tests for user progress tracking and personalization data models.
Tests database operations, skill tracking, and learning analytics.
"""

import pytest
import json
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import psycopg2

from user_progress import (
    UserProgressDatabase, SkillAssessment, LearningGoal, StudySession,
    calculate_learning_velocity, identify_learning_patterns
)
from shared.models.core import (
    UserProfile, UserPreferences, ProgressMetrics, MathDomain
)


class TestUserProgressDatabase:
    """Test suite for UserProgressDatabase class."""
    
    @pytest.fixture
    def mock_connection(self):
        """Mock database connection."""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []
        mock_cursor.description = None  # For UPDATE queries
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_cursor)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_conn.cursor.return_value = mock_context_manager
        return mock_conn, mock_cursor
    
    @pytest.fixture
    def db_instance(self, mock_connection):
        """Create UserProgressDatabase instance with mocked connection."""
        mock_conn, mock_cursor = mock_connection
        
        with patch('psycopg2.connect', return_value=mock_conn):
            db = UserProgressDatabase("test://connection")
            return db, mock_cursor
    
    def test_create_user_profile(self, db_instance):
        """Test creating a new user profile with default settings."""
        db, mock_cursor = db_instance
        mock_cursor.fetchall.return_value = []
        
        user_id = "test-user-123"
        username = "testuser"
        email = "test@example.com"
        
        profile = db.create_user_profile(user_id, username, email)
        
        # Verify profile structure
        assert profile.id == user_id
        assert profile.username == username
        assert profile.email == email
        assert isinstance(profile.preferences, UserPreferences)
        assert isinstance(profile.progress_metrics, ProgressMetrics)
        
        # Verify default skill levels
        for domain in MathDomain:
            assert profile.skill_levels[domain.value] == 1
        
        # Verify database calls
        assert mock_cursor.execute.call_count >= 1  # Profile insert + domain initializations
    
    def test_get_user_profile_existing(self, db_instance):
        """Test retrieving an existing user profile."""
        db, mock_cursor = db_instance
        
        # Mock database response
        mock_profile_data = {
            'user_id': 'test-user-123',
            'username': 'testuser',
            'email': 'test@example.com',
            'skill_levels': json.dumps({'algebra': 2, 'calculus': 1}),
            'learning_goals': ['Master calculus'],
            'preferences': json.dumps({
                'preferred_explanation_level': 'detailed',
                'visual_learning': True,
                'step_by_step_detail': 'detailed',
                'notification_settings': {'quiz_reminders': True}
            }),
            'total_problems_solved': 25,
            'current_streak': 5,
            'longest_streak': 10,
            'average_accuracy': 0.85,
            'time_spent_learning': 120,
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        }
        
        mock_cursor.fetchall.return_value = [mock_profile_data]
        
        # Mock completed topics query
        with patch.object(db, '_get_completed_topics', return_value=['algebra']):
            profile = db.get_user_profile('test-user-123')
        
        assert profile is not None
        assert profile.username == 'testuser'
        assert profile.skill_levels['algebra'] == 2
        assert profile.progress_metrics.total_problems_solved == 25
        assert profile.progress_metrics.current_streak == 5
    
    def test_get_user_profile_nonexistent(self, db_instance):
        """Test retrieving a non-existent user profile."""
        db, mock_cursor = db_instance
        mock_cursor.fetchall.return_value = []
        
        profile = db.get_user_profile('nonexistent-user')
        assert profile is None
    
    def test_update_skill_level_valid(self, db_instance):
        """Test updating skill level with valid input."""
        db, mock_cursor = db_instance
        
        # Mock existing profile
        mock_profile = Mock()
        mock_profile.skill_levels = {'algebra': 1, 'calculus': 1}
        
        with patch.object(db, 'get_user_profile', return_value=mock_profile):
            result = db.update_skill_level('test-user', 'algebra', 2)
        
        assert result is True
        mock_cursor.execute.assert_called()
    
    def test_update_skill_level_invalid(self, db_instance):
        """Test updating skill level with invalid input."""
        db, mock_cursor = db_instance
        
        with pytest.raises(ValueError, match="Skill level must be between 1 and 4"):
            db.update_skill_level('test-user', 'algebra', 5)
        
        with pytest.raises(ValueError, match="Skill level must be between 1 and 4"):
            db.update_skill_level('test-user', 'algebra', 0)
    
    def test_track_problem_attempt_correct(self, db_instance):
        """Test tracking a correct problem attempt."""
        db, mock_cursor = db_instance
        
        # Mock the internal methods
        with patch.object(db, '_update_attempt_statistics') as mock_stats, \
             patch.object(db, '_update_domain_progress') as mock_domain, \
             patch.object(db, '_update_streak') as mock_streak, \
             patch.object(db, '_assess_skill_level_change') as mock_assess:
            
            result = db.track_problem_attempt(
                user_id='test-user',
                domain='algebra',
                difficulty=2,
                is_correct=True,
                time_taken=120,
                hints_used=1
            )
        
        assert result is True
        mock_stats.assert_called_once_with('test-user', True, 120)
        mock_domain.assert_called_once_with('test-user', 'algebra', True, 120)
        mock_streak.assert_called_once_with('test-user', True)
        mock_assess.assert_called_once_with('test-user', 'algebra', 2, True)
    
    def test_track_problem_attempt_incorrect(self, db_instance):
        """Test tracking an incorrect problem attempt."""
        db, mock_cursor = db_instance
        
        with patch.object(db, '_update_attempt_statistics') as mock_stats, \
             patch.object(db, '_update_domain_progress') as mock_domain, \
             patch.object(db, '_update_streak') as mock_streak, \
             patch.object(db, '_assess_skill_level_change') as mock_assess:
            
            result = db.track_problem_attempt(
                user_id='test-user',
                domain='calculus',
                difficulty=3,
                is_correct=False,
                time_taken=300
            )
        
        assert result is True
        mock_stats.assert_called_once_with('test-user', False, 300)
        mock_domain.assert_called_once_with('test-user', 'calculus', False, 300)
        mock_streak.assert_called_once_with('test-user', False)
        mock_assess.assert_called_once_with('test-user', 'calculus', 3, False)
    
    def test_update_streak_correct_answer(self, db_instance):
        """Test streak update for correct answer."""
        db, mock_cursor = db_instance
        
        db._update_streak('test-user', True)
        
        # Verify the correct SQL was executed
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert 'current_streak = current_streak + 1' in call_args[0]
        assert 'longest_streak = GREATEST' in call_args[0]
    
    def test_update_streak_incorrect_answer(self, db_instance):
        """Test streak update for incorrect answer."""
        db, mock_cursor = db_instance
        
        db._update_streak('test-user', False)
        
        # Verify the correct SQL was executed
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args[0]
        assert 'current_streak = 0' in call_args[0]
    
    def test_update_domain_progress_new_domain(self, db_instance):
        """Test updating progress for a new domain."""
        db, mock_cursor = db_instance
        
        # Mock no existing progress
        mock_cursor.fetchall.return_value = []
        
        with patch.object(db, '_initialize_domain_progress') as mock_init:
            db._update_domain_progress('test-user', 'algebra', True, 120)
        
        mock_init.assert_called_once_with('test-user', 'algebra')
        # Should execute update query
        assert mock_cursor.execute.call_count >= 2  # Select + Update
    
    def test_update_domain_progress_existing_domain(self, db_instance):
        """Test updating progress for existing domain."""
        db, mock_cursor = db_instance
        
        # Mock existing progress
        mock_cursor.fetchall.return_value = [{
            'mastery_level': 0.6,
            'practice_count': 10,
            'total_time_spent': 60
        }]
        
        db._update_domain_progress('test-user', 'algebra', True, 120)
        
        # Should execute select and update queries
        assert mock_cursor.execute.call_count == 2
    
    def test_assess_skill_level_change_level_up(self, db_instance):
        """Test skill level assessment for level up scenario."""
        db, mock_cursor = db_instance
        
        # Mock high recent accuracy
        mock_cursor.fetchall.return_value = [{
            'recent_accuracy': 0.85,
            'attempt_count': 10
        }]
        
        # Mock current profile
        mock_profile = Mock()
        mock_profile.skill_levels = {'algebra': 2}
        
        with patch.object(db, 'get_user_profile', return_value=mock_profile), \
             patch.object(db, 'update_skill_level') as mock_update:
            
            db._assess_skill_level_change('test-user', 'algebra', 2, True)
            
            mock_update.assert_called_once_with('test-user', 'algebra', 3)
    
    def test_assess_skill_level_change_level_down(self, db_instance):
        """Test skill level assessment for level down scenario."""
        db, mock_cursor = db_instance
        
        # Mock low recent accuracy
        mock_cursor.fetchall.return_value = [{
            'recent_accuracy': 0.35,
            'attempt_count': 8
        }]
        
        # Mock current profile
        mock_profile = Mock()
        mock_profile.skill_levels = {'algebra': 3}
        
        with patch.object(db, 'get_user_profile', return_value=mock_profile), \
             patch.object(db, 'update_skill_level') as mock_update:
            
            db._assess_skill_level_change('test-user', 'algebra', 3, False)
            
            mock_update.assert_called_once_with('test-user', 'algebra', 2)
    
    def test_assess_skill_level_change_insufficient_data(self, db_instance):
        """Test skill level assessment with insufficient data."""
        db, mock_cursor = db_instance
        
        # Mock insufficient attempts
        mock_cursor.fetchall.return_value = [{
            'recent_accuracy': 0.9,
            'attempt_count': 3
        }]
        
        with patch.object(db, 'update_skill_level') as mock_update:
            db._assess_skill_level_change('test-user', 'algebra', 2, True)
            
            # Should not update skill level
            mock_update.assert_not_called()
    
    def test_get_skill_assessment(self, db_instance):
        """Test getting detailed skill assessment."""
        db, mock_cursor = db_instance
        
        # Mock domain progress data
        progress_data = {
            'mastery_level': 0.75,
            'practice_count': 25,
            'total_time_spent': 180,
            'last_practiced': datetime.now()
        }
        
        # Mock recent performance data
        recent_data = {
            'accuracy': 0.8,
            'attempts': 15
        }
        
        mock_cursor.fetchall.side_effect = [[progress_data], [recent_data]]
        
        # Mock user profile
        mock_profile = Mock()
        mock_profile.skill_levels = {'algebra': 3}
        
        with patch.object(db, 'get_user_profile', return_value=mock_profile):
            assessment = db.get_skill_assessment('test-user', 'algebra')
        
        assert assessment is not None
        assert assessment.domain == 'algebra'
        assert assessment.current_level == 3
        assert assessment.mastery_percentage == 0.75
        assert assessment.recent_performance == 0.8
        assert assessment.practice_count == 25
        assert len(assessment.strengths) > 0
    
    def test_get_learning_analytics(self, db_instance):
        """Test getting comprehensive learning analytics."""
        db, mock_cursor = db_instance
        
        # Mock analytics data
        activity_data = [
            {'date': '2024-01-01', 'problems_attempted': 5, 'problems_solved': 4, 'avg_time': 120},
            {'date': '2024-01-02', 'problems_attempted': 3, 'problems_solved': 3, 'avg_time': 90}
        ]
        
        domain_data = [
            {'domain': 'algebra', 'attempts': 8, 'accuracy': 0.8, 'avg_time': 105},
            {'domain': 'calculus', 'attempts': 3, 'accuracy': 0.6, 'avg_time': 150}
        ]
        
        difficulty_data = [
            {'difficulty_level': 1, 'attempts': 5, 'accuracy': 0.9},
            {'difficulty_level': 2, 'attempts': 6, 'accuracy': 0.7}
        ]
        
        mock_cursor.fetchall.side_effect = [activity_data, domain_data, difficulty_data]
        
        analytics = db.get_learning_analytics('test-user', 7)
        
        assert analytics['period_days'] == 7
        assert len(analytics['daily_activity']) == 2
        assert len(analytics['domain_performance']) == 2
        assert len(analytics['difficulty_progression']) == 2
        assert 'generated_at' in analytics


class TestSkillAssessment:
    """Test suite for SkillAssessment data class."""
    
    def test_skill_assessment_creation(self):
        """Test creating a SkillAssessment instance."""
        assessment = SkillAssessment(
            domain='algebra',
            current_level=2,
            mastery_percentage=0.75,
            recent_performance=0.8,
            practice_count=20,
            last_practiced=datetime.now(),
            time_spent=120,
            strengths=['Problem solving', 'Consistency'],
            weaknesses=['Speed']
        )
        
        assert assessment.domain == 'algebra'
        assert assessment.current_level == 2
        assert assessment.mastery_percentage == 0.75
        assert len(assessment.strengths) == 2
        assert len(assessment.weaknesses) == 1


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_calculate_learning_velocity(self):
        """Test learning velocity calculation."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': [
                {'problems_solved': 5},
                {'problems_solved': 3},
                {'problems_solved': 4}
            ]
        }
        
        velocity = calculate_learning_velocity('test-user', mock_db, 3)
        assert velocity == 4.0  # (5 + 3 + 4) / 3
    
    def test_calculate_learning_velocity_no_data(self):
        """Test learning velocity calculation with no data."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': []
        }
        
        velocity = calculate_learning_velocity('test-user', mock_db, 7)
        assert velocity == 0.0
    
    def test_identify_learning_patterns(self):
        """Test learning pattern identification."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'domain_performance': [
                {'domain': 'algebra', 'accuracy': 0.8},
                {'domain': 'calculus', 'accuracy': 0.6}
            ],
            'daily_activity': [
                {'problems_attempted': 5, 'problems_solved': 4},
                {'problems_attempted': 3, 'problems_solved': 2}
            ]
        }
        
        with patch('user_progress.calculate_learning_velocity', return_value=2.5):
            patterns = identify_learning_patterns('test-user', mock_db)
        
        assert patterns['best_domain'] == 'algebra'
        assert patterns['worst_domain'] == 'calculus'
        assert patterns['learning_velocity'] == 2.5
        assert patterns['total_practice_days'] == 2
        assert 'consistency_score' in patterns


class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    def test_db_connection(self):
        """Create test database connection."""
        # This would require a test database setup
        # For now, we'll skip these tests unless TEST_DB_URL is set
        test_db_url = os.getenv('TEST_DB_URL')
        if not test_db_url:
            pytest.skip("TEST_DB_URL not set, skipping integration tests")
        
        return test_db_url
    
    def test_full_user_lifecycle(self, test_db_connection):
        """Test complete user lifecycle with real database."""
        db = UserProgressDatabase(test_db_connection)
        
        try:
            # Create user profile
            user_id = f"test-user-{datetime.now().timestamp()}"
            profile = db.create_user_profile(user_id, "testuser", "test@example.com")
            
            assert profile.id == user_id
            
            # Track some problem attempts
            db.track_problem_attempt(user_id, 'algebra', 1, True, 60)
            db.track_problem_attempt(user_id, 'algebra', 1, True, 45)
            db.track_problem_attempt(user_id, 'algebra', 2, False, 120)
            
            # Get updated profile
            updated_profile = db.get_user_profile(user_id)
            assert updated_profile.progress_metrics.total_problems_solved == 3
            assert updated_profile.progress_metrics.current_streak == 0  # Last was incorrect
            
            # Get skill assessment
            assessment = db.get_skill_assessment(user_id, 'algebra')
            assert assessment is not None
            assert assessment.practice_count == 3
            
            # Get analytics
            analytics = db.get_learning_analytics(user_id, 1)
            assert len(analytics['daily_activity']) > 0
            
        finally:
            # Cleanup
            db.close()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])