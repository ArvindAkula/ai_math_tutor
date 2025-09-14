"""
Tests for adaptive learning path algorithms.
Tests performance analysis, topic recommendations, and difficulty adjustment.
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from adaptive_learning import (
    AdaptiveLearningEngine, RecommendationType, TopicDependency, 
    LearningObjective, PerformanceAnalysis, optimize_study_schedule,
    calculate_learning_efficiency
)
from shared.models.core import (
    UserProfile, UserPreferences, ProgressMetrics, LearningPath, 
    Recommendation, MathDomain
)
from user_progress import UserProgressDatabase


class TestAdaptiveLearningEngine:
    """Test suite for AdaptiveLearningEngine class."""
    
    @pytest.fixture
    def mock_progress_db(self):
        """Mock UserProgressDatabase."""
        return Mock(spec=UserProgressDatabase)
    
    @pytest.fixture
    def learning_engine(self, mock_progress_db):
        """Create AdaptiveLearningEngine instance with mocked database."""
        return AdaptiveLearningEngine(mock_progress_db)
    
    @pytest.fixture
    def sample_user_profile(self):
        """Create sample user profile for testing."""
        preferences = UserPreferences(
            preferred_explanation_level="standard",
            visual_learning=True,
            step_by_step_detail="standard",
            notification_settings={}
        )
        
        metrics = ProgressMetrics(
            total_problems_solved=50,
            current_streak=5,
            longest_streak=12,
            average_accuracy=0.75,
            time_spent_learning=300,
            topics_completed=['basic_algebra'],
            skill_levels={
                'basic_algebra': 3,
                'advanced_algebra': 2,
                'calculus_1': 1,
                'linear_algebra': 1
            }
        )
        
        return UserProfile(
            id='test-user-123',
            username='testuser',
            email='test@example.com',
            skill_levels=metrics.skill_levels,
            learning_goals=['calculus_1', 'linear_algebra'],
            preferences=preferences,
            progress_metrics=metrics,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @pytest.fixture
    def sample_analytics(self):
        """Create sample analytics data for testing."""
        return {
            'period_days': 30,
            'daily_activity': [
                {'date': '2024-01-01', 'problems_attempted': 5, 'problems_solved': 4, 'avg_time': 120},
                {'date': '2024-01-02', 'problems_attempted': 3, 'problems_solved': 2, 'avg_time': 90},
                {'date': '2024-01-03', 'problems_attempted': 4, 'problems_solved': 3, 'avg_time': 100},
                {'date': '2024-01-04', 'problems_attempted': 6, 'problems_solved': 5, 'avg_time': 110},
                {'date': '2024-01-05', 'problems_attempted': 2, 'problems_solved': 2, 'avg_time': 80}
            ],
            'domain_performance': [
                {'domain': 'algebra', 'attempts': 12, 'accuracy': 0.8, 'avg_time': 105},
                {'domain': 'calculus', 'attempts': 8, 'accuracy': 0.6, 'avg_time': 150}
            ],
            'difficulty_progression': [
                {'difficulty_level': 1, 'attempts': 8, 'accuracy': 0.9},
                {'difficulty_level': 2, 'attempts': 10, 'accuracy': 0.7},
                {'difficulty_level': 3, 'attempts': 2, 'accuracy': 0.5}
            ]
        }
    
    def test_initialization(self, learning_engine):
        """Test proper initialization of AdaptiveLearningEngine."""
        assert learning_engine.progress_db is not None
        assert len(learning_engine.topic_dependencies) > 0
        assert len(learning_engine.learning_objectives) > 0
        
        # Check that basic topics are present
        assert 'basic_algebra' in learning_engine.topic_dependencies
        assert 'calculus_1' in learning_engine.topic_dependencies
        assert 'linear_algebra' in learning_engine.topic_dependencies
    
    def test_topic_dependencies_structure(self, learning_engine):
        """Test that topic dependencies are properly structured."""
        deps = learning_engine.topic_dependencies
        
        # Basic algebra should have no prerequisites
        assert deps['basic_algebra'].prerequisites == []
        
        # Advanced algebra should require basic algebra
        assert 'basic_algebra' in deps['advanced_algebra'].prerequisites
        
        # Calculus should require precalculus
        assert 'precalculus' in deps['calculus_1'].prerequisites
        
        # AI/ML math should have multiple prerequisites
        ai_ml_prereqs = deps['ai_ml_math'].prerequisites
        assert 'linear_algebra' in ai_ml_prereqs
        assert 'calculus_1' in ai_ml_prereqs
        assert 'statistics' in ai_ml_prereqs
    
    def test_analyze_user_performance_success(self, learning_engine, sample_user_profile, sample_analytics):
        """Test successful user performance analysis."""
        learning_engine.progress_db.get_user_profile.return_value = sample_user_profile
        learning_engine.progress_db.get_learning_analytics.return_value = sample_analytics
        
        analysis = learning_engine.analyze_user_performance('test-user-123')
        
        assert isinstance(analysis, PerformanceAnalysis)
        assert analysis.user_id == 'test-user-123'
        assert 0.0 <= analysis.overall_accuracy <= 1.0
        assert 0.0 <= analysis.consistency_score <= 1.0
        assert analysis.learning_velocity >= 0.0
        assert isinstance(analysis.strengths, list)
        assert isinstance(analysis.weaknesses, list)
        assert 1 <= analysis.recommended_difficulty <= 4
        assert 0.0 <= analysis.confidence_level <= 1.0
    
    def test_analyze_user_performance_no_profile(self, learning_engine):
        """Test performance analysis when user profile doesn't exist."""
        learning_engine.progress_db.get_user_profile.return_value = None
        
        analysis = learning_engine.analyze_user_performance('nonexistent-user')
        
        # Should return default analysis
        assert analysis.user_id == 'nonexistent-user'
        assert analysis.overall_accuracy == 0.0
        assert analysis.learning_velocity == 0.0
        assert 'insufficient_data' in analysis.weaknesses
    
    def test_identify_strengths_weaknesses(self, learning_engine):
        """Test identification of strengths and weaknesses."""
        domain_performance = [
            {'domain': 'algebra', 'accuracy': 0.85, 'attempts': 10},  # Strength
            {'domain': 'calculus', 'accuracy': 0.45, 'attempts': 8},  # Weakness
            {'domain': 'statistics', 'accuracy': 0.65, 'attempts': 2}  # Insufficient data
        ]
        
        strengths, weaknesses = learning_engine._identify_strengths_weaknesses(domain_performance)
        
        assert 'algebra' in strengths
        assert 'calculus' in weaknesses
        assert 'statistics' not in strengths and 'statistics' not in weaknesses
    
    def test_calculate_recommended_difficulty(self, learning_engine, sample_user_profile):
        """Test difficulty recommendation calculation."""
        analytics = {
            'difficulty_progression': [
                {'difficulty_level': 1, 'accuracy': 0.9, 'attempts': 5},
                {'difficulty_level': 2, 'accuracy': 0.8, 'attempts': 8},
                {'difficulty_level': 3, 'accuracy': 0.6, 'attempts': 3}
            ]
        }
        
        recommended = learning_engine._calculate_recommended_difficulty(sample_user_profile, analytics)
        
        assert 1 <= recommended <= 4
        # Should recommend level 3 since user performs well at level 2
        assert recommended >= 2
    
    def test_calculate_confidence_level(self, learning_engine):
        """Test confidence level calculation."""
        # Test with improving trend
        analytics_improving = {
            'daily_activity': [
                {'problems_attempted': 5, 'problems_solved': 2},  # 0.4 accuracy
                {'problems_attempted': 4, 'problems_solved': 2},  # 0.5 accuracy
                {'problems_attempted': 3, 'problems_solved': 2},  # 0.67 accuracy
                {'problems_attempted': 4, 'problems_solved': 3}   # 0.75 accuracy
            ]
        }
        
        confidence = learning_engine._calculate_confidence_level(analytics_improving)
        assert 0.0 <= confidence <= 1.0
        # Should be relatively high due to improving trend
        assert confidence > 0.5
    
    def test_generate_topic_recommendations(self, learning_engine, sample_user_profile, sample_analytics):
        """Test topic recommendation generation."""
        learning_engine.progress_db.get_user_profile.return_value = sample_user_profile
        learning_engine.progress_db.get_learning_analytics.return_value = sample_analytics
        
        recommendations = learning_engine.generate_topic_recommendations('test-user-123', max_recommendations=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.type in [rt.value for rt in RecommendationType]
            assert rec.priority >= 1
            assert rec.estimated_time > 0
    
    def test_find_available_topics(self, learning_engine):
        """Test finding available topics based on prerequisites."""
        skill_levels = {
            'basic_algebra': 3,
            'advanced_algebra': 2,
            'precalculus': 1
        }
        completed_topics = ['basic_algebra']
        
        available = learning_engine._find_available_topics(skill_levels, completed_topics)
        
        # Should include advanced_algebra (prerequisite met)
        # Should not include calculus_1 (precalculus not completed)
        assert 'advanced_algebra' in available
        assert 'calculus_1' not in available
    
    def test_adjust_difficulty_dynamically_increase(self, learning_engine):
        """Test dynamic difficulty adjustment - increase scenario."""
        recent_performance = [True, True, True, True, True]  # 100% accuracy
        current_difficulty = 2
        
        new_difficulty = learning_engine.adjust_difficulty_dynamically(
            'test-user', recent_performance, current_difficulty
        )
        
        assert new_difficulty > current_difficulty
        assert new_difficulty <= 4  # Max difficulty
    
    def test_adjust_difficulty_dynamically_decrease(self, learning_engine):
        """Test dynamic difficulty adjustment - decrease scenario."""
        recent_performance = [False, False, True, False, False]  # 20% accuracy
        current_difficulty = 3
        
        new_difficulty = learning_engine.adjust_difficulty_dynamically(
            'test-user', recent_performance, current_difficulty
        )
        
        assert new_difficulty < current_difficulty
        assert new_difficulty >= 1  # Min difficulty
    
    def test_adjust_difficulty_dynamically_maintain(self, learning_engine):
        """Test dynamic difficulty adjustment - maintain scenario."""
        recent_performance = [True, False, True, True, False]  # 60% accuracy
        current_difficulty = 2
        
        new_difficulty = learning_engine.adjust_difficulty_dynamically(
            'test-user', recent_performance, current_difficulty
        )
        
        assert new_difficulty == current_difficulty
    
    def test_calculate_performance_trend(self, learning_engine):
        """Test performance trend calculation."""
        # Improving trend
        improving_performance = [False, False, True, True, True]
        trend = learning_engine._calculate_performance_trend(improving_performance)
        assert trend > 0
        
        # Declining trend
        declining_performance = [True, True, True, False, False]
        trend = learning_engine._calculate_performance_trend(declining_performance)
        assert trend < 0
        
        # Stable trend
        stable_performance = [True, False, True, False, True]
        trend = learning_engine._calculate_performance_trend(stable_performance)
        assert abs(trend) < 0.5  # Should be close to 0
    
    def test_generate_learning_path(self, learning_engine, sample_user_profile, sample_analytics):
        """Test learning path generation."""
        learning_engine.progress_db.get_user_profile.return_value = sample_user_profile
        learning_engine.progress_db.get_learning_analytics.return_value = sample_analytics
        
        learning_path = learning_engine.generate_learning_path('test-user-123')
        
        assert isinstance(learning_path, LearningPath)
        assert learning_path.user_id == 'test-user-123'
        assert learning_path.current_topic is not None
        assert isinstance(learning_path.next_topics, list)
        assert isinstance(learning_path.recommendations, list)
        assert learning_path.estimated_completion_time > 0
        assert 0.0 <= learning_path.progress_percentage <= 100.0
    
    def test_determine_current_topic(self, learning_engine, sample_user_profile):
        """Test current topic determination."""
        analysis = PerformanceAnalysis(
            user_id='test-user',
            overall_accuracy=0.75,
            consistency_score=0.8,
            learning_velocity=2.0,
            strengths=['algebra'],
            weaknesses=[],
            recommended_difficulty=2,
            confidence_level=0.7,
            analysis_timestamp=datetime.now()
        )
        
        current_topic = learning_engine._determine_current_topic(sample_user_profile, analysis)
        
        # Should return a valid topic
        assert current_topic in learning_engine.topic_dependencies
        # Should be based on skill levels (advanced_algebra has level 2)
        assert current_topic == 'advanced_algebra'
    
    def test_plan_topic_sequence(self, learning_engine, sample_user_profile):
        """Test topic sequence planning."""
        target_topics = ['calculus_1', 'linear_algebra']
        
        sequence = learning_engine._plan_topic_sequence(sample_user_profile, target_topics)
        
        assert isinstance(sequence, list)
        assert len(sequence) <= 10  # Should be limited
        
        # Should include prerequisites
        if 'calculus_1' in sequence:
            # Precalculus should come before calculus_1
            if 'precalculus' in sequence:
                assert sequence.index('precalculus') < sequence.index('calculus_1')
    
    def test_get_all_prerequisites(self, learning_engine):
        """Test recursive prerequisite finding."""
        prereqs = learning_engine._get_all_prerequisites('ai_ml_math')
        
        # Should include direct and indirect prerequisites
        assert 'linear_algebra' in prereqs
        assert 'calculus_1' in prereqs
        assert 'statistics' in prereqs
        assert 'advanced_algebra' in prereqs  # Indirect prerequisite
        assert 'basic_algebra' in prereqs     # Indirect prerequisite
    
    def test_topological_sort(self, learning_engine):
        """Test topological sorting of topics."""
        topics = {'basic_algebra', 'advanced_algebra', 'precalculus', 'calculus_1'}
        
        sorted_topics = learning_engine._topological_sort(topics)
        
        assert len(sorted_topics) == len(topics)
        # Basic algebra should come before advanced algebra
        assert sorted_topics.index('basic_algebra') < sorted_topics.index('advanced_algebra')
        # Precalculus should come before calculus
        assert sorted_topics.index('precalculus') < sorted_topics.index('calculus_1')
    
    def test_estimate_completion_time(self, learning_engine):
        """Test completion time estimation."""
        topics = ['advanced_algebra', 'precalculus', 'calculus_1']
        analysis = PerformanceAnalysis(
            user_id='test-user',
            overall_accuracy=0.75,
            consistency_score=0.8,
            learning_velocity=2.0,  # Good velocity
            strengths=[],
            weaknesses=[],
            recommended_difficulty=2,
            confidence_level=0.8,   # High confidence
            analysis_timestamp=datetime.now()
        )
        
        completion_time = learning_engine._estimate_completion_time(topics, analysis)
        
        assert completion_time >= 1
        assert isinstance(completion_time, int)
        # Should be reasonable (not too high due to good performance)
        assert completion_time < 365  # Less than a year
    
    def test_calculate_progress_percentage(self, learning_engine, sample_user_profile):
        """Test progress percentage calculation."""
        target_topics = ['calculus_1']
        
        progress = learning_engine._calculate_progress_percentage(sample_user_profile, target_topics)
        
        assert 0.0 <= progress <= 100.0
        # Should be > 0 since user has some skill in prerequisites
        assert progress > 0.0


class TestTopicDependency:
    """Test suite for TopicDependency data class."""
    
    def test_topic_dependency_creation(self):
        """Test creating a TopicDependency instance."""
        dependency = TopicDependency(
            topic='calculus_1',
            prerequisites=['precalculus'],
            difficulty_level=3,
            estimated_hours=20.0,
            importance_weight=0.9
        )
        
        assert dependency.topic == 'calculus_1'
        assert dependency.prerequisites == ['precalculus']
        assert dependency.difficulty_level == 3
        assert dependency.estimated_hours == 20.0
        assert dependency.importance_weight == 0.9


class TestLearningObjective:
    """Test suite for LearningObjective data class."""
    
    def test_learning_objective_creation(self):
        """Test creating a LearningObjective instance."""
        objective = LearningObjective(
            id='c1_001',
            topic='calculus_1',
            title='Limits',
            description='Understand and compute limits',
            difficulty=2,
            prerequisites=[],
            skills_developed=['limit_computation'],
            estimated_time=120
        )
        
        assert objective.id == 'c1_001'
        assert objective.topic == 'calculus_1'
        assert objective.title == 'Limits'
        assert objective.difficulty == 2
        assert 'limit_computation' in objective.skills_developed


class TestPerformanceAnalysis:
    """Test suite for PerformanceAnalysis data class."""
    
    def test_performance_analysis_creation(self):
        """Test creating a PerformanceAnalysis instance."""
        analysis = PerformanceAnalysis(
            user_id='test-user',
            overall_accuracy=0.75,
            consistency_score=0.8,
            learning_velocity=2.0,
            strengths=['algebra'],
            weaknesses=['calculus'],
            recommended_difficulty=2,
            confidence_level=0.7,
            analysis_timestamp=datetime.now()
        )
        
        assert analysis.user_id == 'test-user'
        assert analysis.overall_accuracy == 0.75
        assert analysis.consistency_score == 0.8
        assert analysis.learning_velocity == 2.0
        assert 'algebra' in analysis.strengths
        assert 'calculus' in analysis.weaknesses
        assert analysis.recommended_difficulty == 2
        assert analysis.confidence_level == 0.7


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_optimize_study_schedule(self):
        """Test study schedule optimization."""
        learning_path = LearningPath(
            user_id='test-user',
            current_topic='algebra',
            next_topics=['advanced_algebra', 'precalculus', 'calculus_1'],
            recommendations=[
                Recommendation(
                    type='topic_introduction',
                    content='Start learning Advanced Algebra',
                    reason='Prerequisites met',
                    priority=2,
                    estimated_time=60
                )
            ],
            estimated_completion_time=30,
            progress_percentage=25.0
        )
        
        schedule = optimize_study_schedule(learning_path, available_hours_per_day=2.0)
        
        assert 'total_days' in schedule
        assert 'daily_schedule' in schedule
        assert 'study_hours_per_day' in schedule
        assert 'completion_date' in schedule
        
        assert schedule['study_hours_per_day'] == 2.0
        assert schedule['total_days'] > 0
        assert len(schedule['daily_schedule']) > 0
        
        # Check daily schedule structure
        for day_plan in schedule['daily_schedule']:
            assert 'day' in day_plan
            assert 'topic' in day_plan
            assert 'duration_days' in day_plan
            assert 'focus_areas' in day_plan
    
    def test_calculate_learning_efficiency(self):
        """Test learning efficiency calculation."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': [
                {'problems_attempted': 5, 'problems_solved': 4, 'avg_time': 120},
                {'problems_attempted': 3, 'problems_solved': 2, 'avg_time': 90},
                {'problems_attempted': 4, 'problems_solved': 3, 'avg_time': 100}
            ]
        }
        
        efficiency = calculate_learning_efficiency('test-user', mock_db, 7)
        
        assert 0.0 <= efficiency <= 1.0
        assert isinstance(efficiency, float)
    
    def test_calculate_learning_efficiency_no_data(self):
        """Test learning efficiency calculation with no data."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': []
        }
        
        efficiency = calculate_learning_efficiency('test-user', mock_db, 7)
        
        assert efficiency == 0.0
    
    def test_calculate_learning_efficiency_error(self):
        """Test learning efficiency calculation with database error."""
        mock_db = Mock()
        mock_db.get_learning_analytics.side_effect = Exception("Database error")
        
        efficiency = calculate_learning_efficiency('test-user', mock_db, 7)
        
        assert efficiency == 0.0


class TestIntegration:
    """Integration tests for adaptive learning components."""
    
    def test_full_adaptive_learning_workflow(self):
        """Test complete adaptive learning workflow."""
        # Mock database
        mock_db = Mock()
        
        # Sample user profile
        profile = UserProfile(
            id='test-user',
            username='testuser',
            email='test@example.com',
            skill_levels={'basic_algebra': 3, 'advanced_algebra': 2},
            learning_goals=['calculus_1'],
            preferences=UserPreferences(
                preferred_explanation_level="standard",
                visual_learning=True,
                step_by_step_detail="standard",
                notification_settings={}
            ),
            progress_metrics=ProgressMetrics(
                total_problems_solved=30,
                current_streak=3,
                longest_streak=8,
                average_accuracy=0.75,
                time_spent_learning=180,
                topics_completed=['basic_algebra'],
                skill_levels={'basic_algebra': 3, 'advanced_algebra': 2}
            ),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Sample analytics
        analytics = {
            'daily_activity': [
                {'problems_attempted': 5, 'problems_solved': 4, 'avg_time': 120},
                {'problems_attempted': 3, 'problems_solved': 2, 'avg_time': 90}
            ],
            'domain_performance': [
                {'domain': 'algebra', 'accuracy': 0.8, 'attempts': 8}
            ],
            'difficulty_progression': [
                {'difficulty_level': 2, 'accuracy': 0.75, 'attempts': 8}
            ]
        }
        
        mock_db.get_user_profile.return_value = profile
        mock_db.get_learning_analytics.return_value = analytics
        
        # Create engine and test workflow
        engine = AdaptiveLearningEngine(mock_db)
        
        # 1. Analyze performance
        analysis = engine.analyze_user_performance('test-user')
        assert isinstance(analysis, PerformanceAnalysis)
        
        # 2. Generate recommendations
        recommendations = engine.generate_topic_recommendations('test-user')
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # 3. Generate learning path
        learning_path = engine.generate_learning_path('test-user')
        assert isinstance(learning_path, LearningPath)
        
        # 4. Optimize schedule
        schedule = optimize_study_schedule(learning_path)
        assert 'total_days' in schedule
        
        # 5. Calculate efficiency
        efficiency = calculate_learning_efficiency('test-user', mock_db)
        assert 0.0 <= efficiency <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])