"""
Tests for progress analytics and reporting system.
Tests visualization, streak tracking, achievement system, and learning analytics.
"""

import pytest
import base64
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from progress_analytics import (
    ProgressAnalytics, AchievementType, ReportType, Achievement, 
    StreakInfo, ProgressVisualization, LearningInsight, ProgressReport,
    calculate_learning_momentum, generate_study_recommendations
)
from shared.models.core import (
    UserProfile, UserPreferences, ProgressMetrics
)
from user_progress import UserProgressDatabase
from adaptive_learning import AdaptiveLearningEngine, PerformanceAnalysis


class TestProgressAnalytics:
    """Test suite for ProgressAnalytics class."""
    
    @pytest.fixture
    def mock_progress_db(self):
        """Mock UserProgressDatabase."""
        return Mock(spec=UserProgressDatabase)
    
    @pytest.fixture
    def mock_learning_engine(self):
        """Mock AdaptiveLearningEngine."""
        return Mock(spec=AdaptiveLearningEngine)
    
    @pytest.fixture
    def progress_analytics(self, mock_progress_db, mock_learning_engine):
        """Create ProgressAnalytics instance with mocked dependencies."""
        return ProgressAnalytics(mock_progress_db, mock_learning_engine)
    
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
                'calculus': 1
            }
        )
        
        return UserProfile(
            id='test-user-123',
            username='testuser',
            email='test@example.com',
            skill_levels=metrics.skill_levels,
            learning_goals=['calculus'],
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
    
    def test_initialization(self, progress_analytics):
        """Test proper initialization of ProgressAnalytics."""
        assert progress_analytics.progress_db is not None
        assert progress_analytics.learning_engine is not None
        assert len(progress_analytics.achievement_definitions) > 0
        
        # Check that basic achievements are defined
        assert 'first_problem' in progress_analytics.achievement_definitions
        assert 'streak_7' in progress_analytics.achievement_definitions
        assert 'accuracy_master' in progress_analytics.achievement_definitions
    
    def test_achievement_definitions_structure(self, progress_analytics):
        """Test that achievement definitions are properly structured."""
        definitions = progress_analytics.achievement_definitions
        
        for achievement_id, definition in definitions.items():
            assert 'type' in definition
            assert 'title' in definition
            assert 'description' in definition
            assert 'icon' in definition
            assert 'points' in definition
            assert 'requirements' in definition
            
            # Check that type is valid
            assert definition['type'] in AchievementType
            
            # Check that points are positive
            assert definition['points'] > 0
    
    def test_check_achievement_requirements_problems_solved(self, progress_analytics, sample_user_profile):
        """Test achievement requirement checking for problems solved."""
        analytics = {}
        
        # Test first problem achievement
        first_problem_def = progress_analytics.achievement_definitions['first_problem']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, first_problem_def
        )
        assert result is True  # User has solved 50 problems
        
        # Test century club achievement
        century_def = progress_analytics.achievement_definitions['problem_solver_100']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, century_def
        )
        assert result is False  # User has only solved 50 problems
    
    def test_check_achievement_requirements_streak(self, progress_analytics, sample_user_profile):
        """Test achievement requirement checking for streaks."""
        analytics = {}
        
        # Test 3-day streak achievement
        streak_3_def = progress_analytics.achievement_definitions['streak_3']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, streak_3_def
        )
        assert result is True  # User's longest streak is 12
        
        # Test 30-day streak achievement
        streak_30_def = progress_analytics.achievement_definitions['streak_30']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, streak_30_def
        )
        assert result is False  # User's longest streak is only 12
    
    def test_check_achievement_requirements_accuracy(self, progress_analytics, sample_user_profile):
        """Test achievement requirement checking for accuracy."""
        analytics = {}
        
        accuracy_def = progress_analytics.achievement_definitions['accuracy_master']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, accuracy_def
        )
        assert result is False  # User's accuracy is 0.75, requirement is 0.9
    
    def test_check_achievement_requirements_topic_mastery(self, progress_analytics, sample_user_profile):
        """Test achievement requirement checking for topic mastery."""
        analytics = {
            'domain_performance': [
                {'domain': 'algebra', 'accuracy': 0.85},
                {'domain': 'calculus', 'accuracy': 0.6}
            ]
        }
        
        algebra_def = progress_analytics.achievement_definitions['algebra_master']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, algebra_def
        )
        assert result is True  # Algebra accuracy is 0.85, requirement is 0.8
        
        calculus_def = progress_analytics.achievement_definitions['calculus_master']
        result = progress_analytics._check_achievement_requirements(
            sample_user_profile, analytics, calculus_def
        )
        assert result is False  # Calculus accuracy is 0.6, requirement is 0.8
    
    def test_check_and_award_achievements(self, progress_analytics, sample_user_profile, sample_analytics):
        """Test checking and awarding achievements."""
        progress_analytics.progress_db.get_user_profile.return_value = sample_user_profile
        progress_analytics.progress_db.get_learning_analytics.return_value = sample_analytics
        
        with patch.object(progress_analytics, '_get_user_achievements', return_value=[]), \
             patch.object(progress_analytics, '_save_achievement') as mock_save:
            
            achievements = progress_analytics.check_and_award_achievements('test-user-123')
            
            assert isinstance(achievements, list)
            # Should have earned some achievements
            assert len(achievements) > 0
            
            # Check that achievements are properly structured
            for achievement in achievements:
                assert isinstance(achievement, Achievement)
                assert achievement.user_id == 'test-user-123'
                assert achievement.points > 0
                assert achievement.earned_at is not None
    
    def test_get_streak_info(self, progress_analytics, sample_user_profile, sample_analytics):
        """Test getting streak information."""
        progress_analytics.progress_db.get_user_profile.return_value = sample_user_profile
        progress_analytics.progress_db.get_learning_analytics.return_value = sample_analytics
        
        streak_info = progress_analytics.get_streak_info('test-user-123')
        
        assert isinstance(streak_info, StreakInfo)
        assert streak_info.user_id == 'test-user-123'
        assert streak_info.current_streak == 5
        assert streak_info.longest_streak == 12
        assert streak_info.streak_type == 'daily'
        assert isinstance(streak_info.is_active, bool)
    
    def test_get_streak_info_no_profile(self, progress_analytics):
        """Test getting streak info when user profile doesn't exist."""
        progress_analytics.progress_db.get_user_profile.return_value = None
        
        streak_info = progress_analytics.get_streak_info('nonexistent-user')
        
        assert streak_info.user_id == 'nonexistent-user'
        assert streak_info.current_streak == 0
        assert streak_info.longest_streak == 0
        assert streak_info.is_active is False
    
    def test_generate_progress_visualizations(self, progress_analytics, sample_analytics):
        """Test generating progress visualizations."""
        progress_analytics.progress_db.get_learning_analytics.return_value = sample_analytics
        
        visualizations = progress_analytics.generate_progress_visualizations('test-user-123')
        
        assert isinstance(visualizations, list)
        assert len(visualizations) > 0
        
        # Check that all expected visualization types are present
        chart_types = {viz.chart_type for viz in visualizations}
        expected_types = {'daily_activity', 'domain_performance', 'difficulty_progress', 'accuracy_trend'}
        assert expected_types.issubset(chart_types)
        
        # Check visualization structure
        for viz in visualizations:
            assert isinstance(viz, ProgressVisualization)
            assert viz.title is not None
            assert viz.data is not None
            # Chart image should be base64 encoded string or None
            if viz.chart_image:
                assert isinstance(viz.chart_image, str)
    
    def test_create_daily_activity_chart(self, progress_analytics):
        """Test creating daily activity chart."""
        daily_activity = [
            {'date': '2024-01-01', 'problems_attempted': 5, 'problems_solved': 4},
            {'date': '2024-01-02', 'problems_attempted': 3, 'problems_solved': 2}
        ]
        
        viz = progress_analytics._create_daily_activity_chart(daily_activity)
        
        assert isinstance(viz, ProgressVisualization)
        assert viz.chart_type == 'daily_activity'
        assert viz.title is not None
        assert 'dates' in viz.data
        assert 'attempted' in viz.data
        assert 'solved' in viz.data
        assert len(viz.data['dates']) == 2
    
    def test_create_daily_activity_chart_no_data(self, progress_analytics):
        """Test creating daily activity chart with no data."""
        viz = progress_analytics._create_daily_activity_chart([])
        
        assert viz.chart_type == 'daily_activity'
        assert 'message' in viz.data
        assert viz.chart_image is None
    
    def test_create_domain_performance_chart(self, progress_analytics):
        """Test creating domain performance chart."""
        domain_performance = [
            {'domain': 'algebra', 'accuracy': 0.8, 'attempts': 10},
            {'domain': 'calculus', 'accuracy': 0.6, 'attempts': 5}
        ]
        
        viz = progress_analytics._create_domain_performance_chart(domain_performance)
        
        assert isinstance(viz, ProgressVisualization)
        assert viz.chart_type == 'domain_performance'
        assert 'domains' in viz.data
        assert 'accuracies' in viz.data
        assert 'attempts' in viz.data
        assert len(viz.data['domains']) == 2
    
    def test_create_difficulty_progress_chart(self, progress_analytics):
        """Test creating difficulty progress chart."""
        difficulty_progression = [
            {'difficulty_level': 1, 'accuracy': 0.9, 'attempts': 8},
            {'difficulty_level': 2, 'accuracy': 0.7, 'attempts': 10}
        ]
        
        viz = progress_analytics._create_difficulty_progress_chart(difficulty_progression)
        
        assert isinstance(viz, ProgressVisualization)
        assert viz.chart_type == 'difficulty_progress'
        assert 'levels' in viz.data
        assert 'accuracies' in viz.data
        assert 'attempts' in viz.data
    
    def test_create_accuracy_trend_chart(self, progress_analytics):
        """Test creating accuracy trend chart."""
        daily_activity = [
            {'date': '2024-01-01', 'problems_attempted': 5, 'problems_solved': 4},
            {'date': '2024-01-02', 'problems_attempted': 4, 'problems_solved': 3},
            {'date': '2024-01-03', 'problems_attempted': 3, 'problems_solved': 3}
        ]
        
        viz = progress_analytics._create_accuracy_trend_chart(daily_activity)
        
        assert isinstance(viz, ProgressVisualization)
        assert viz.chart_type == 'accuracy_trend'
        assert 'dates' in viz.data
        assert 'accuracies' in viz.data
        assert len(viz.data['dates']) == 3
    
    def test_generate_learning_insights(self, progress_analytics, sample_user_profile, sample_analytics):
        """Test generating learning insights."""
        progress_analytics.progress_db.get_user_profile.return_value = sample_user_profile
        progress_analytics.progress_db.get_learning_analytics.return_value = sample_analytics
        
        # Mock performance analysis
        mock_analysis = PerformanceAnalysis(
            user_id='test-user-123',
            overall_accuracy=0.75,
            consistency_score=0.8,
            learning_velocity=2.0,
            strengths=['algebra'],
            weaknesses=['calculus'],
            recommended_difficulty=2,
            confidence_level=0.7,
            analysis_timestamp=datetime.now()
        )
        progress_analytics.learning_engine.analyze_user_performance.return_value = mock_analysis
        
        insights = progress_analytics.generate_learning_insights('test-user-123')
        
        assert isinstance(insights, list)
        
        # Check insight structure
        for insight in insights:
            assert isinstance(insight, LearningInsight)
            assert insight.insight_type is not None
            assert insight.title is not None
            assert insight.description is not None
            assert insight.recommendation is not None
            assert 0.0 <= insight.confidence <= 1.0
            assert isinstance(insight.supporting_data, dict)
    
    def test_generate_progress_report(self, progress_analytics, sample_user_profile, sample_analytics):
        """Test generating comprehensive progress report."""
        progress_analytics.progress_db.get_user_profile.return_value = sample_user_profile
        progress_analytics.progress_db.get_learning_analytics.return_value = sample_analytics
        
        # Mock other methods
        with patch.object(progress_analytics, 'generate_progress_visualizations', return_value=[]), \
             patch.object(progress_analytics, 'check_and_award_achievements', return_value=[]), \
             patch.object(progress_analytics, 'generate_learning_insights', return_value=[]):
            
            report = progress_analytics.generate_progress_report(
                'test-user-123', ReportType.WEEKLY_REPORT
            )
            
            assert isinstance(report, ProgressReport)
            assert report.user_id == 'test-user-123'
            assert report.report_type == ReportType.WEEKLY_REPORT
            assert report.period_start < report.period_end
            assert isinstance(report.summary_stats, dict)
            assert isinstance(report.visualizations, list)
            assert isinstance(report.achievements, list)
            assert isinstance(report.insights, list)
            assert report.generated_at is not None
    
    def test_calculate_summary_stats(self, progress_analytics, sample_user_profile, sample_analytics):
        """Test calculating summary statistics."""
        stats = progress_analytics._calculate_summary_stats(sample_user_profile, sample_analytics, 30)
        
        assert isinstance(stats, dict)
        
        # Check required fields
        required_fields = [
            'period_days', 'problems_attempted', 'problems_solved', 'accuracy',
            'active_days', 'time_spent_minutes', 'current_streak', 'longest_streak',
            'total_problems_solved', 'overall_accuracy', 'domains_practiced'
        ]
        
        for field in required_fields:
            assert field in stats
        
        # Check value ranges
        assert stats['period_days'] == 30
        assert stats['problems_attempted'] >= stats['problems_solved']
        assert 0.0 <= stats['accuracy'] <= 1.0
        assert stats['active_days'] >= 0
        assert stats['current_streak'] >= 0
        assert stats['longest_streak'] >= stats['current_streak']
    
    def test_calculate_average_problem_time(self, progress_analytics, sample_analytics):
        """Test calculating average problem time."""
        avg_time = progress_analytics._calculate_average_problem_time(sample_analytics)
        
        assert avg_time > 0
        assert isinstance(avg_time, float)
        # Should be reasonable (between 1 second and 1 hour)
        assert 1 <= avg_time <= 3600


class TestAchievement:
    """Test suite for Achievement data class."""
    
    def test_achievement_creation(self):
        """Test creating an Achievement instance."""
        achievement = Achievement(
            id='test_achievement',
            user_id='test-user',
            achievement_type=AchievementType.PROBLEMS_SOLVED,
            title='Test Achievement',
            description='A test achievement',
            icon='🏆',
            points=100,
            earned_at=datetime.now(),
            requirements_met={'problems_solved': 10},
            is_milestone=True
        )
        
        assert achievement.id == 'test_achievement'
        assert achievement.user_id == 'test-user'
        assert achievement.achievement_type == AchievementType.PROBLEMS_SOLVED
        assert achievement.title == 'Test Achievement'
        assert achievement.points == 100
        assert achievement.is_milestone is True


class TestStreakInfo:
    """Test suite for StreakInfo data class."""
    
    def test_streak_info_creation(self):
        """Test creating a StreakInfo instance."""
        now = datetime.now()
        streak_info = StreakInfo(
            user_id='test-user',
            current_streak=5,
            longest_streak=10,
            last_activity_date=now,
            streak_start_date=now - timedelta(days=5),
            is_active=True,
            streak_type='daily'
        )
        
        assert streak_info.user_id == 'test-user'
        assert streak_info.current_streak == 5
        assert streak_info.longest_streak == 10
        assert streak_info.is_active is True
        assert streak_info.streak_type == 'daily'


class TestProgressVisualization:
    """Test suite for ProgressVisualization data class."""
    
    def test_progress_visualization_creation(self):
        """Test creating a ProgressVisualization instance."""
        viz = ProgressVisualization(
            chart_type='daily_activity',
            title='Daily Activity Chart',
            data={'dates': ['2024-01-01'], 'values': [5]},
            chart_image='base64encodedimage',
            interactive_data={'config': 'test'}
        )
        
        assert viz.chart_type == 'daily_activity'
        assert viz.title == 'Daily Activity Chart'
        assert 'dates' in viz.data
        assert viz.chart_image == 'base64encodedimage'
        assert viz.interactive_data is not None


class TestLearningInsight:
    """Test suite for LearningInsight data class."""
    
    def test_learning_insight_creation(self):
        """Test creating a LearningInsight instance."""
        insight = LearningInsight(
            insight_type='consistency',
            title='Improve Consistency',
            description='You need to practice more regularly',
            recommendation='Try to practice daily',
            confidence=0.8,
            supporting_data={'active_days': 3, 'total_days': 7}
        )
        
        assert insight.insight_type == 'consistency'
        assert insight.title == 'Improve Consistency'
        assert insight.confidence == 0.8
        assert 'active_days' in insight.supporting_data


class TestProgressReport:
    """Test suite for ProgressReport data class."""
    
    def test_progress_report_creation(self):
        """Test creating a ProgressReport instance."""
        now = datetime.now()
        report = ProgressReport(
            user_id='test-user',
            report_type=ReportType.WEEKLY_REPORT,
            period_start=now - timedelta(days=7),
            period_end=now,
            summary_stats={'problems_solved': 10},
            visualizations=[],
            achievements=[],
            insights=[],
            generated_at=now
        )
        
        assert report.user_id == 'test-user'
        assert report.report_type == ReportType.WEEKLY_REPORT
        assert report.period_start < report.period_end
        assert isinstance(report.summary_stats, dict)
        assert isinstance(report.visualizations, list)


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_calculate_learning_momentum(self):
        """Test learning momentum calculation."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': [
                {'problems_solved': 2},
                {'problems_solved': 3},
                {'problems_solved': 4},
                {'problems_solved': 5}
            ]
        }
        
        momentum = calculate_learning_momentum('test-user', mock_db, 14)
        
        assert -1.0 <= momentum <= 1.0
        assert momentum > 0  # Should be positive due to increasing trend
    
    def test_calculate_learning_momentum_no_data(self):
        """Test learning momentum calculation with no data."""
        mock_db = Mock()
        mock_db.get_learning_analytics.return_value = {
            'daily_activity': []
        }
        
        momentum = calculate_learning_momentum('test-user', mock_db, 14)
        
        assert momentum == 0.0
    
    def test_calculate_learning_momentum_error(self):
        """Test learning momentum calculation with database error."""
        mock_db = Mock()
        mock_db.get_learning_analytics.side_effect = Exception("Database error")
        
        momentum = calculate_learning_momentum('test-user', mock_db, 14)
        
        assert momentum == 0.0
    
    def test_generate_study_recommendations(self):
        """Test generating study recommendations from progress report."""
        # Create mock progress report
        report = ProgressReport(
            user_id='test-user',
            report_type=ReportType.WEEKLY_REPORT,
            period_start=datetime.now() - timedelta(days=7),
            period_end=datetime.now(),
            summary_stats={
                'accuracy': 0.5,  # Low accuracy
                'active_days': 2,
                'period_days': 7,
                'current_streak': 0
            },
            visualizations=[],
            achievements=[],
            insights=[
                LearningInsight(
                    insight_type='focus_area',
                    title='Focus on Algebra',
                    description='Need improvement in algebra',
                    recommendation='Practice more algebra problems',
                    confidence=0.8,
                    supporting_data={}
                )
            ],
            generated_at=datetime.now()
        )
        
        recommendations = generate_study_recommendations(report)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5  # Should be limited to 5
        
        # Should include recommendations based on low accuracy and consistency
        rec_text = ' '.join(recommendations)
        assert any(keyword in rec_text.lower() for keyword in ['concept', 'understanding', 'practice'])


class TestIntegration:
    """Integration tests for progress analytics components."""
    
    def test_full_progress_analytics_workflow(self):
        """Test complete progress analytics workflow."""
        # Mock dependencies
        mock_db = Mock()
        mock_engine = Mock()
        
        # Sample data
        profile = UserProfile(
            id='test-user',
            username='testuser',
            email='test@example.com',
            skill_levels={'algebra': 2, 'calculus': 1},
            learning_goals=['calculus'],
            preferences=UserPreferences(
                preferred_explanation_level="standard",
                visual_learning=True,
                step_by_step_detail="standard",
                notification_settings={}
            ),
            progress_metrics=ProgressMetrics(
                total_problems_solved=25,
                current_streak=3,
                longest_streak=8,
                average_accuracy=0.75,
                time_spent_learning=180,
                topics_completed=['basic_algebra'],
                skill_levels={'algebra': 2, 'calculus': 1}
            ),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        analytics = {
            'daily_activity': [
                {'date': '2024-01-01', 'problems_attempted': 5, 'problems_solved': 4, 'avg_time': 120}
            ],
            'domain_performance': [
                {'domain': 'algebra', 'accuracy': 0.8, 'attempts': 10}
            ],
            'difficulty_progression': [
                {'difficulty_level': 2, 'accuracy': 0.75, 'attempts': 10}
            ]
        }
        
        mock_analysis = PerformanceAnalysis(
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
        
        mock_db.get_user_profile.return_value = profile
        mock_db.get_learning_analytics.return_value = analytics
        mock_engine.analyze_user_performance.return_value = mock_analysis
        
        # Create analytics system and test workflow
        analytics_system = ProgressAnalytics(mock_db, mock_engine)
        
        # 1. Check achievements
        achievements = analytics_system.check_and_award_achievements('test-user')
        assert isinstance(achievements, list)
        
        # 2. Get streak info
        streak_info = analytics_system.get_streak_info('test-user')
        assert isinstance(streak_info, StreakInfo)
        
        # 3. Generate visualizations
        visualizations = analytics_system.generate_progress_visualizations('test-user')
        assert isinstance(visualizations, list)
        
        # 4. Generate insights
        insights = analytics_system.generate_learning_insights('test-user')
        assert isinstance(insights, list)
        
        # 5. Generate full report
        report = analytics_system.generate_progress_report('test-user', ReportType.WEEKLY_REPORT)
        assert isinstance(report, ProgressReport)
        
        # 6. Generate recommendations
        recommendations = generate_study_recommendations(report)
        assert isinstance(recommendations, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])