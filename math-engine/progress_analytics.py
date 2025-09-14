"""
Progress analytics and reporting system for AI Math Tutor.
Implements progress visualization, streak tracking, achievement system, and learning analytics.
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from io import BytesIO
import base64

from shared.models.core import UserProfile, ProgressMetrics
from user_progress import UserProgressDatabase
from adaptive_learning import AdaptiveLearningEngine, PerformanceAnalysis

logger = logging.getLogger(__name__)


class AchievementType(Enum):
    """Types of achievements users can earn."""
    STREAK_MILESTONE = "streak_milestone"
    PROBLEMS_SOLVED = "problems_solved"
    TOPIC_MASTERY = "topic_mastery"
    ACCURACY_MILESTONE = "accuracy_milestone"
    TIME_DEDICATION = "time_dedication"
    CONSISTENCY = "consistency"
    IMPROVEMENT = "improvement"
    SPEED_DEMON = "speed_demon"


class ReportType(Enum):
    """Types of progress reports."""
    DAILY_SUMMARY = "daily_summary"
    WEEKLY_REPORT = "weekly_report"
    MONTHLY_REPORT = "monthly_report"
    TOPIC_PROGRESS = "topic_progress"
    PERFORMANCE_ANALYSIS = "performance_analysis"


@dataclass
class Achievement:
    """Represents a user achievement."""
    id: str
    user_id: str
    achievement_type: AchievementType
    title: str
    description: str
    icon: str
    points: int
    earned_at: datetime
    requirements_met: Dict[str, Any]
    is_milestone: bool = False


@dataclass
class StreakInfo:
    """Information about user's learning streak."""
    user_id: str
    current_streak: int
    longest_streak: int
    last_activity_date: datetime
    streak_start_date: datetime
    is_active: bool
    streak_type: str  # 'daily', 'weekly'


@dataclass
class ProgressVisualization:
    """Data structure for progress visualizations."""
    chart_type: str
    title: str
    data: Dict[str, Any]
    chart_image: Optional[str] = None  # Base64 encoded image
    interactive_data: Optional[Dict] = None


@dataclass
class LearningInsight:
    """Actionable insight derived from learning analytics."""
    insight_type: str
    title: str
    description: str
    recommendation: str
    confidence: float
    supporting_data: Dict[str, Any]


@dataclass
class ProgressReport:
    """Comprehensive progress report."""
    user_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    summary_stats: Dict[str, Any]
    visualizations: List[ProgressVisualization]
    achievements: List[Achievement]
    insights: List[LearningInsight]
    generated_at: datetime


class ProgressAnalytics:
    """Main class for progress analytics and reporting."""
    
    def __init__(self, progress_db: UserProgressDatabase, learning_engine: AdaptiveLearningEngine):
        """Initialize progress analytics system."""
        self.progress_db = progress_db
        self.learning_engine = learning_engine
        self.achievement_definitions = self._initialize_achievements()
    
    def _initialize_achievements(self) -> Dict[str, Dict]:
        """Initialize achievement definitions."""
        return {
            'first_problem': {
                'type': AchievementType.PROBLEMS_SOLVED,
                'title': 'First Steps',
                'description': 'Solved your first math problem!',
                'icon': '🎯',
                'points': 10,
                'requirements': {'problems_solved': 1}
            },
            'problem_solver_10': {
                'type': AchievementType.PROBLEMS_SOLVED,
                'title': 'Problem Solver',
                'description': 'Solved 10 math problems',
                'icon': '🧮',
                'points': 50,
                'requirements': {'problems_solved': 10}
            },
            'problem_solver_50': {
                'type': AchievementType.PROBLEMS_SOLVED,
                'title': 'Math Enthusiast',
                'description': 'Solved 50 math problems',
                'icon': '📚',
                'points': 200,
                'requirements': {'problems_solved': 50}
            },
            'problem_solver_100': {
                'type': AchievementType.PROBLEMS_SOLVED,
                'title': 'Century Club',
                'description': 'Solved 100 math problems',
                'icon': '💯',
                'points': 500,
                'requirements': {'problems_solved': 100},
                'is_milestone': True
            },
            'streak_3': {
                'type': AchievementType.STREAK_MILESTONE,
                'title': 'Getting Started',
                'description': 'Maintained a 3-day learning streak',
                'icon': '🔥',
                'points': 30,
                'requirements': {'streak_days': 3}
            },
            'streak_7': {
                'type': AchievementType.STREAK_MILESTONE,
                'title': 'Week Warrior',
                'description': 'Maintained a 7-day learning streak',
                'icon': '⚡',
                'points': 100,
                'requirements': {'streak_days': 7}
            },
            'streak_30': {
                'type': AchievementType.STREAK_MILESTONE,
                'title': 'Consistency Champion',
                'description': 'Maintained a 30-day learning streak',
                'icon': '👑',
                'points': 500,
                'requirements': {'streak_days': 30},
                'is_milestone': True
            },
            'accuracy_master': {
                'type': AchievementType.ACCURACY_MILESTONE,
                'title': 'Accuracy Master',
                'description': 'Achieved 90%+ accuracy over 20 problems',
                'icon': '🎯',
                'points': 300,
                'requirements': {'accuracy': 0.9, 'min_problems': 20}
            },
            'algebra_master': {
                'type': AchievementType.TOPIC_MASTERY,
                'title': 'Algebra Master',
                'description': 'Mastered algebra fundamentals',
                'icon': '🔢',
                'points': 400,
                'requirements': {'topic': 'algebra', 'mastery_level': 0.8}
            },
            'calculus_master': {
                'type': AchievementType.TOPIC_MASTERY,
                'title': 'Calculus Conqueror',
                'description': 'Mastered calculus concepts',
                'icon': '∫',
                'points': 600,
                'requirements': {'topic': 'calculus', 'mastery_level': 0.8}
            },
            'speed_demon': {
                'type': AchievementType.SPEED_DEMON,
                'title': 'Speed Demon',
                'description': 'Solved 10 problems in under 5 minutes each',
                'icon': '⚡',
                'points': 250,
                'requirements': {'fast_problems': 10, 'max_time': 300}
            },
            'dedicated_learner': {
                'type': AchievementType.TIME_DEDICATION,
                'title': 'Dedicated Learner',
                'description': 'Spent 10+ hours learning math',
                'icon': '⏰',
                'points': 300,
                'requirements': {'total_time_hours': 10}
            }
        }
    
    def check_and_award_achievements(self, user_id: str) -> List[Achievement]:
        """Check for new achievements and award them."""
        try:
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                return []
            
            analytics = self.progress_db.get_learning_analytics(user_id, 365)  # Full year
            new_achievements = []
            
            # Get existing achievements (would need to be stored in database)
            existing_achievements = self._get_user_achievements(user_id)
            existing_ids = {ach.id for ach in existing_achievements}
            
            for achievement_id, definition in self.achievement_definitions.items():
                if achievement_id in existing_ids:
                    continue  # Already earned
                
                if self._check_achievement_requirements(profile, analytics, definition):
                    achievement = Achievement(
                        id=achievement_id,
                        user_id=user_id,
                        achievement_type=definition['type'],
                        title=definition['title'],
                        description=definition['description'],
                        icon=definition['icon'],
                        points=definition['points'],
                        earned_at=datetime.now(),
                        requirements_met=definition['requirements'],
                        is_milestone=definition.get('is_milestone', False)
                    )
                    new_achievements.append(achievement)
                    self._save_achievement(achievement)
            
            return new_achievements
            
        except Exception as e:
            logger.error(f"Failed to check achievements: {e}")
            return []
    
    def _check_achievement_requirements(self, profile: UserProfile, analytics: Dict, definition: Dict) -> bool:
        """Check if achievement requirements are met."""
        requirements = definition['requirements']
        
        if definition['type'] == AchievementType.PROBLEMS_SOLVED:
            return profile.progress_metrics.total_problems_solved >= requirements['problems_solved']
        
        elif definition['type'] == AchievementType.STREAK_MILESTONE:
            return profile.progress_metrics.longest_streak >= requirements['streak_days']
        
        elif definition['type'] == AchievementType.ACCURACY_MILESTONE:
            return (profile.progress_metrics.average_accuracy >= requirements['accuracy'] and
                    profile.progress_metrics.total_problems_solved >= requirements['min_problems'])
        
        elif definition['type'] == AchievementType.TOPIC_MASTERY:
            topic = requirements['topic']
            # Check if user has high mastery in topic
            for domain_data in analytics.get('domain_performance', []):
                if domain_data['domain'] == topic:
                    return domain_data['accuracy'] >= requirements['mastery_level']
            return False
        
        elif definition['type'] == AchievementType.TIME_DEDICATION:
            total_hours = profile.progress_metrics.time_spent_learning / 60.0
            return total_hours >= requirements['total_time_hours']
        
        elif definition['type'] == AchievementType.SPEED_DEMON:
            # Check for fast problem solving (simplified)
            fast_count = 0
            for day in analytics.get('daily_activity', []):
                if day.get('avg_time', 0) <= requirements['max_time']:
                    fast_count += day.get('problems_solved', 0)
            return fast_count >= requirements['fast_problems']
        
        return False
    
    def _get_user_achievements(self, user_id: str) -> List[Achievement]:
        """Get user's existing achievements (placeholder - would query database)."""
        # In a real implementation, this would query the achievements table
        return []
    
    def _save_achievement(self, achievement: Achievement):
        """Save achievement to database (placeholder)."""
        # In a real implementation, this would save to achievements table
        logger.info(f"Achievement earned: {achievement.title} by user {achievement.user_id}")
    
    def get_streak_info(self, user_id: str) -> StreakInfo:
        """Get detailed streak information for user."""
        try:
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                return StreakInfo(
                    user_id=user_id,
                    current_streak=0,
                    longest_streak=0,
                    last_activity_date=datetime.now(),
                    streak_start_date=datetime.now(),
                    is_active=False,
                    streak_type='daily'
                )
            
            analytics = self.progress_db.get_learning_analytics(user_id, 30)
            daily_activity = analytics.get('daily_activity', [])
            
            # Calculate streak start date
            current_streak = profile.progress_metrics.current_streak
            streak_start_date = datetime.now() - timedelta(days=current_streak)
            
            # Determine if streak is active (activity in last 24 hours)
            last_activity = datetime.now() - timedelta(days=1)
            if daily_activity:
                last_activity_str = daily_activity[-1].get('date', '')
                if last_activity_str:
                    last_activity = datetime.strptime(last_activity_str, '%Y-%m-%d')
            
            is_active = (datetime.now() - last_activity).days <= 1
            
            return StreakInfo(
                user_id=user_id,
                current_streak=current_streak,
                longest_streak=profile.progress_metrics.longest_streak,
                last_activity_date=last_activity,
                streak_start_date=streak_start_date,
                is_active=is_active,
                streak_type='daily'
            )
            
        except Exception as e:
            logger.error(f"Failed to get streak info: {e}")
            return StreakInfo(
                user_id=user_id,
                current_streak=0,
                longest_streak=0,
                last_activity_date=datetime.now(),
                streak_start_date=datetime.now(),
                is_active=False,
                streak_type='daily'
            )
    
    def generate_progress_visualizations(self, user_id: str, days: int = 30) -> List[ProgressVisualization]:
        """Generate various progress visualizations."""
        try:
            analytics = self.progress_db.get_learning_analytics(user_id, days)
            visualizations = []
            
            # 1. Daily Activity Chart
            daily_viz = self._create_daily_activity_chart(analytics['daily_activity'])
            visualizations.append(daily_viz)
            
            # 2. Domain Performance Chart
            domain_viz = self._create_domain_performance_chart(analytics['domain_performance'])
            visualizations.append(domain_viz)
            
            # 3. Difficulty Progress Chart
            difficulty_viz = self._create_difficulty_progress_chart(analytics['difficulty_progression'])
            visualizations.append(difficulty_viz)
            
            # 4. Accuracy Trend Chart
            accuracy_viz = self._create_accuracy_trend_chart(analytics['daily_activity'])
            visualizations.append(accuracy_viz)
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            return []
    
    def _create_daily_activity_chart(self, daily_activity: List[Dict]) -> ProgressVisualization:
        """Create daily activity visualization."""
        if not daily_activity:
            return ProgressVisualization(
                chart_type='daily_activity',
                title='Daily Activity',
                data={'message': 'No activity data available'}
            )
        
        # Extract data
        dates = [datetime.strptime(day['date'], '%Y-%m-%d') for day in daily_activity]
        problems_attempted = [day.get('problems_attempted', 0) for day in daily_activity]
        problems_solved = [day.get('problems_solved', 0) for day in daily_activity]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(dates, problems_attempted, alpha=0.6, label='Attempted', color='lightblue')
        ax.bar(dates, problems_solved, alpha=0.8, label='Solved', color='darkblue')
        
        ax.set_title('Daily Problem Solving Activity')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Problems')
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return ProgressVisualization(
            chart_type='daily_activity',
            title='Daily Problem Solving Activity',
            data={
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'attempted': problems_attempted,
                'solved': problems_solved
            },
            chart_image=chart_image
        )
    
    def _create_domain_performance_chart(self, domain_performance: List[Dict]) -> ProgressVisualization:
        """Create domain performance visualization."""
        if not domain_performance:
            return ProgressVisualization(
                chart_type='domain_performance',
                title='Domain Performance',
                data={'message': 'No domain performance data available'}
            )
        
        # Extract data
        domains = [d['domain'] for d in domain_performance]
        accuracies = [d['accuracy'] for d in domain_performance]
        attempts = [d['attempts'] for d in domain_performance]
        
        # Create chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy chart
        bars1 = ax1.bar(domains, accuracies, color='green', alpha=0.7)
        ax1.set_title('Accuracy by Domain')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom')
        
        # Attempts chart
        bars2 = ax2.bar(domains, attempts, color='blue', alpha=0.7)
        ax2.set_title('Practice Volume by Domain')
        ax2.set_ylabel('Number of Attempts')
        
        # Add value labels on bars
        for bar, att in zip(bars2, attempts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(att), ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return ProgressVisualization(
            chart_type='domain_performance',
            title='Performance by Mathematical Domain',
            data={
                'domains': domains,
                'accuracies': accuracies,
                'attempts': attempts
            },
            chart_image=chart_image
        )
    
    def _create_difficulty_progress_chart(self, difficulty_progression: List[Dict]) -> ProgressVisualization:
        """Create difficulty progression visualization."""
        if not difficulty_progression:
            return ProgressVisualization(
                chart_type='difficulty_progress',
                title='Difficulty Progression',
                data={'message': 'No difficulty progression data available'}
            )
        
        # Extract data
        levels = [d['difficulty_level'] for d in difficulty_progression]
        accuracies = [d['accuracy'] for d in difficulty_progression]
        attempts = [d['attempts'] for d in difficulty_progression]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create bubble chart (size = attempts, y = accuracy)
        scatter = ax.scatter(levels, accuracies, s=[a*10 for a in attempts], 
                           alpha=0.6, c=levels, cmap='viridis')
        
        ax.set_title('Performance vs Difficulty Level')
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        ax.set_xlim(0.5, 4.5)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(['Beginner', 'Intermediate', 'Advanced', 'Expert'])
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Difficulty Level')
        
        # Add annotations
        for level, acc, att in zip(levels, accuracies, attempts):
            ax.annotate(f'{att} attempts', (level, acc), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return ProgressVisualization(
            chart_type='difficulty_progress',
            title='Performance vs Difficulty Level',
            data={
                'levels': levels,
                'accuracies': accuracies,
                'attempts': attempts
            },
            chart_image=chart_image
        )
    
    def _create_accuracy_trend_chart(self, daily_activity: List[Dict]) -> ProgressVisualization:
        """Create accuracy trend visualization."""
        if not daily_activity:
            return ProgressVisualization(
                chart_type='accuracy_trend',
                title='Accuracy Trend',
                data={'message': 'No activity data available'}
            )
        
        # Calculate daily accuracy
        dates = []
        accuracies = []
        
        for day in daily_activity:
            attempted = day.get('problems_attempted', 0)
            solved = day.get('problems_solved', 0)
            if attempted > 0:
                dates.append(datetime.strptime(day['date'], '%Y-%m-%d'))
                accuracies.append(solved / attempted)
        
        if not dates:
            return ProgressVisualization(
                chart_type='accuracy_trend',
                title='Accuracy Trend',
                data={'message': 'No accuracy data available'}
            )
        
        # Create chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(dates, accuracies, marker='o', linewidth=2, markersize=6, color='green')
        
        # Add trend line
        if len(dates) > 1:
            x_numeric = mdates.date2num(dates)
            z = np.polyfit(x_numeric, accuracies, 1)
            p = np.poly1d(z)
            ax.plot(dates, p(x_numeric), "--", alpha=0.7, color='red', label='Trend')
            ax.legend()
        
        ax.set_title('Accuracy Trend Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        chart_image = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return ProgressVisualization(
            chart_type='accuracy_trend',
            title='Accuracy Trend Over Time',
            data={
                'dates': [d.strftime('%Y-%m-%d') for d in dates],
                'accuracies': accuracies
            },
            chart_image=chart_image
        )
    
    def generate_learning_insights(self, user_id: str) -> List[LearningInsight]:
        """Generate actionable learning insights."""
        try:
            profile = self.progress_db.get_user_profile(user_id)
            if not profile:
                return []
            
            analytics = self.progress_db.get_learning_analytics(user_id, 30)
            performance_analysis = self.learning_engine.analyze_user_performance(user_id)
            
            insights = []
            
            # 1. Consistency Insight
            daily_activity = analytics.get('daily_activity', [])
            if len(daily_activity) >= 7:
                active_days = len([d for d in daily_activity if d.get('problems_attempted', 0) > 0])
                consistency_rate = active_days / len(daily_activity)
                
                if consistency_rate < 0.5:
                    insights.append(LearningInsight(
                        insight_type='consistency',
                        title='Improve Learning Consistency',
                        description=f'You\'ve been active on {active_days} out of {len(daily_activity)} days',
                        recommendation='Try to practice a little bit every day, even if just for 10 minutes',
                        confidence=0.8,
                        supporting_data={'active_days': active_days, 'total_days': len(daily_activity)}
                    ))
            
            # 2. Difficulty Adjustment Insight
            if performance_analysis.recommended_difficulty != self._get_current_avg_difficulty(profile):
                if performance_analysis.recommended_difficulty > self._get_current_avg_difficulty(profile):
                    insights.append(LearningInsight(
                        insight_type='difficulty',
                        title='Ready for More Challenge',
                        description='Your recent performance suggests you can handle harder problems',
                        recommendation=f'Try Level {performance_analysis.recommended_difficulty} problems',
                        confidence=0.7,
                        supporting_data={'current_accuracy': performance_analysis.overall_accuracy}
                    ))
                else:
                    insights.append(LearningInsight(
                        insight_type='difficulty',
                        title='Consider Easier Problems',
                        description='You might benefit from practicing easier problems to build confidence',
                        recommendation=f'Focus on Level {performance_analysis.recommended_difficulty} problems',
                        confidence=0.6,
                        supporting_data={'current_accuracy': performance_analysis.overall_accuracy}
                    ))
            
            # 3. Weak Area Focus Insight
            if performance_analysis.weaknesses:
                weakest_area = performance_analysis.weaknesses[0]
                insights.append(LearningInsight(
                    insight_type='focus_area',
                    title=f'Focus on {weakest_area.title()}',
                    description=f'Your performance in {weakest_area} could use improvement',
                    recommendation=f'Spend extra time practicing {weakest_area} problems',
                    confidence=0.8,
                    supporting_data={'weak_areas': performance_analysis.weaknesses}
                ))
            
            # 4. Streak Motivation Insight
            streak_info = self.get_streak_info(user_id)
            if not streak_info.is_active and streak_info.current_streak > 0:
                insights.append(LearningInsight(
                    insight_type='motivation',
                    title='Keep Your Streak Alive!',
                    description=f'You had a {streak_info.current_streak}-day streak going',
                    recommendation='Solve just one problem today to restart your learning streak',
                    confidence=0.9,
                    supporting_data={'previous_streak': streak_info.current_streak}
                ))
            
            # 5. Speed Improvement Insight
            avg_time = self._calculate_average_problem_time(analytics)
            if avg_time > 300:  # More than 5 minutes per problem
                insights.append(LearningInsight(
                    insight_type='efficiency',
                    title='Work on Problem-Solving Speed',
                    description=f'Your average time per problem is {avg_time//60:.1f} minutes',
                    recommendation='Try timing yourself and aim to solve problems more quickly',
                    confidence=0.6,
                    supporting_data={'avg_time_seconds': avg_time}
                ))
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            return []
    
    def _get_current_avg_difficulty(self, profile: UserProfile) -> float:
        """Calculate user's current average difficulty level."""
        skill_levels = profile.skill_levels
        if not skill_levels:
            return 1.0
        return sum(skill_levels.values()) / len(skill_levels)
    
    def _calculate_average_problem_time(self, analytics: Dict) -> float:
        """Calculate average time per problem."""
        daily_activity = analytics.get('daily_activity', [])
        total_time = 0
        total_problems = 0
        
        for day in daily_activity:
            problems = day.get('problems_attempted', 0)
            avg_time = day.get('avg_time', 0)
            if problems > 0 and avg_time > 0:
                total_time += problems * avg_time
                total_problems += problems
        
        return total_time / max(total_problems, 1)
    
    def generate_progress_report(self, user_id: str, report_type: ReportType, 
                               days: int = None) -> ProgressReport:
        """Generate comprehensive progress report."""
        try:
            # Determine period based on report type
            if days is None:
                days = {
                    ReportType.DAILY_SUMMARY: 1,
                    ReportType.WEEKLY_REPORT: 7,
                    ReportType.MONTHLY_REPORT: 30,
                    ReportType.TOPIC_PROGRESS: 30,
                    ReportType.PERFORMANCE_ANALYSIS: 30
                }.get(report_type, 30)
            
            period_end = datetime.now()
            period_start = period_end - timedelta(days=days)
            
            # Gather data
            profile = self.progress_db.get_user_profile(user_id)
            analytics = self.progress_db.get_learning_analytics(user_id, days)
            visualizations = self.generate_progress_visualizations(user_id, days)
            achievements = self.check_and_award_achievements(user_id)
            insights = self.generate_learning_insights(user_id)
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(profile, analytics, days)
            
            return ProgressReport(
                user_id=user_id,
                report_type=report_type,
                period_start=period_start,
                period_end=period_end,
                summary_stats=summary_stats,
                visualizations=visualizations,
                achievements=achievements,
                insights=insights,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Failed to generate progress report: {e}")
            # Return minimal report
            return ProgressReport(
                user_id=user_id,
                report_type=report_type,
                period_start=datetime.now() - timedelta(days=days or 30),
                period_end=datetime.now(),
                summary_stats={},
                visualizations=[],
                achievements=[],
                insights=[],
                generated_at=datetime.now()
            )
    
    def _calculate_summary_stats(self, profile: UserProfile, analytics: Dict, days: int) -> Dict[str, Any]:
        """Calculate summary statistics for the report period."""
        if not profile:
            return {}
        
        daily_activity = analytics.get('daily_activity', [])
        
        # Calculate period statistics
        period_problems_attempted = sum(day.get('problems_attempted', 0) for day in daily_activity)
        period_problems_solved = sum(day.get('problems_solved', 0) for day in daily_activity)
        period_accuracy = period_problems_solved / max(period_problems_attempted, 1)
        
        active_days = len([d for d in daily_activity if d.get('problems_attempted', 0) > 0])
        
        # Calculate time spent
        total_time_minutes = sum(
            day.get('problems_attempted', 0) * day.get('avg_time', 0) / 60
            for day in daily_activity
        )
        
        return {
            'period_days': days,
            'problems_attempted': period_problems_attempted,
            'problems_solved': period_problems_solved,
            'accuracy': period_accuracy,
            'active_days': active_days,
            'time_spent_minutes': total_time_minutes,
            'current_streak': profile.progress_metrics.current_streak,
            'longest_streak': profile.progress_metrics.longest_streak,
            'total_problems_solved': profile.progress_metrics.total_problems_solved,
            'overall_accuracy': profile.progress_metrics.average_accuracy,
            'domains_practiced': len(analytics.get('domain_performance', [])),
            'improvement_areas': len([d for d in analytics.get('domain_performance', []) 
                                    if d.get('accuracy', 0) < 0.7])
        }


# Utility functions for analytics
def calculate_learning_momentum(user_id: str, progress_db: UserProgressDatabase, days: int = 14) -> float:
    """Calculate user's learning momentum (trend in recent activity)."""
    try:
        analytics = progress_db.get_learning_analytics(user_id, days)
        daily_activity = analytics.get('daily_activity', [])
        
        if len(daily_activity) < 3:
            return 0.0
        
        # Calculate daily scores (problems solved)
        daily_scores = [day.get('problems_solved', 0) for day in daily_activity]
        
        # Calculate trend using linear regression
        x = np.arange(len(daily_scores))
        if len(daily_scores) > 1:
            trend = np.polyfit(x, daily_scores, 1)[0]
            return max(-1.0, min(1.0, trend))  # Normalize to [-1, 1]
        
        return 0.0
        
    except Exception as e:
        logger.error(f"Failed to calculate learning momentum: {e}")
        return 0.0


def generate_study_recommendations(progress_report: ProgressReport) -> List[str]:
    """Generate study recommendations based on progress report."""
    recommendations = []
    
    stats = progress_report.summary_stats
    insights = progress_report.insights
    
    # Based on accuracy
    accuracy = stats.get('accuracy', 0)
    if accuracy < 0.6:
        recommendations.append("Focus on understanding concepts before attempting more problems")
    elif accuracy > 0.85:
        recommendations.append("You're doing great! Consider trying more challenging problems")
    
    # Based on consistency
    active_days = stats.get('active_days', 0)
    period_days = stats.get('period_days', 30)
    if active_days / period_days < 0.5:
        recommendations.append("Try to practice more consistently, even if just for a few minutes daily")
    
    # Based on insights
    for insight in insights:
        if insight.insight_type == 'focus_area':
            recommendations.append(insight.recommendation)
    
    # Based on streak
    current_streak = stats.get('current_streak', 0)
    if current_streak == 0:
        recommendations.append("Start a learning streak by solving at least one problem daily")
    elif current_streak >= 7:
        recommendations.append("Great streak! Keep it up to build lasting habits")
    
    return recommendations[:5]  # Limit to top 5 recommendations