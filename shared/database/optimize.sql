-- AI Math Tutor Database Optimization
-- Additional indexes and query optimizations for performance

-- ============================================================================
-- PERFORMANCE INDEXES
-- ============================================================================

-- User-related indexes for frequent queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at);

-- User profiles - frequently queried by skill levels and learning goals
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_profiles_skill_levels ON user_profiles USING GIN(skill_levels);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_profiles_learning_goals ON user_profiles USING GIN(learning_goals);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_profiles_updated_at ON user_profiles(updated_at);

-- Problems - frequently filtered by domain, difficulty, and tags
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problems_domain ON problems(domain);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problems_difficulty ON problems(difficulty_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problems_domain_difficulty ON problems(domain, difficulty_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problems_tags ON problems USING GIN(tags);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problems_metadata ON problems USING GIN(metadata);

-- Problem attempts - critical for analytics and progress tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_user_correct ON problem_attempts(user_id, is_correct);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_user_timestamp ON problem_attempts(user_id, attempt_timestamp DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_problem_correct ON problem_attempts(problem_id, is_correct);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_time_taken ON problem_attempts(time_taken) WHERE time_taken IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_hints_used ON problem_attempts(hints_used);

-- Quiz sessions - for user dashboard and analytics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_sessions_user_completed ON quiz_sessions(user_id, completed_at DESC) WHERE completed_at IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_sessions_topic_difficulty ON quiz_sessions(topic, difficulty_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_sessions_started_at ON quiz_sessions(started_at DESC);

-- Quiz questions - for session management
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_questions_session_answered ON quiz_questions(quiz_session_id, answered_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_questions_correct ON quiz_questions(is_correct) WHERE is_correct IS NOT NULL;

-- Learning progress - frequently queried for recommendations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_progress_user_mastery ON learning_progress(user_id, mastery_level DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_progress_topic_mastery ON learning_progress(topic, mastery_level DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_progress_last_practiced ON learning_progress(last_practiced DESC);

-- Cache tables - for efficient cleanup and retrieval
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_explanations_problem_type ON ai_explanations(problem_id, explanation_type);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ai_explanations_complexity ON ai_explanations(complexity_level);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_visualizations_problem_type ON visualizations(problem_id, plot_type);

-- User sessions - for authentication and session management
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_user_expires ON user_sessions(user_id, expires_at DESC);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_user_sessions_last_accessed ON user_sessions(last_accessed DESC);

-- ============================================================================
-- COMPOSITE INDEXES FOR COMPLEX QUERIES
-- ============================================================================

-- User performance analytics
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_attempts_analytics ON problem_attempts(user_id, attempt_timestamp DESC, is_correct, time_taken);

-- Quiz performance tracking
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_quiz_performance ON quiz_sessions(user_id, topic, difficulty_level, completed_at DESC) WHERE completed_at IS NOT NULL;

-- Learning path recommendations
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_learning_recommendations ON learning_progress(user_id, mastery_level, last_practiced DESC);

-- Problem difficulty analysis
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_problem_difficulty_analysis ON problem_attempts(problem_id, is_correct, time_taken, hints_used);

-- ============================================================================
-- PARTIAL INDEXES FOR SPECIFIC USE CASES
-- ============================================================================

-- Active user sessions only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_user_sessions ON user_sessions(user_id, last_accessed DESC) WHERE expires_at > NOW();

-- Recent problem attempts (last 30 days)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_problem_attempts ON problem_attempts(user_id, attempt_timestamp DESC) WHERE attempt_timestamp > NOW() - INTERVAL '30 days';

-- Completed quizzes only
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_completed_quizzes ON quiz_sessions(user_id, completed_at DESC, points_earned) WHERE status = 'completed';

-- Incorrect attempts for remediation
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_incorrect_attempts ON problem_attempts(user_id, problem_id, attempt_timestamp DESC) WHERE is_correct = false;

-- High-difficulty problems
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_advanced_problems ON problems(domain, created_at DESC) WHERE difficulty_level >= 3;

-- ============================================================================
-- MATERIALIZED VIEWS FOR ANALYTICS
-- ============================================================================

-- User performance summary
CREATE MATERIALIZED VIEW IF NOT EXISTS user_performance_summary AS
SELECT 
    u.id as user_id,
    u.username,
    up.total_problems_solved,
    up.current_streak,
    up.longest_streak,
    up.average_accuracy,
    up.time_spent_learning,
    COUNT(DISTINCT pa.problem_id) as unique_problems_attempted,
    COUNT(pa.id) as total_attempts,
    COUNT(pa.id) FILTER (WHERE pa.is_correct) as correct_attempts,
    ROUND(
        COUNT(pa.id) FILTER (WHERE pa.is_correct)::decimal / 
        NULLIF(COUNT(pa.id), 0) * 100, 2
    ) as calculated_accuracy,
    AVG(pa.time_taken) FILTER (WHERE pa.time_taken IS NOT NULL) as avg_time_per_problem,
    COUNT(DISTINCT qs.id) as quizzes_completed,
    AVG(qs.points_earned::decimal / NULLIF(qs.total_points, 0)) as avg_quiz_score
FROM users u
LEFT JOIN user_profiles up ON u.id = up.user_id
LEFT JOIN problem_attempts pa ON u.id = pa.user_id
LEFT JOIN quiz_sessions qs ON u.id = qs.user_id AND qs.status = 'completed'
WHERE u.is_active = true
GROUP BY u.id, u.username, up.total_problems_solved, up.current_streak, 
         up.longest_streak, up.average_accuracy, up.time_spent_learning;

-- Create index on materialized view
CREATE INDEX IF NOT EXISTS idx_user_performance_summary_accuracy ON user_performance_summary(calculated_accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_user_performance_summary_problems ON user_performance_summary(total_problems_solved DESC);

-- Topic mastery summary
CREATE MATERIALIZED VIEW IF NOT EXISTS topic_mastery_summary AS
SELECT 
    lp.topic,
    COUNT(DISTINCT lp.user_id) as users_practicing,
    AVG(lp.mastery_level) as avg_mastery_level,
    COUNT(lp.user_id) FILTER (WHERE lp.mastery_level >= 0.8) as users_mastered,
    AVG(lp.practice_count) as avg_practice_count,
    AVG(lp.total_time_spent) as avg_time_spent
FROM learning_progress lp
GROUP BY lp.topic;

-- Problem difficulty analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS problem_difficulty_analysis AS
SELECT 
    p.id as problem_id,
    p.domain,
    p.difficulty_level,
    COUNT(pa.id) as total_attempts,
    COUNT(pa.id) FILTER (WHERE pa.is_correct) as correct_attempts,
    ROUND(
        COUNT(pa.id) FILTER (WHERE pa.is_correct)::decimal / 
        NULLIF(COUNT(pa.id), 0) * 100, 2
    ) as success_rate,
    AVG(pa.time_taken) FILTER (WHERE pa.time_taken IS NOT NULL) as avg_time_taken,
    AVG(pa.hints_used) as avg_hints_used,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY pa.time_taken) as median_time_taken
FROM problems p
LEFT JOIN problem_attempts pa ON p.id = pa.problem_id
GROUP BY p.id, p.domain, p.difficulty_level
HAVING COUNT(pa.id) > 0;

-- ============================================================================
-- FUNCTIONS FOR QUERY OPTIMIZATION
-- ============================================================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_analytics_views() RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_performance_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY topic_mastery_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY problem_difficulty_analysis;
END;
$$ LANGUAGE plpgsql;

-- Function to get user learning recommendations
CREATE OR REPLACE FUNCTION get_user_learning_recommendations(p_user_id UUID, p_limit INTEGER DEFAULT 5)
RETURNS TABLE(
    topic VARCHAR(100),
    mastery_level DECIMAL(3,2),
    recommendation_reason TEXT,
    priority_score DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        lp.topic,
        lp.mastery_level,
        CASE 
            WHEN lp.mastery_level < 0.3 THEN 'Needs fundamental practice'
            WHEN lp.mastery_level < 0.6 THEN 'Ready for intermediate challenges'
            WHEN lp.mastery_level < 0.8 THEN 'Approaching mastery'
            ELSE 'Maintain proficiency'
        END as recommendation_reason,
        -- Priority score: lower mastery + recent practice = higher priority
        (1.0 - lp.mastery_level) * 
        CASE 
            WHEN lp.last_practiced > NOW() - INTERVAL '7 days' THEN 1.5
            WHEN lp.last_practiced > NOW() - INTERVAL '30 days' THEN 1.0
            ELSE 0.5
        END as priority_score
    FROM learning_progress lp
    WHERE lp.user_id = p_user_id
    ORDER BY priority_score DESC, lp.last_practiced ASC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- Function to get problem recommendations based on user performance
CREATE OR REPLACE FUNCTION get_problem_recommendations(
    p_user_id UUID, 
    p_domain VARCHAR(50) DEFAULT NULL,
    p_limit INTEGER DEFAULT 10
)
RETURNS TABLE(
    problem_id UUID,
    problem_text TEXT,
    domain VARCHAR(50),
    difficulty_level INTEGER,
    success_rate DECIMAL(5,2),
    recommendation_score DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    WITH user_stats AS (
        SELECT 
            pa.problem_id,
            COUNT(pa.id) as attempts,
            COUNT(pa.id) FILTER (WHERE pa.is_correct) as correct_attempts
        FROM problem_attempts pa
        WHERE pa.user_id = p_user_id
        GROUP BY pa.problem_id
    ),
    user_mastery AS (
        SELECT topic, mastery_level
        FROM learning_progress
        WHERE user_id = p_user_id
    )
    SELECT 
        p.id,
        p.problem_text,
        p.domain,
        p.difficulty_level,
        pda.success_rate,
        -- Recommendation score based on difficulty match and success rate
        CASE 
            WHEN us.problem_id IS NOT NULL THEN 0.0 -- Already attempted
            WHEN um.mastery_level IS NULL THEN 0.5 -- New domain
            WHEN p.difficulty_level = LEAST(4, GREATEST(1, ROUND(um.mastery_level * 4))) THEN 1.0
            WHEN ABS(p.difficulty_level - ROUND(um.mastery_level * 4)) = 1 THEN 0.8
            ELSE 0.3
        END * 
        CASE 
            WHEN pda.success_rate BETWEEN 40 AND 80 THEN 1.0 -- Sweet spot
            WHEN pda.success_rate > 80 THEN 0.6 -- Too easy
            WHEN pda.success_rate < 40 THEN 0.4 -- Too hard
            ELSE 0.5
        END as recommendation_score
    FROM problems p
    LEFT JOIN problem_difficulty_analysis pda ON p.id = pda.problem_id
    LEFT JOIN user_stats us ON p.id = us.problem_id
    LEFT JOIN user_mastery um ON p.domain = um.topic
    WHERE (p_domain IS NULL OR p.domain = p_domain)
        AND us.problem_id IS NULL -- Not already attempted
    ORDER BY recommendation_score DESC, pda.success_rate DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- QUERY PERFORMANCE MONITORING
-- ============================================================================

-- Enable query statistics collection
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Function to get slow queries
CREATE OR REPLACE FUNCTION get_slow_queries(p_limit INTEGER DEFAULT 10)
RETURNS TABLE(
    query TEXT,
    calls BIGINT,
    total_time DOUBLE PRECISION,
    mean_time DOUBLE PRECISION,
    rows BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        pss.query,
        pss.calls,
        pss.total_exec_time as total_time,
        pss.mean_exec_time as mean_time,
        pss.rows
    FROM pg_stat_statements pss
    WHERE pss.mean_exec_time > 100 -- queries taking more than 100ms on average
    ORDER BY pss.mean_exec_time DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- MAINTENANCE PROCEDURES
-- ============================================================================

-- Procedure to update table statistics
CREATE OR REPLACE FUNCTION update_table_statistics() RETURNS void AS $$
BEGIN
    ANALYZE users;
    ANALYZE user_profiles;
    ANALYZE problems;
    ANALYZE problem_attempts;
    ANALYZE quiz_sessions;
    ANALYZE quiz_questions;
    ANALYZE learning_progress;
    ANALYZE ai_explanations;
    ANALYZE visualizations;
    ANALYZE user_sessions;
END;
$$ LANGUAGE plpgsql;

-- Procedure to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data() RETURNS void AS $$
BEGIN
    -- Clean up expired cache entries
    DELETE FROM ai_explanations WHERE expires_at < NOW();
    DELETE FROM visualizations WHERE expires_at < NOW();
    DELETE FROM user_sessions WHERE expires_at < NOW();
    
    -- Archive old problem attempts (older than 1 year)
    -- In production, you might want to move these to an archive table instead
    DELETE FROM problem_attempts 
    WHERE attempt_timestamp < NOW() - INTERVAL '1 year';
    
    -- Clean up abandoned quiz sessions (older than 1 day)
    DELETE FROM quiz_sessions 
    WHERE status = 'active' 
    AND started_at < NOW() - INTERVAL '1 day';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- SCHEDULED MAINTENANCE (to be run by cron or scheduler)
-- ============================================================================

-- Create a maintenance log table
CREATE TABLE IF NOT EXISTS maintenance_log (
    id SERIAL PRIMARY KEY,
    operation VARCHAR(100) NOT NULL,
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',
    details TEXT
);

-- Comprehensive maintenance procedure
CREATE OR REPLACE FUNCTION run_maintenance() RETURNS void AS $$
DECLARE
    log_id INTEGER;
BEGIN
    -- Log maintenance start
    INSERT INTO maintenance_log (operation, status) 
    VALUES ('full_maintenance', 'running') 
    RETURNING id INTO log_id;
    
    BEGIN
        -- Update statistics
        PERFORM update_table_statistics();
        
        -- Refresh materialized views
        PERFORM refresh_analytics_views();
        
        -- Clean up old data
        PERFORM cleanup_old_data();
        
        -- Update log with success
        UPDATE maintenance_log 
        SET completed_at = NOW(), status = 'completed', 
            details = 'All maintenance tasks completed successfully'
        WHERE id = log_id;
        
    EXCEPTION WHEN OTHERS THEN
        -- Log error
        UPDATE maintenance_log 
        SET completed_at = NOW(), status = 'failed', 
            details = SQLERRM
        WHERE id = log_id;
        RAISE;
    END;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INITIAL OPTIMIZATION SETUP
-- ============================================================================

-- Set optimal PostgreSQL configuration parameters
-- Note: These should be adjusted based on your server specifications

-- Increase shared_buffers (25% of RAM is a good starting point)
-- ALTER SYSTEM SET shared_buffers = '256MB';

-- Increase effective_cache_size (75% of RAM)
-- ALTER SYSTEM SET effective_cache_size = '1GB';

-- Optimize for mixed workload
-- ALTER SYSTEM SET random_page_cost = 1.1;
-- ALTER SYSTEM SET effective_io_concurrency = 200;

-- Enable parallel query execution
-- ALTER SYSTEM SET max_parallel_workers_per_gather = 2;
-- ALTER SYSTEM SET max_parallel_workers = 8;

-- Optimize checkpoint behavior
-- ALTER SYSTEM SET checkpoint_completion_target = 0.9;
-- ALTER SYSTEM SET wal_buffers = '16MB';

-- Note: After changing these settings, you need to reload PostgreSQL configuration
-- SELECT pg_reload_conf();

-- Run initial maintenance
SELECT run_maintenance();

-- Create initial materialized view data
SELECT refresh_analytics_views();