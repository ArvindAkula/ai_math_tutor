-- AI Math Tutor Database Schema
-- Initialize the database with core tables

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) DEFAULT 'student' CHECK (role IN ('student', 'educator', 'admin')),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User Profiles and Progress
CREATE TABLE user_profiles (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    skill_levels JSONB NOT NULL DEFAULT '{}',
    learning_goals TEXT[],
    preferences JSONB NOT NULL DEFAULT '{}',
    total_problems_solved INTEGER DEFAULT 0,
    current_streak INTEGER DEFAULT 0,
    longest_streak INTEGER DEFAULT 0,
    average_accuracy DECIMAL(5,2) DEFAULT 0.00,
    time_spent_learning INTEGER DEFAULT 0, -- minutes
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Mathematical Problems
CREATE TABLE problems (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    problem_text TEXT NOT NULL,
    domain VARCHAR(50) NOT NULL,
    difficulty_level INTEGER NOT NULL CHECK (difficulty_level BETWEEN 1 AND 4),
    solution_steps JSONB NOT NULL,
    correct_answer TEXT NOT NULL,
    tags TEXT[],
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- User Problem Attempts
CREATE TABLE problem_attempts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    problem_id UUID REFERENCES problems(id) ON DELETE CASCADE,
    user_answer TEXT,
    is_correct BOOLEAN NOT NULL,
    partial_credit DECIMAL(3,2) DEFAULT 0.00,
    time_taken INTEGER, -- seconds
    hints_used INTEGER DEFAULT 0,
    attempt_timestamp TIMESTAMP DEFAULT NOW()
);

-- Quizzes and Sessions
CREATE TABLE quiz_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    topic VARCHAR(100) NOT NULL,
    difficulty_level INTEGER NOT NULL CHECK (difficulty_level BETWEEN 1 AND 4),
    total_questions INTEGER NOT NULL,
    correct_answers INTEGER DEFAULT 0,
    total_points INTEGER DEFAULT 0,
    points_earned INTEGER DEFAULT 0,
    time_limit INTEGER, -- seconds
    started_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'completed', 'abandoned'))
);

-- Quiz Questions
CREATE TABLE quiz_questions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    quiz_session_id UUID REFERENCES quiz_sessions(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type VARCHAR(20) NOT NULL,
    options JSONB, -- for multiple choice questions
    correct_answer TEXT NOT NULL,
    user_answer TEXT,
    is_correct BOOLEAN,
    points_possible INTEGER DEFAULT 1,
    points_earned INTEGER DEFAULT 0,
    time_taken INTEGER, -- seconds
    hints_used INTEGER DEFAULT 0,
    answered_at TIMESTAMP
);

-- Learning Progress Tracking
CREATE TABLE learning_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    topic VARCHAR(100) NOT NULL,
    mastery_level DECIMAL(3,2) NOT NULL CHECK (mastery_level BETWEEN 0.00 AND 1.00),
    last_practiced TIMESTAMP DEFAULT NOW(),
    practice_count INTEGER DEFAULT 0,
    total_time_spent INTEGER DEFAULT 0, -- minutes
    UNIQUE(user_id, topic)
);

-- AI Explanations Cache
CREATE TABLE ai_explanations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    context_hash VARCHAR(64) UNIQUE NOT NULL,
    problem_id UUID REFERENCES problems(id) ON DELETE CASCADE,
    explanation_type VARCHAR(50) NOT NULL, -- 'step', 'hint', 'concept'
    content TEXT NOT NULL,
    complexity_level VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '2 hours'
);

-- Visualization Cache
CREATE TABLE visualizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    viz_hash VARCHAR(64) UNIQUE NOT NULL,
    problem_id UUID REFERENCES problems(id) ON DELETE CASCADE,
    plot_type VARCHAR(50) NOT NULL,
    plot_data JSONB NOT NULL,
    file_path VARCHAR(500), -- path to generated image/animation
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT NOW() + INTERVAL '30 minutes'
);

-- User Sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,
    expires_at TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    last_accessed TIMESTAMP DEFAULT NOW()
);

-- Password Reset Tokens
CREATE TABLE password_reset_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash VARCHAR(255) NOT NULL UNIQUE,
    expires_at TIMESTAMP NOT NULL,
    used_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    ip_address INET,
    user_agent TEXT
);

-- Indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_problem_attempts_user_id ON problem_attempts(user_id);
CREATE INDEX idx_problem_attempts_problem_id ON problem_attempts(problem_id);
CREATE INDEX idx_problem_attempts_timestamp ON problem_attempts(attempt_timestamp);
CREATE INDEX idx_quiz_sessions_user_id ON quiz_sessions(user_id);
CREATE INDEX idx_quiz_sessions_status ON quiz_sessions(status);
CREATE INDEX idx_learning_progress_user_id ON learning_progress(user_id);
CREATE INDEX idx_learning_progress_topic ON learning_progress(topic);
CREATE INDEX idx_ai_explanations_hash ON ai_explanations(context_hash);
CREATE INDEX idx_ai_explanations_expires ON ai_explanations(expires_at);
CREATE INDEX idx_visualizations_hash ON visualizations(viz_hash);
CREATE INDEX idx_visualizations_expires ON visualizations(expires_at);
CREATE INDEX idx_user_sessions_token ON user_sessions(session_token);
CREATE INDEX idx_user_sessions_expires ON user_sessions(expires_at);
CREATE INDEX idx_password_reset_tokens_hash ON password_reset_tokens(token_hash);
CREATE INDEX idx_password_reset_tokens_expires ON password_reset_tokens(expires_at);
CREATE INDEX idx_password_reset_tokens_user_id ON password_reset_tokens(user_id);

-- Sample data for development
INSERT INTO users (email, username, password_hash, role) VALUES 
('demo@example.com', 'demo_user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'student'), -- password: demo123
('student@example.com', 'student_user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'student'), -- password: demo123
('teacher@example.com', 'teacher_user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'educator'), -- password: demo123
('admin@example.com', 'admin_user', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'admin'), -- password: demo123
('alice@example.com', 'alice_student', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'student'), -- password: demo123
('bob@example.com', 'bob_educator', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6hsxq9w5KS', 'educator'); -- password: demo123

-- Create user profiles for all users
INSERT INTO user_profiles (user_id, skill_levels, learning_goals, preferences) 
SELECT id, 
    CASE 
        WHEN username = 'demo_user' THEN '{"algebra": 2, "calculus": 1}'
        WHEN username = 'student_user' THEN '{"algebra": 1, "geometry": 2}'
        WHEN username = 'alice_student' THEN '{"algebra": 3, "calculus": 2, "statistics": 1}'
        WHEN username = 'teacher_user' THEN '{"algebra": 4, "calculus": 4, "statistics": 3}'
        WHEN username = 'bob_educator' THEN '{"algebra": 4, "geometry": 4, "trigonometry": 3}'
        ELSE '{}'
    END::jsonb,
    CASE 
        WHEN username = 'demo_user' THEN ARRAY['Master calculus', 'Improve problem solving']
        WHEN username = 'student_user' THEN ARRAY['Learn basic algebra', 'Understand geometry']
        WHEN username = 'alice_student' THEN ARRAY['Advanced calculus', 'Statistics mastery']
        WHEN username = 'teacher_user' THEN ARRAY['Help students learn', 'Create engaging content']
        WHEN username = 'bob_educator' THEN ARRAY['Develop curriculum', 'Advanced mathematics']
        ELSE ARRAY[]::text[]
    END,
    CASE 
        WHEN role = 'student' THEN '{"preferred_explanation_level": "standard", "visual_learning": true, "practice_reminders": true}'
        WHEN role = 'educator' THEN '{"preferred_explanation_level": "detailed", "show_analytics": true, "advanced_features": true}'
        WHEN role = 'admin' THEN '{"preferred_explanation_level": "detailed", "show_analytics": true, "admin_dashboard": true}'
        ELSE '{}'
    END::jsonb
FROM users;

INSERT INTO problems (problem_text, domain, difficulty_level, solution_steps, correct_answer, tags) VALUES
('Solve for x: 2x + 3 = 7', 'algebra', 1, '[{"step": 1, "operation": "subtract 3", "result": "2x = 4"}, {"step": 2, "operation": "divide by 2", "result": "x = 2"}]', '2', ARRAY['linear_equation', 'basic_algebra']),
('Find the derivative of x^2 + 3x', 'calculus', 2, '[{"step": 1, "operation": "apply power rule", "result": "2x + 3"}]', '2x + 3', ARRAY['derivatives', 'power_rule']);

-- Clean up expired cache entries (run periodically)
CREATE OR REPLACE FUNCTION cleanup_expired_cache() RETURNS void AS $$
BEGIN
    DELETE FROM ai_explanations WHERE expires_at < NOW();
    DELETE FROM visualizations WHERE expires_at < NOW();
    DELETE FROM user_sessions WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;