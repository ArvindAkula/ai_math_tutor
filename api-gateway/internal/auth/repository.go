package auth

import (
	"database/sql"
	"encoding/json"
	"errors"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

var (
	ErrUserNotFound      = errors.New("user not found")
	ErrUserAlreadyExists = errors.New("user already exists")
	ErrSessionNotFound   = errors.New("session not found")
	ErrSessionExpired    = errors.New("session expired")
)

// Repository handles database operations for authentication
type Repository struct {
	db *sql.DB
}

// NewRepository creates a new auth repository
func NewRepository(db *sql.DB) *Repository {
	return &Repository{db: db}
}

// CreateUser creates a new user in the database
func (r *Repository) CreateUser(user *User) error {
	query := `
		INSERT INTO users (id, email, username, password_hash, role, created_at, updated_at, is_active)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`
	
	_, err := r.db.Exec(query,
		user.ID,
		user.Email,
		user.Username,
		user.PasswordHash,
		string(user.Role),
		user.CreatedAt,
		user.UpdatedAt,
		user.IsActive,
	)
	
	if err != nil {
		if pqErr, ok := err.(*pq.Error); ok {
			if pqErr.Code == "23505" { // unique_violation
				return ErrUserAlreadyExists
			}
		}
		return err
	}
	
	return nil
}

// GetUserByEmail retrieves a user by email
func (r *Repository) GetUserByEmail(email string) (*User, error) {
	user := &User{}
	query := `
		SELECT id, email, username, password_hash, role, created_at, updated_at, is_active, last_login_at
		FROM users
		WHERE email = $1 AND is_active = true
	`
	
	err := r.db.QueryRow(query, email).Scan(
		&user.ID,
		&user.Email,
		&user.Username,
		&user.PasswordHash,
		&user.Role,
		&user.CreatedAt,
		&user.UpdatedAt,
		&user.IsActive,
		&user.LastLoginAt,
	)
	
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrUserNotFound
		}
		return nil, err
	}
	
	return user, nil
}

// GetUserByID retrieves a user by ID
func (r *Repository) GetUserByID(userID uuid.UUID) (*User, error) {
	user := &User{}
	query := `
		SELECT id, email, username, password_hash, role, created_at, updated_at, is_active, last_login_at
		FROM users
		WHERE id = $1 AND is_active = true
	`
	
	err := r.db.QueryRow(query, userID).Scan(
		&user.ID,
		&user.Email,
		&user.Username,
		&user.PasswordHash,
		&user.Role,
		&user.CreatedAt,
		&user.UpdatedAt,
		&user.IsActive,
		&user.LastLoginAt,
	)
	
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrUserNotFound
		}
		return nil, err
	}
	
	return user, nil
}

// UpdateUser updates user information
func (r *Repository) UpdateUser(user *User) error {
	query := `
		UPDATE users
		SET username = $2, updated_at = $3, last_login_at = $4
		WHERE id = $1
	`
	
	result, err := r.db.Exec(query, user.ID, user.Username, user.UpdatedAt, user.LastLoginAt)
	if err != nil {
		return err
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	
	if rowsAffected == 0 {
		return ErrUserNotFound
	}
	
	return nil
}

// UpdateUserPassword updates user password
func (r *Repository) UpdateUserPassword(userID uuid.UUID, passwordHash string) error {
	query := `
		UPDATE users
		SET password_hash = $2, updated_at = $3
		WHERE id = $1
	`
	
	result, err := r.db.Exec(query, userID, passwordHash, time.Now())
	if err != nil {
		return err
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	
	if rowsAffected == 0 {
		return ErrUserNotFound
	}
	
	return nil
}

// CreateUserProfile creates a user profile
func (r *Repository) CreateUserProfile(profile *UserProfile) error {
	skillLevelsJSON, err := json.Marshal(profile.SkillLevels)
	if err != nil {
		return err
	}
	
	learningGoalsJSON, err := json.Marshal(profile.LearningGoals)
	if err != nil {
		return err
	}
	
	preferencesJSON, err := json.Marshal(profile.Preferences)
	if err != nil {
		return err
	}
	
	query := `
		INSERT INTO user_profiles (user_id, skill_levels, learning_goals, preferences, total_problems_solved, current_streak, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`
	
	_, err = r.db.Exec(query,
		profile.UserID,
		skillLevelsJSON,
		learningGoalsJSON,
		preferencesJSON,
		profile.TotalProblems,
		profile.CurrentStreak,
		profile.CreatedAt,
		profile.UpdatedAt,
	)
	
	return err
}

// GetUserProfile retrieves a user profile
func (r *Repository) GetUserProfile(userID uuid.UUID) (*UserProfile, error) {
	profile := &UserProfile{}
	var skillLevelsJSON, learningGoalsJSON, preferencesJSON []byte
	
	query := `
		SELECT user_id, skill_levels, learning_goals, preferences, total_problems_solved, current_streak, created_at, updated_at
		FROM user_profiles
		WHERE user_id = $1
	`
	
	err := r.db.QueryRow(query, userID).Scan(
		&profile.UserID,
		&skillLevelsJSON,
		&learningGoalsJSON,
		&preferencesJSON,
		&profile.TotalProblems,
		&profile.CurrentStreak,
		&profile.CreatedAt,
		&profile.UpdatedAt,
	)
	
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrUserNotFound
		}
		return nil, err
	}
	
	// Unmarshal JSON fields
	if err := json.Unmarshal(skillLevelsJSON, &profile.SkillLevels); err != nil {
		return nil, err
	}
	
	if err := json.Unmarshal(learningGoalsJSON, &profile.LearningGoals); err != nil {
		return nil, err
	}
	
	if err := json.Unmarshal(preferencesJSON, &profile.Preferences); err != nil {
		return nil, err
	}
	
	return profile, nil
}

// UpdateUserProfile updates a user profile
func (r *Repository) UpdateUserProfile(profile *UserProfile) error {
	skillLevelsJSON, err := json.Marshal(profile.SkillLevels)
	if err != nil {
		return err
	}
	
	learningGoalsJSON, err := json.Marshal(profile.LearningGoals)
	if err != nil {
		return err
	}
	
	preferencesJSON, err := json.Marshal(profile.Preferences)
	if err != nil {
		return err
	}
	
	query := `
		UPDATE user_profiles
		SET skill_levels = $2, learning_goals = $3, preferences = $4, updated_at = $5
		WHERE user_id = $1
	`
	
	result, err := r.db.Exec(query,
		profile.UserID,
		skillLevelsJSON,
		learningGoalsJSON,
		preferencesJSON,
		profile.UpdatedAt,
	)
	
	if err != nil {
		return err
	}
	
	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return err
	}
	
	if rowsAffected == 0 {
		return ErrUserNotFound
	}
	
	return nil
}

// CreateSession creates a new user session
func (r *Repository) CreateSession(session *Session) error {
	query := `
		INSERT INTO user_sessions (id, user_id, refresh_token, expires_at, created_at, is_revoked, user_agent, ip_address)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
	`
	
	_, err := r.db.Exec(query,
		session.ID,
		session.UserID,
		session.RefreshToken,
		session.ExpiresAt,
		session.CreatedAt,
		session.IsRevoked,
		session.UserAgent,
		session.IPAddress,
	)
	
	return err
}

// GetSessionByRefreshToken retrieves a session by refresh token
func (r *Repository) GetSessionByRefreshToken(refreshToken string) (*Session, error) {
	session := &Session{}
	query := `
		SELECT id, user_id, refresh_token, expires_at, created_at, is_revoked, user_agent, ip_address
		FROM user_sessions
		WHERE refresh_token = $1 AND is_revoked = false
	`
	
	err := r.db.QueryRow(query, refreshToken).Scan(
		&session.ID,
		&session.UserID,
		&session.RefreshToken,
		&session.ExpiresAt,
		&session.CreatedAt,
		&session.IsRevoked,
		&session.UserAgent,
		&session.IPAddress,
	)
	
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, ErrSessionNotFound
		}
		return nil, err
	}
	
	// Check if session is expired
	if time.Now().After(session.ExpiresAt) {
		return nil, ErrSessionExpired
	}
	
	return session, nil
}

// RevokeSession revokes a session
func (r *Repository) RevokeSession(sessionID uuid.UUID) error {
	query := `
		UPDATE user_sessions
		SET is_revoked = true
		WHERE id = $1
	`
	
	_, err := r.db.Exec(query, sessionID)
	return err
}

// RevokeAllUserSessions revokes all sessions for a user
func (r *Repository) RevokeAllUserSessions(userID uuid.UUID) error {
	query := `
		UPDATE user_sessions
		SET is_revoked = true
		WHERE user_id = $1 AND is_revoked = false
	`
	
	_, err := r.db.Exec(query, userID)
	return err
}

// CleanupExpiredSessions removes expired sessions from the database
func (r *Repository) CleanupExpiredSessions() error {
	query := `
		DELETE FROM user_sessions
		WHERE expires_at < $1 OR is_revoked = true
	`
	
	_, err := r.db.Exec(query, time.Now())
	return err
}