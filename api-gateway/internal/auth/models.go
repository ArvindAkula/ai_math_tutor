package auth

import (
	"time"

	"github.com/google/uuid"
)

// User represents a user in the system
type User struct {
	ID           uuid.UUID `json:"id" db:"id"`
	Email        string    `json:"email" db:"email"`
	Username     string    `json:"username" db:"username"`
	PasswordHash string    `json:"-" db:"password_hash"`
	Role         UserRole  `json:"role" db:"role"`
	CreatedAt    time.Time `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time `json:"updated_at" db:"updated_at"`
	IsActive     bool      `json:"is_active" db:"is_active"`
	LastLoginAt  *time.Time `json:"last_login_at,omitempty" db:"last_login_at"`
}

// UserRole represents different user roles in the system
type UserRole string

const (
	RoleStudent   UserRole = "student"
	RoleEducator  UserRole = "educator"
	RoleAdmin     UserRole = "admin"
)

// UserProfile represents extended user profile information
type UserProfile struct {
	UserID          uuid.UUID              `json:"user_id" db:"user_id"`
	SkillLevels     map[string]int         `json:"skill_levels" db:"skill_levels"`
	LearningGoals   []string               `json:"learning_goals" db:"learning_goals"`
	Preferences     map[string]interface{} `json:"preferences" db:"preferences"`
	TotalProblems   int                    `json:"total_problems_solved" db:"total_problems_solved"`
	CurrentStreak   int                    `json:"current_streak" db:"current_streak"`
	CreatedAt       time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at" db:"updated_at"`
}

// Session represents a user session
type Session struct {
	ID           uuid.UUID `json:"id" db:"id"`
	UserID       uuid.UUID `json:"user_id" db:"user_id"`
	RefreshToken string    `json:"-" db:"refresh_token"`
	ExpiresAt    time.Time `json:"expires_at" db:"expires_at"`
	CreatedAt    time.Time `json:"created_at" db:"created_at"`
	IsRevoked    bool      `json:"is_revoked" db:"is_revoked"`
	UserAgent    string    `json:"user_agent" db:"user_agent"`
	IPAddress    string    `json:"ip_address" db:"ip_address"`
}

// RegisterRequest represents a user registration request
type RegisterRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Username string `json:"username" binding:"required,min=3,max=50"`
	Password string `json:"password" binding:"required,min=8"`
	Role     UserRole `json:"role,omitempty"`
}

// LoginRequest represents a user login request
type LoginRequest struct {
	Email    string `json:"email" binding:"required,email"`
	Password string `json:"password" binding:"required"`
}

// LoginResponse represents a successful login response
type LoginResponse struct {
	User         *User  `json:"user"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int64  `json:"expires_in"`
}

// RefreshTokenRequest represents a token refresh request
type RefreshTokenRequest struct {
	RefreshToken string `json:"refresh_token" binding:"required"`
}

// UpdateProfileRequest represents a profile update request
type UpdateProfileRequest struct {
	Username      *string                `json:"username,omitempty"`
	SkillLevels   map[string]int         `json:"skill_levels,omitempty"`
	LearningGoals []string               `json:"learning_goals,omitempty"`
	Preferences   map[string]interface{} `json:"preferences,omitempty"`
}

// ChangePasswordRequest represents a password change request
type ChangePasswordRequest struct {
	CurrentPassword string `json:"current_password" binding:"required"`
	NewPassword     string `json:"new_password" binding:"required,min=8"`
}

// PasswordResetToken represents a password reset token
type PasswordResetToken struct {
	ID        uuid.UUID  `json:"id" db:"id"`
	UserID    uuid.UUID  `json:"user_id" db:"user_id"`
	TokenHash string     `json:"-" db:"token_hash"`
	ExpiresAt time.Time  `json:"expires_at" db:"expires_at"`
	UsedAt    *time.Time `json:"used_at,omitempty" db:"used_at"`
	CreatedAt time.Time  `json:"created_at" db:"created_at"`
	IPAddress string     `json:"ip_address" db:"ip_address"`
	UserAgent string     `json:"user_agent" db:"user_agent"`
}

// ForgotPasswordRequest represents a password reset request
type ForgotPasswordRequest struct {
	Email string `json:"email" binding:"required,email"`
}

// ResetPasswordRequest represents a password reset with token
type ResetPasswordRequest struct {
	Token       string `json:"token" binding:"required"`
	NewPassword string `json:"new_password" binding:"required,min=8"`
}

// ForgotPasswordResponse represents the response to a reset request
type ForgotPasswordResponse struct {
	Message string `json:"message"`
}

// JWTClaims represents JWT token claims
type JWTClaims struct {
	UserID   uuid.UUID `json:"user_id"`
	Email    string    `json:"email"`
	Username string    `json:"username"`
	Role     UserRole  `json:"role"`
	Type     string    `json:"type"` // "access" or "refresh"
}

// IsValid checks if the user role is valid
func (r UserRole) IsValid() bool {
	switch r {
	case RoleStudent, RoleEducator, RoleAdmin:
		return true
	default:
		return false
	}
}

// HasPermission checks if the user role has the required permission
func (r UserRole) HasPermission(requiredRole UserRole) bool {
	roleHierarchy := map[UserRole]int{
		RoleStudent:  1,
		RoleEducator: 2,
		RoleAdmin:    3,
	}

	userLevel, userExists := roleHierarchy[r]
	requiredLevel, requiredExists := roleHierarchy[requiredRole]

	if !userExists || !requiredExists {
		return false
	}

	return userLevel >= requiredLevel
}

// ToPublic returns a user struct without sensitive information
func (u *User) ToPublic() *User {
	return &User{
		ID:          u.ID,
		Email:       u.Email,
		Username:    u.Username,
		Role:        u.Role,
		CreatedAt:   u.CreatedAt,
		UpdatedAt:   u.UpdatedAt,
		IsActive:    u.IsActive,
		LastLoginAt: u.LastLoginAt,
	}
}