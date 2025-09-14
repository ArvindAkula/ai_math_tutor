package auth

import (
	"database/sql"
	"errors"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/bcrypt"
)

var (
	ErrInvalidCredentials = errors.New("invalid credentials")
	ErrWeakPassword      = errors.New("password is too weak")
)

// Service handles authentication business logic
type Service struct {
	repo       *Repository
	jwtService *JWTService
}

// NewService creates a new authentication service
func NewService(db *sql.DB) *Service {
	return &Service{
		repo:       NewRepository(db),
		jwtService: NewJWTService(),
	}
}

// Register creates a new user account
func (s *Service) Register(req *RegisterRequest) (*User, error) {
	// Validate password strength
	if err := s.validatePassword(req.Password); err != nil {
		return nil, err
	}

	// Set default role if not specified
	if req.Role == "" {
		req.Role = RoleStudent
	}

	// Validate role
	if !req.Role.IsValid() {
		req.Role = RoleStudent
	}

	// Hash password
	passwordHash, err := s.hashPassword(req.Password)
	if err != nil {
		return nil, err
	}

	// Create user
	user := &User{
		ID:           uuid.New(),
		Email:        req.Email,
		Username:     req.Username,
		PasswordHash: passwordHash,
		Role:         req.Role,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		IsActive:     true,
	}

	// Save user to database
	if err := s.repo.CreateUser(user); err != nil {
		return nil, err
	}

	// Create default user profile
	profile := &UserProfile{
		UserID:        user.ID,
		SkillLevels:   make(map[string]int),
		LearningGoals: []string{},
		Preferences:   make(map[string]interface{}),
		TotalProblems: 0,
		CurrentStreak: 0,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
	}

	if err := s.repo.CreateUserProfile(profile); err != nil {
		// Log error but don't fail registration
		// In production, you might want to handle this differently
	}

	return user.ToPublic(), nil
}

// Login authenticates a user and returns tokens
func (s *Service) Login(req *LoginRequest, userAgent, ipAddress string) (*LoginResponse, error) {
	// Get user by email
	user, err := s.repo.GetUserByEmail(req.Email)
	if err != nil {
		if err == ErrUserNotFound {
			return nil, ErrInvalidCredentials
		}
		return nil, err
	}

	// Verify password
	if err := s.verifyPassword(req.Password, user.PasswordHash); err != nil {
		return nil, ErrInvalidCredentials
	}

	// Generate token pair
	accessToken, refreshToken, err := s.jwtService.GenerateTokenPair(user)
	if err != nil {
		return nil, err
	}

	// Create session
	session := &Session{
		ID:           uuid.New(),
		UserID:       user.ID,
		RefreshToken: refreshToken,
		ExpiresAt:    time.Now().Add(s.jwtService.GetRefreshTokenTTL()),
		CreatedAt:    time.Now(),
		IsRevoked:    false,
		UserAgent:    userAgent,
		IPAddress:    ipAddress,
	}

	if err := s.repo.CreateSession(session); err != nil {
		return nil, err
	}

	// Update last login time
	now := time.Now()
	user.LastLoginAt = &now
	user.UpdatedAt = now
	if err := s.repo.UpdateUser(user); err != nil {
		// Log error but don't fail login
	}

	return &LoginResponse{
		User:         user.ToPublic(),
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		ExpiresIn:    int64(s.jwtService.GetAccessTokenTTL().Seconds()),
	}, nil
}

// RefreshToken generates new tokens using a refresh token
func (s *Service) RefreshToken(req *RefreshTokenRequest, userAgent, ipAddress string) (*LoginResponse, error) {
	// Validate refresh token
	claims, err := s.jwtService.ValidateRefreshToken(req.RefreshToken)
	if err != nil {
		return nil, err
	}

	// Get session
	session, err := s.repo.GetSessionByRefreshToken(req.RefreshToken)
	if err != nil {
		return nil, err
	}

	// Get user
	user, err := s.repo.GetUserByID(claims.UserID)
	if err != nil {
		return nil, err
	}

	// Generate new token pair
	accessToken, newRefreshToken, err := s.jwtService.GenerateTokenPair(user)
	if err != nil {
		return nil, err
	}

	// Revoke old session
	if err := s.repo.RevokeSession(session.ID); err != nil {
		return nil, err
	}

	// Create new session
	newSession := &Session{
		ID:           uuid.New(),
		UserID:       user.ID,
		RefreshToken: newRefreshToken,
		ExpiresAt:    time.Now().Add(s.jwtService.GetRefreshTokenTTL()),
		CreatedAt:    time.Now(),
		IsRevoked:    false,
		UserAgent:    userAgent,
		IPAddress:    ipAddress,
	}

	if err := s.repo.CreateSession(newSession); err != nil {
		return nil, err
	}

	return &LoginResponse{
		User:         user.ToPublic(),
		AccessToken:  accessToken,
		RefreshToken: newRefreshToken,
		ExpiresIn:    int64(s.jwtService.GetAccessTokenTTL().Seconds()),
	}, nil
}

// Logout revokes a user session
func (s *Service) Logout(refreshToken string) error {
	session, err := s.repo.GetSessionByRefreshToken(refreshToken)
	if err != nil {
		if err == ErrSessionNotFound {
			return nil // Already logged out
		}
		return err
	}

	return s.repo.RevokeSession(session.ID)
}

// LogoutAll revokes all user sessions
func (s *Service) LogoutAll(userID uuid.UUID) error {
	return s.repo.RevokeAllUserSessions(userID)
}

// GetUserProfile retrieves a user profile
func (s *Service) GetUserProfile(userID uuid.UUID) (*UserProfile, error) {
	return s.repo.GetUserProfile(userID)
}

// UpdateUserProfile updates a user profile
func (s *Service) UpdateUserProfile(userID uuid.UUID, req *UpdateProfileRequest) (*UserProfile, error) {
	// Get existing profile
	profile, err := s.repo.GetUserProfile(userID)
	if err != nil {
		return nil, err
	}

	// Update fields if provided
	if req.SkillLevels != nil {
		profile.SkillLevels = req.SkillLevels
	}
	if req.LearningGoals != nil {
		profile.LearningGoals = req.LearningGoals
	}
	if req.Preferences != nil {
		profile.Preferences = req.Preferences
	}
	profile.UpdatedAt = time.Now()

	// Update username if provided
	if req.Username != nil {
		user, err := s.repo.GetUserByID(userID)
		if err != nil {
			return nil, err
		}
		user.Username = *req.Username
		user.UpdatedAt = time.Now()
		if err := s.repo.UpdateUser(user); err != nil {
			return nil, err
		}
	}

	// Save profile
	if err := s.repo.UpdateUserProfile(profile); err != nil {
		return nil, err
	}

	return profile, nil
}

// ChangePassword changes a user's password
func (s *Service) ChangePassword(userID uuid.UUID, req *ChangePasswordRequest) error {
	// Get user
	user, err := s.repo.GetUserByID(userID)
	if err != nil {
		return err
	}

	// Verify current password
	if err := s.verifyPassword(req.CurrentPassword, user.PasswordHash); err != nil {
		return ErrInvalidCredentials
	}

	// Validate new password
	if err := s.validatePassword(req.NewPassword); err != nil {
		return err
	}

	// Hash new password
	newPasswordHash, err := s.hashPassword(req.NewPassword)
	if err != nil {
		return err
	}

	// Update password
	if err := s.repo.UpdateUserPassword(userID, newPasswordHash); err != nil {
		return err
	}

	// Revoke all sessions to force re-login
	return s.repo.RevokeAllUserSessions(userID)
}

// ValidateAccessToken validates an access token and returns user claims
func (s *Service) ValidateAccessToken(token string) (*JWTClaims, error) {
	return s.jwtService.ValidateAccessToken(token)
}

// GetUserByID retrieves a user by ID
func (s *Service) GetUserByID(userID uuid.UUID) (*User, error) {
	user, err := s.repo.GetUserByID(userID)
	if err != nil {
		return nil, err
	}
	return user.ToPublic(), nil
}

// hashPassword hashes a password using bcrypt
func (s *Service) hashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(bytes), err
}

// verifyPassword verifies a password against its hash
func (s *Service) verifyPassword(password, hash string) error {
	return bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
}

// validatePassword validates password strength
func (s *Service) validatePassword(password string) error {
	if len(password) < 8 {
		return ErrWeakPassword
	}
	
	// Add more password validation rules as needed
	// - Must contain uppercase letter
	// - Must contain lowercase letter
	// - Must contain number
	// - Must contain special character
	
	return nil
}

// CleanupExpiredSessions removes expired sessions
func (s *Service) CleanupExpiredSessions() error {
	return s.repo.CleanupExpiredSessions()
}