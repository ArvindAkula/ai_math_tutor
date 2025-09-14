package auth

import (
	"database/sql"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	_ "github.com/lib/pq"
)

// Mock database for testing
func setupTestDB(t *testing.T) *sql.DB {
	// In a real test environment, you would set up a test database
	// For this example, we'll use a mock or in-memory database
	// This is a placeholder - in production you'd use testcontainers or similar
	return nil
}

func TestUserRole_IsValid(t *testing.T) {
	tests := []struct {
		role     UserRole
		expected bool
	}{
		{RoleStudent, true},
		{RoleEducator, true},
		{RoleAdmin, true},
		{"invalid", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(string(tt.role), func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.role.IsValid())
		})
	}
}

func TestUserRole_HasPermission(t *testing.T) {
	tests := []struct {
		userRole     UserRole
		requiredRole UserRole
		expected     bool
	}{
		{RoleAdmin, RoleStudent, true},
		{RoleAdmin, RoleEducator, true},
		{RoleAdmin, RoleAdmin, true},
		{RoleEducator, RoleStudent, true},
		{RoleEducator, RoleEducator, true},
		{RoleEducator, RoleAdmin, false},
		{RoleStudent, RoleStudent, true},
		{RoleStudent, RoleEducator, false},
		{RoleStudent, RoleAdmin, false},
	}

	for _, tt := range tests {
		t.Run(string(tt.userRole)+"_"+string(tt.requiredRole), func(t *testing.T) {
			assert.Equal(t, tt.expected, tt.userRole.HasPermission(tt.requiredRole))
		})
	}
}

func TestUser_ToPublic(t *testing.T) {
	user := &User{
		ID:           uuid.New(),
		Email:        "test@example.com",
		Username:     "testuser",
		PasswordHash: "hashed_password",
		Role:         RoleStudent,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		IsActive:     true,
	}

	publicUser := user.ToPublic()

	assert.Equal(t, user.ID, publicUser.ID)
	assert.Equal(t, user.Email, publicUser.Email)
	assert.Equal(t, user.Username, publicUser.Username)
	assert.Equal(t, user.Role, publicUser.Role)
	assert.Equal(t, user.CreatedAt, publicUser.CreatedAt)
	assert.Equal(t, user.UpdatedAt, publicUser.UpdatedAt)
	assert.Equal(t, user.IsActive, publicUser.IsActive)
	assert.Empty(t, publicUser.PasswordHash) // Should be empty in public view
}

func TestJWTService_GenerateTokenPair(t *testing.T) {
	jwtService := NewJWTService()
	
	user := &User{
		ID:       uuid.New(),
		Email:    "test@example.com",
		Username: "testuser",
		Role:     RoleStudent,
	}

	accessToken, refreshToken, err := jwtService.GenerateTokenPair(user)
	
	require.NoError(t, err)
	assert.NotEmpty(t, accessToken)
	assert.NotEmpty(t, refreshToken)
	assert.NotEqual(t, accessToken, refreshToken)
}

func TestJWTService_ValidateToken(t *testing.T) {
	jwtService := NewJWTService()
	
	user := &User{
		ID:       uuid.New(),
		Email:    "test@example.com",
		Username: "testuser",
		Role:     RoleStudent,
	}

	accessToken, refreshToken, err := jwtService.GenerateTokenPair(user)
	require.NoError(t, err)

	// Test access token validation
	accessClaims, err := jwtService.ValidateAccessToken(accessToken)
	require.NoError(t, err)
	assert.Equal(t, user.ID, accessClaims.UserID)
	assert.Equal(t, user.Email, accessClaims.Email)
	assert.Equal(t, user.Username, accessClaims.Username)
	assert.Equal(t, user.Role, accessClaims.Role)
	assert.Equal(t, "access", accessClaims.Type)

	// Test refresh token validation
	refreshClaims, err := jwtService.ValidateRefreshToken(refreshToken)
	require.NoError(t, err)
	assert.Equal(t, user.ID, refreshClaims.UserID)
	assert.Equal(t, "refresh", refreshClaims.Type)

	// Test invalid token
	_, err = jwtService.ValidateAccessToken("invalid_token")
	assert.Error(t, err)
	assert.Equal(t, ErrInvalidToken, err)

	// Test wrong token type
	_, err = jwtService.ValidateAccessToken(refreshToken)
	assert.Error(t, err)
	assert.Equal(t, ErrInvalidTokenType, err)
}

func TestJWTService_GenerateSecureToken(t *testing.T) {
	jwtService := NewJWTService()
	
	token1, err := jwtService.GenerateSecureToken(32)
	require.NoError(t, err)
	assert.Len(t, token1, 64) // 32 bytes = 64 hex characters

	token2, err := jwtService.GenerateSecureToken(32)
	require.NoError(t, err)
	assert.NotEqual(t, token1, token2) // Should be different each time
}

func TestService_Register(t *testing.T) {
	// Skip if no test database available
	db := setupTestDB(t)
	if db == nil {
		t.Skip("Test database not available")
	}
	defer db.Close()

	service := NewService(db)

	req := &RegisterRequest{
		Email:    "test@example.com",
		Username: "testuser",
		Password: "password123",
		Role:     RoleStudent,
	}

	user, err := service.Register(req)
	require.NoError(t, err)
	assert.Equal(t, req.Email, user.Email)
	assert.Equal(t, req.Username, user.Username)
	assert.Equal(t, req.Role, user.Role)
	assert.True(t, user.IsActive)
	assert.Empty(t, user.PasswordHash) // Should be empty in returned user
}

func TestService_Login(t *testing.T) {
	// Skip if no test database available
	db := setupTestDB(t)
	if db == nil {
		t.Skip("Test database not available")
	}
	defer db.Close()

	service := NewService(db)

	// First register a user
	registerReq := &RegisterRequest{
		Email:    "test@example.com",
		Username: "testuser",
		Password: "password123",
		Role:     RoleStudent,
	}

	_, err := service.Register(registerReq)
	require.NoError(t, err)

	// Now try to login
	loginReq := &LoginRequest{
		Email:    "test@example.com",
		Password: "password123",
	}

	response, err := service.Login(loginReq, "test-agent", "127.0.0.1")
	require.NoError(t, err)
	assert.NotEmpty(t, response.AccessToken)
	assert.NotEmpty(t, response.RefreshToken)
	assert.Equal(t, registerReq.Email, response.User.Email)
	assert.Greater(t, response.ExpiresIn, int64(0))

	// Test invalid credentials
	invalidReq := &LoginRequest{
		Email:    "test@example.com",
		Password: "wrongpassword",
	}

	_, err = service.Login(invalidReq, "test-agent", "127.0.0.1")
	assert.Error(t, err)
	assert.Equal(t, ErrInvalidCredentials, err)
}

func TestService_RefreshToken(t *testing.T) {
	// Skip if no test database available
	db := setupTestDB(t)
	if db == nil {
		t.Skip("Test database not available")
	}
	defer db.Close()

	service := NewService(db)

	// Register and login a user
	registerReq := &RegisterRequest{
		Email:    "test@example.com",
		Username: "testuser",
		Password: "password123",
		Role:     RoleStudent,
	}

	_, err := service.Register(registerReq)
	require.NoError(t, err)

	loginReq := &LoginRequest{
		Email:    "test@example.com",
		Password: "password123",
	}

	loginResponse, err := service.Login(loginReq, "test-agent", "127.0.0.1")
	require.NoError(t, err)

	// Test token refresh
	refreshReq := &RefreshTokenRequest{
		RefreshToken: loginResponse.RefreshToken,
	}

	refreshResponse, err := service.RefreshToken(refreshReq, "test-agent", "127.0.0.1")
	require.NoError(t, err)
	assert.NotEmpty(t, refreshResponse.AccessToken)
	assert.NotEmpty(t, refreshResponse.RefreshToken)
	assert.NotEqual(t, loginResponse.AccessToken, refreshResponse.AccessToken)
	assert.NotEqual(t, loginResponse.RefreshToken, refreshResponse.RefreshToken)

	// Test invalid refresh token
	invalidRefreshReq := &RefreshTokenRequest{
		RefreshToken: "invalid_token",
	}

	_, err = service.RefreshToken(invalidRefreshReq, "test-agent", "127.0.0.1")
	assert.Error(t, err)
}

func TestService_ChangePassword(t *testing.T) {
	// Skip if no test database available
	db := setupTestDB(t)
	if db == nil {
		t.Skip("Test database not available")
	}
	defer db.Close()

	service := NewService(db)

	// Register a user
	registerReq := &RegisterRequest{
		Email:    "test@example.com",
		Username: "testuser",
		Password: "password123",
		Role:     RoleStudent,
	}

	user, err := service.Register(registerReq)
	require.NoError(t, err)

	// Change password
	changeReq := &ChangePasswordRequest{
		CurrentPassword: "password123",
		NewPassword:     "newpassword123",
	}

	err = service.ChangePassword(user.ID, changeReq)
	require.NoError(t, err)

	// Test login with new password
	loginReq := &LoginRequest{
		Email:    "test@example.com",
		Password: "newpassword123",
	}

	_, err = service.Login(loginReq, "test-agent", "127.0.0.1")
	require.NoError(t, err)

	// Test login with old password should fail
	oldLoginReq := &LoginRequest{
		Email:    "test@example.com",
		Password: "password123",
	}

	_, err = service.Login(oldLoginReq, "test-agent", "127.0.0.1")
	assert.Error(t, err)
	assert.Equal(t, ErrInvalidCredentials, err)
}

func TestService_validatePassword(t *testing.T) {
	service := &Service{}

	tests := []struct {
		password string
		valid    bool
	}{
		{"password123", true},
		{"12345678", true},
		{"short", false},
		{"", false},
	}

	for _, tt := range tests {
		t.Run(tt.password, func(t *testing.T) {
			err := service.validatePassword(tt.password)
			if tt.valid {
				assert.NoError(t, err)
			} else {
				assert.Error(t, err)
				assert.Equal(t, ErrWeakPassword, err)
			}
		})
	}
}

func TestService_hashAndVerifyPassword(t *testing.T) {
	service := &Service{}
	password := "testpassword123"

	// Test password hashing
	hash, err := service.hashPassword(password)
	require.NoError(t, err)
	assert.NotEmpty(t, hash)
	assert.NotEqual(t, password, hash)

	// Test password verification
	err = service.verifyPassword(password, hash)
	assert.NoError(t, err)

	// Test wrong password
	err = service.verifyPassword("wrongpassword", hash)
	assert.Error(t, err)
}

// Benchmark tests
func BenchmarkJWTService_GenerateTokenPair(b *testing.B) {
	jwtService := NewJWTService()
	user := &User{
		ID:       uuid.New(),
		Email:    "test@example.com",
		Username: "testuser",
		Role:     RoleStudent,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := jwtService.GenerateTokenPair(user)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkJWTService_ValidateToken(b *testing.B) {
	jwtService := NewJWTService()
	user := &User{
		ID:       uuid.New(),
		Email:    "test@example.com",
		Username: "testuser",
		Role:     RoleStudent,
	}

	accessToken, _, err := jwtService.GenerateTokenPair(user)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := jwtService.ValidateAccessToken(accessToken)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkService_hashPassword(b *testing.B) {
	service := &Service{}
	password := "testpassword123"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := service.hashPassword(password)
		if err != nil {
			b.Fatal(err)
		}
	}
}