package auth

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"os"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
)

var (
	ErrInvalidToken     = errors.New("invalid token")
	ErrExpiredToken     = errors.New("token has expired")
	ErrInvalidTokenType = errors.New("invalid token type")
)

// JWTService handles JWT token operations
type JWTService struct {
	secretKey        []byte
	accessTokenTTL   time.Duration
	refreshTokenTTL  time.Duration
}

// NewJWTService creates a new JWT service
func NewJWTService() *JWTService {
	secretKey := os.Getenv("JWT_SECRET")
	if secretKey == "" {
		secretKey = "your-super-secret-jwt-key-change-this-in-production"
	}

	accessTTL := 15 * time.Minute
	if ttl := os.Getenv("JWT_ACCESS_TTL"); ttl != "" {
		if parsed, err := time.ParseDuration(ttl); err == nil {
			accessTTL = parsed
		}
	}

	refreshTTL := 7 * 24 * time.Hour // 7 days
	if ttl := os.Getenv("JWT_REFRESH_TTL"); ttl != "" {
		if parsed, err := time.ParseDuration(ttl); err == nil {
			refreshTTL = parsed
		}
	}

	return &JWTService{
		secretKey:       []byte(secretKey),
		accessTokenTTL:  accessTTL,
		refreshTokenTTL: refreshTTL,
	}
}

// GenerateTokenPair generates both access and refresh tokens
func (j *JWTService) GenerateTokenPair(user *User) (accessToken, refreshToken string, err error) {
	// Generate access token
	accessToken, err = j.generateToken(user, "access", j.accessTokenTTL)
	if err != nil {
		return "", "", err
	}

	// Generate refresh token
	refreshToken, err = j.generateToken(user, "refresh", j.refreshTokenTTL)
	if err != nil {
		return "", "", err
	}

	return accessToken, refreshToken, nil
}

// generateToken creates a JWT token with the specified type and TTL
func (j *JWTService) generateToken(user *User, tokenType string, ttl time.Duration) (string, error) {
	now := time.Now()
	claims := jwt.MapClaims{
		"user_id":  user.ID.String(),
		"email":    user.Email,
		"username": user.Username,
		"role":     string(user.Role),
		"type":     tokenType,
		"iat":      now.Unix(),
		"exp":      now.Add(ttl).Unix(),
		"nbf":      now.Unix(),
		"iss":      "ai-math-tutor",
		"aud":      "ai-math-tutor-users",
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString(j.secretKey)
}

// ValidateToken validates a JWT token and returns the claims
func (j *JWTService) ValidateToken(tokenString string) (*JWTClaims, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		// Validate the signing method
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, errors.New("invalid signing method")
		}
		return j.secretKey, nil
	})

	if err != nil {
		if errors.Is(err, jwt.ErrTokenExpired) {
			return nil, ErrExpiredToken
		}
		return nil, ErrInvalidToken
	}

	if !token.Valid {
		return nil, ErrInvalidToken
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return nil, ErrInvalidToken
	}

	// Extract and validate claims
	userIDStr, ok := claims["user_id"].(string)
	if !ok {
		return nil, ErrInvalidToken
	}

	userID, err := uuid.Parse(userIDStr)
	if err != nil {
		return nil, ErrInvalidToken
	}

	email, ok := claims["email"].(string)
	if !ok {
		return nil, ErrInvalidToken
	}

	username, ok := claims["username"].(string)
	if !ok {
		return nil, ErrInvalidToken
	}

	roleStr, ok := claims["role"].(string)
	if !ok {
		return nil, ErrInvalidToken
	}

	role := UserRole(roleStr)
	if !role.IsValid() {
		return nil, ErrInvalidToken
	}

	tokenType, ok := claims["type"].(string)
	if !ok {
		return nil, ErrInvalidToken
	}

	return &JWTClaims{
		UserID:   userID,
		Email:    email,
		Username: username,
		Role:     role,
		Type:     tokenType,
	}, nil
}

// ValidateAccessToken validates an access token specifically
func (j *JWTService) ValidateAccessToken(tokenString string) (*JWTClaims, error) {
	claims, err := j.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.Type != "access" {
		return nil, ErrInvalidTokenType
	}

	return claims, nil
}

// ValidateRefreshToken validates a refresh token specifically
func (j *JWTService) ValidateRefreshToken(tokenString string) (*JWTClaims, error) {
	claims, err := j.ValidateToken(tokenString)
	if err != nil {
		return nil, err
	}

	if claims.Type != "refresh" {
		return nil, ErrInvalidTokenType
	}

	return claims, nil
}

// GenerateSecureToken generates a cryptographically secure random token
func (j *JWTService) GenerateSecureToken(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return hex.EncodeToString(bytes), nil
}

// GetAccessTokenTTL returns the access token TTL
func (j *JWTService) GetAccessTokenTTL() time.Duration {
	return j.accessTokenTTL
}

// GetRefreshTokenTTL returns the refresh token TTL
func (j *JWTService) GetRefreshTokenTTL() time.Duration {
	return j.refreshTokenTTL
}