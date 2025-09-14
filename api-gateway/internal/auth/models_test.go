package auth

import (
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestPasswordResetToken(t *testing.T) {
	userID := uuid.New()
	tokenHash := "hashed_token_value"
	ipAddress := "192.168.1.1"
	userAgent := "Mozilla/5.0"

	token := &PasswordResetToken{
		ID:        uuid.New(),
		UserID:    userID,
		TokenHash: tokenHash,
		ExpiresAt: time.Now().Add(time.Hour),
		CreatedAt: time.Now(),
		IPAddress: ipAddress,
		UserAgent: userAgent,
	}

	// Test struct fields
	if token.UserID != userID {
		t.Errorf("Expected UserID %v, got %v", userID, token.UserID)
	}

	if token.TokenHash != tokenHash {
		t.Errorf("Expected TokenHash %s, got %s", tokenHash, token.TokenHash)
	}

	if token.IPAddress != ipAddress {
		t.Errorf("Expected IPAddress %s, got %s", ipAddress, token.IPAddress)
	}

	if token.UserAgent != userAgent {
		t.Errorf("Expected UserAgent %s, got %s", userAgent, token.UserAgent)
	}

	// Test that UsedAt is nil initially
	if token.UsedAt != nil {
		t.Errorf("Expected UsedAt to be nil, got %v", token.UsedAt)
	}

	// Test marking token as used
	now := time.Now()
	token.UsedAt = &now
	if token.UsedAt == nil {
		t.Error("Expected UsedAt to be set")
	}
}

func TestForgotPasswordRequest(t *testing.T) {
	email := "test@example.com"
	req := &ForgotPasswordRequest{
		Email: email,
	}

	if req.Email != email {
		t.Errorf("Expected Email %s, got %s", email, req.Email)
	}
}

func TestResetPasswordRequest(t *testing.T) {
	token := "reset_token_123"
	password := "newpassword123"
	
	req := &ResetPasswordRequest{
		Token:       token,
		NewPassword: password,
	}

	if req.Token != token {
		t.Errorf("Expected Token %s, got %s", token, req.Token)
	}

	if req.NewPassword != password {
		t.Errorf("Expected NewPassword %s, got %s", password, req.NewPassword)
	}
}

func TestForgotPasswordResponse(t *testing.T) {
	message := "Password reset email sent"
	resp := &ForgotPasswordResponse{
		Message: message,
	}

	if resp.Message != message {
		t.Errorf("Expected Message %s, got %s", message, resp.Message)
	}
}