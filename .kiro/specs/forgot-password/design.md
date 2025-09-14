# Design Document

## Overview

The forgot password functionality will extend the existing authentication system to provide secure password reset capabilities. The design integrates with the current Go-based API gateway and PostgreSQL database, adding new database tables, service methods, and API endpoints while maintaining security best practices.

## Architecture

### High-Level Flow
1. User requests password reset via email
2. System generates secure reset token and stores it in database
3. Email service sends reset link to user
4. User clicks link, validates token, and sets new password
5. System invalidates token and revokes existing sessions

### Components Integration
- **API Gateway**: New HTTP endpoints for reset request and password reset
- **Auth Service**: Extended with password reset methods
- **Database**: New table for reset tokens
- **Email Service**: New service for sending reset emails
- **Frontend**: New UI components for reset flow

## Components and Interfaces

### Database Schema

New table for password reset tokens:

```sql
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

CREATE INDEX idx_password_reset_tokens_hash ON password_reset_tokens(token_hash);
CREATE INDEX idx_password_reset_tokens_expires ON password_reset_tokens(expires_at);
CREATE INDEX idx_password_reset_tokens_user_id ON password_reset_tokens(user_id);
```

### Data Models

#### Go Models (auth/models.go)

```go
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
```

### Email Service

#### Interface Design

```go
// EmailService interface for sending emails
type EmailService interface {
    SendPasswordResetEmail(email, resetURL string) error
    SendPasswordResetConfirmation(email string) error
}

// SMTPEmailService implements EmailService using SMTP
type SMTPEmailService struct {
    host     string
    port     int
    username string
    password string
    from     string
}
```

#### Email Templates

**Password Reset Email Template:**
```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Password Reset - AI Math Tutor</title>
</head>
<body>
    <h2>Password Reset Request</h2>
    <p>You requested a password reset for your AI Math Tutor account.</p>
    <p>Click the link below to reset your password:</p>
    <p><a href="{{.ResetURL}}" style="background-color: #007bff; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reset Password</a></p>
    <p>This link will expire in 1 hour.</p>
    <p>If you didn't request this reset, please ignore this email.</p>
    <p>Best regards,<br>AI Math Tutor Team</p>
</body>
</html>
```

### API Endpoints

#### New Authentication Endpoints

```go
// POST /api/v1/auth/forgot-password
// Request password reset
func ForgotPasswordHandler(authService *Service, emailService EmailService) gin.HandlerFunc

// POST /api/v1/auth/reset-password
// Reset password with token
func ResetPasswordHandler(authService *Service) gin.HandlerFunc

// GET /api/v1/auth/reset-password/:token
// Validate reset token (for frontend)
func ValidateResetTokenHandler(authService *Service) gin.HandlerFunc
```

### Service Layer Methods

#### Extended Auth Service

```go
// ForgotPassword generates reset token and sends email
func (s *Service) ForgotPassword(email, ipAddress, userAgent string) error

// ValidateResetToken validates a reset token
func (s *Service) ValidateResetToken(token string) (*PasswordResetToken, error)

// ResetPassword resets password using valid token
func (s *Service) ResetPassword(token, newPassword string) error

// CleanupExpiredResetTokens removes expired tokens
func (s *Service) CleanupExpiredResetTokens() error
```

#### Repository Layer Methods

```go
// CreatePasswordResetToken stores a new reset token
func (r *Repository) CreatePasswordResetToken(token *PasswordResetToken) error

// GetPasswordResetTokenByHash retrieves token by hash
func (r *Repository) GetPasswordResetTokenByHash(tokenHash string) (*PasswordResetToken, error)

// MarkResetTokenAsUsed marks token as used
func (r *Repository) MarkResetTokenAsUsed(tokenID uuid.UUID) error

// InvalidateUserResetTokens invalidates all user's reset tokens
func (r *Repository) InvalidateUserResetTokens(userID uuid.UUID) error

// CleanupExpiredResetTokens removes expired tokens
func (r *Repository) CleanupExpiredResetTokens() error
```

## Error Handling

### Error Types

```go
var (
    ErrInvalidResetToken = errors.New("invalid or expired reset token")
    ErrResetTokenUsed    = errors.New("reset token has already been used")
    ErrEmailSendFailed   = errors.New("failed to send reset email")
    ErrUserNotFound      = errors.New("user not found")
)
```

### Error Response Format

```json
{
    "error": "Invalid or expired reset token",
    "code": "INVALID_RESET_TOKEN",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Security Considerations

1. **Token Security**: Reset tokens are cryptographically secure (32 bytes) and hashed before storage
2. **Rate Limiting**: Limit reset requests per email (max 3 per hour)
3. **Email Enumeration Prevention**: Same response for existing/non-existing emails
4. **Token Expiration**: 1-hour expiration for reset tokens
5. **Session Invalidation**: All user sessions revoked after password reset
6. **Audit Trail**: Log all reset attempts with IP and user agent

## Testing Strategy

### Unit Tests

1. **Service Layer Tests**
   - Token generation and validation
   - Password reset flow
   - Error handling scenarios
   - Security validations

2. **Repository Layer Tests**
   - Database operations
   - Token storage and retrieval
   - Cleanup operations

3. **Handler Tests**
   - HTTP request/response handling
   - Input validation
   - Error responses

### Integration Tests

1. **End-to-End Flow Tests**
   - Complete password reset process
   - Email delivery verification
   - Frontend integration

2. **Security Tests**
   - Token tampering attempts
   - Expired token handling
   - Rate limiting verification

### Frontend Components

#### New React Components

1. **ForgotPasswordForm**: Email input form
2. **ResetPasswordForm**: New password form with token validation
3. **ResetPasswordSuccess**: Confirmation page

#### Frontend Routes

```javascript
// New routes to add
/forgot-password     // Request reset form
/reset-password/:token  // Reset password form
/reset-success      // Success confirmation
```

## Configuration

### Environment Variables

```bash
# Email Service Configuration
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=noreply@aimathtutor.com
SMTP_PASSWORD=app_password
SMTP_FROM=noreply@aimathtutor.com

# Frontend URL for reset links
FRONTEND_URL=http://localhost:3000

# Reset token settings
RESET_TOKEN_EXPIRY=1h
RESET_REQUEST_RATE_LIMIT=3
```

### Email Service Selection

The design supports multiple email service implementations:
1. **SMTP Service**: For production with services like SendGrid, Mailgun
2. **Mock Service**: For development and testing
3. **Console Service**: For local development (logs to console)

## Performance Considerations

1. **Database Indexing**: Proper indexes on token_hash and expires_at
2. **Token Cleanup**: Scheduled cleanup of expired tokens
3. **Email Queue**: Asynchronous email sending to avoid blocking requests
4. **Rate Limiting**: Prevent abuse with request rate limiting

## Monitoring and Logging

### Metrics to Track
- Password reset requests per hour
- Reset completion rate
- Failed reset attempts
- Email delivery success rate

### Log Events
- Reset token generation
- Reset token validation attempts
- Password reset completions
- Failed reset attempts with reasons