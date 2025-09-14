# Implementation Plan

- [x] 1. Set up database schema and models for password reset tokens
  - Create database migration for password_reset_tokens table with proper indexes
  - Add PasswordResetToken, ForgotPasswordRequest, and ResetPasswordRequest models to auth/models.go
  - Write unit tests for model validation and structure
  - _Requirements: 3.1, 3.2, 3.5_

- [ ] 2. Implement email service infrastructure
  - [ ] 2.1 Create email service interface and SMTP implementation
    - Define EmailService interface with methods for sending reset emails
    - Implement SMTPEmailService struct with SMTP configuration
    - Create email templates for password reset and confirmation emails
    - _Requirements: 1.5, 2.1_

  - [ ] 2.2 Add email service configuration and environment variables
    - Add SMTP configuration to environment variables
    - Create email service factory for different environments (SMTP, mock, console)
    - Write unit tests for email service implementations
    - _Requirements: 1.5_

- [ ] 3. Extend repository layer with password reset token operations
  - [ ] 3.1 Add password reset token database operations
    - Implement CreatePasswordResetToken method in repository
    - Implement GetPasswordResetTokenByHash method for token retrieval
    - Implement MarkResetTokenAsUsed and InvalidateUserResetTokens methods
    - _Requirements: 3.1, 3.2, 3.4_

  - [ ] 3.2 Add token cleanup and maintenance operations
    - Implement CleanupExpiredResetTokens method in repository
    - Write unit tests for all repository methods
    - Add database error handling for token operations
    - _Requirements: 3.2_

- [ ] 4. Implement password reset business logic in auth service
  - [ ] 4.1 Add forgot password functionality
    - Implement ForgotPassword method that generates secure tokens and sends emails
    - Add token generation using crypto/rand for security
    - Implement rate limiting logic to prevent abuse (max 3 requests per hour per email)
    - _Requirements: 1.3, 1.4, 3.1, 3.4_

  - [ ] 4.2 Add password reset validation and execution
    - Implement ValidateResetToken method for token verification
    - Implement ResetPassword method that validates token and updates password
    - Add session invalidation after successful password reset
    - Write comprehensive unit tests for all service methods
    - _Requirements: 2.2, 2.3, 2.6, 3.2, 3.3_

- [ ] 5. Create HTTP handlers for password reset endpoints
  - [ ] 5.1 Implement forgot password request handler
    - Create ForgotPasswordHandler that accepts email and triggers reset process
    - Add input validation and sanitization for email addresses
    - Implement consistent response messaging to prevent email enumeration
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.1_

  - [ ] 5.2 Implement password reset execution handler
    - Create ResetPasswordHandler that processes token and new password
    - Add password strength validation using existing validation logic
    - Implement proper error handling and user-friendly error messages
    - Create ValidateResetTokenHandler for frontend token validation
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.3, 4.4_

- [ ] 6. Add password reset routes to API gateway
  - Register new authentication routes for forgot password functionality
  - Add rate limiting middleware specifically for password reset endpoints
  - Update main.go to include new handlers with proper middleware chain
  - Write integration tests for all new endpoints
  - _Requirements: 1.1, 2.1, 4.1, 4.2, 4.3, 4.4_

- [ ] 7. Implement frontend components for password reset flow
  - [ ] 7.1 Create forgot password request form component
    - Build ForgotPasswordForm component with email input and validation
    - Add form submission handling and loading states
    - Implement user feedback for successful submission
    - _Requirements: 1.1, 1.2, 4.1_

  - [ ] 7.2 Create password reset form component
    - Build ResetPasswordForm component that extracts token from URL
    - Add new password input with strength validation
    - Implement token validation on component mount
    - Add success and error state handling
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 4.3, 4.4_

- [ ] 8. Add frontend routing and navigation for password reset
  - Add new routes for /forgot-password and /reset-password/:token
  - Update Login component to include "Forgot Password" link
  - Create ResetPasswordSuccess component for completion confirmation
  - Write React component tests for all new components
  - _Requirements: 1.1, 2.1, 2.4, 4.4_

- [ ] 9. Add scheduled cleanup and monitoring
  - [ ] 9.1 Implement token cleanup background job
    - Add scheduled cleanup of expired reset tokens to main.go
    - Implement cleanup routine that runs every hour
    - Add logging for cleanup operations and metrics
    - _Requirements: 3.2_

  - [ ] 9.2 Add monitoring and logging for password reset operations
    - Add structured logging for all password reset operations
    - Implement metrics tracking for reset requests and completions
    - Add security logging for failed attempts and suspicious activity
    - Write tests for logging and monitoring functionality
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 10. Write comprehensive integration tests
  - Create end-to-end tests for complete password reset flow
  - Add security tests for token tampering and expiration scenarios
  - Implement rate limiting tests to verify abuse prevention
  - Add email delivery verification tests using mock email service
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 3.1, 3.2, 3.3, 3.4, 3.5, 4.1, 4.2, 4.3, 4.4, 4.5_