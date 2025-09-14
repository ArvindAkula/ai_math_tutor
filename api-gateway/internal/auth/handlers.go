package auth

import (
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

// RegisterHandler handles user registration
func RegisterHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req RegisterRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		user, err := authService.Register(&req)
		if err != nil {
			switch err {
			case ErrUserAlreadyExists:
				c.JSON(http.StatusConflict, gin.H{
					"error": "User with this email already exists",
				})
			case ErrWeakPassword:
				c.JSON(http.StatusBadRequest, gin.H{
					"error": "Password is too weak. Must be at least 8 characters long",
				})
			default:
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to register user",
				})
			}
			return
		}

		c.JSON(http.StatusCreated, gin.H{
			"message": "User registered successfully",
			"user":    user,
		})
	}
}

// LoginHandler handles user login
func LoginHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req LoginRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		userAgent := c.GetHeader("User-Agent")
		ipAddress := c.ClientIP()

		response, err := authService.Login(&req, userAgent, ipAddress)
		if err != nil {
			switch err {
			case ErrInvalidCredentials:
				c.JSON(http.StatusUnauthorized, gin.H{
					"error": "Invalid email or password",
				})
			default:
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to login",
				})
			}
			return
		}

		c.JSON(http.StatusOK, response)
	}
}

// LogoutHandler handles user logout
func LogoutHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req struct {
			RefreshToken string `json:"refresh_token" binding:"required"`
		}

		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		if err := authService.Logout(req.RefreshToken); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to logout",
			})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message": "Logged out successfully",
		})
	}
}

// RefreshTokenHandler handles token refresh
func RefreshTokenHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req RefreshTokenRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		userAgent := c.GetHeader("User-Agent")
		ipAddress := c.ClientIP()

		response, err := authService.RefreshToken(&req, userAgent, ipAddress)
		if err != nil {
			switch err {
			case ErrInvalidToken, ErrExpiredToken, ErrInvalidTokenType:
				c.JSON(http.StatusUnauthorized, gin.H{
					"error": "Invalid or expired refresh token",
				})
			case ErrSessionNotFound, ErrSessionExpired:
				c.JSON(http.StatusUnauthorized, gin.H{
					"error": "Session not found or expired",
				})
			default:
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to refresh token",
				})
			}
			return
		}

		c.JSON(http.StatusOK, response)
	}
}

// GetProfileHandler handles getting user profile
func GetProfileHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		userIDValue, exists := c.Get("user_id")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "User not authenticated",
			})
			return
		}

		userID, ok := userIDValue.(uuid.UUID)
		if !ok {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid user ID format",
			})
			return
		}

		// Get user basic info
		user, err := authService.GetUserByID(userID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Failed to get user information",
			})
			return
		}

		// Get user profile
		profile, err := authService.GetUserProfile(userID)
		if err != nil {
			if err == ErrUserNotFound {
				// Create default profile if it doesn't exist
				profile = &UserProfile{
					UserID:        userID,
					SkillLevels:   make(map[string]int),
					LearningGoals: []string{},
					Preferences:   make(map[string]interface{}),
					TotalProblems: 0,
					CurrentStreak: 0,
				}
			} else {
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to get user profile",
				})
				return
			}
		}

		c.JSON(http.StatusOK, gin.H{
			"user":    user,
			"profile": profile,
		})
	}
}

// UpdateProfileHandler handles updating user profile
func UpdateProfileHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		userIDValue, exists := c.Get("user_id")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "User not authenticated",
			})
			return
		}

		userID, ok := userIDValue.(uuid.UUID)
		if !ok {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid user ID format",
			})
			return
		}

		var req UpdateProfileRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		profile, err := authService.UpdateUserProfile(userID, &req)
		if err != nil {
			switch err {
			case ErrUserNotFound:
				c.JSON(http.StatusNotFound, gin.H{
					"error": "User not found",
				})
			default:
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to update profile",
				})
			}
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message": "Profile updated successfully",
			"profile": profile,
		})
	}
}

// ChangePasswordHandler handles password changes
func ChangePasswordHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		userIDValue, exists := c.Get("user_id")
		if !exists {
			c.JSON(http.StatusUnauthorized, gin.H{
				"error": "User not authenticated",
			})
			return
		}

		userID, ok := userIDValue.(uuid.UUID)
		if !ok {
			c.JSON(http.StatusInternalServerError, gin.H{
				"error": "Invalid user ID format",
			})
			return
		}

		var req ChangePasswordRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		err := authService.ChangePassword(userID, &req)
		if err != nil {
			switch err {
			case ErrInvalidCredentials:
				c.JSON(http.StatusUnauthorized, gin.H{
					"error": "Current password is incorrect",
				})
			case ErrWeakPassword:
				c.JSON(http.StatusBadRequest, gin.H{
					"error": "New password is too weak. Must be at least 8 characters long",
				})
			case ErrUserNotFound:
				c.JSON(http.StatusNotFound, gin.H{
					"error": "User not found",
				})
			default:
				c.JSON(http.StatusInternalServerError, gin.H{
					"error": "Failed to change password",
				})
			}
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"message": "Password changed successfully. Please login again with your new password.",
		})
	}
}

// ForgotPasswordHandler handles password reset requests (placeholder)
func ForgotPasswordHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ForgotPasswordRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		// Placeholder response - feature under construction
		c.JSON(http.StatusOK, gin.H{
			"message": "🚧 Forgot Password feature is under construction and will be available in the next phase. Please contact support for password reset assistance.",
			"status":  "under_construction",
		})
	}
}

// ResetPasswordHandler handles password reset with token (placeholder)
func ResetPasswordHandler(authService *Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ResetPasswordRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{
				"error":   "Invalid request format",
				"details": err.Error(),
			})
			return
		}

		// Placeholder response - feature under construction
		c.JSON(http.StatusOK, gin.H{
			"message": "🚧 Password Reset feature is under construction and will be available in the next phase.",
			"status":  "under_construction",
		})
	}
}