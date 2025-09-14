package middleware

import (
	"log"
	"net/http"
	"strings"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"

	"github.com/gin-gonic/gin"
)

// WebSocketRateLimit creates rate limiting middleware for WebSocket connections
func WebSocketRateLimit() gin.HandlerFunc {
	return func(c *gin.Context) {
		clientIP := c.ClientIP()
		
		// Use a separate rate limiter for WebSocket connections
		limiter := getOrCreateLimiter(clientIP+"_ws", 10, time.Minute) // 10 connections per minute
		
		if !limiter.Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "WebSocket connection rate limit exceeded",
			})
			c.Abort()
			return
		}
		
		c.Next()
	}
}

// WebSocketAuth creates optional authentication middleware for WebSocket upgrades
// This extracts token from query parameters for initial connection
func WebSocketAuth(authService *auth.Service) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Try to get token from query parameter for WebSocket connections
		token := c.Query("token")
		if token == "" {
			// Try Authorization header as fallback
			authHeader := c.GetHeader("Authorization")
			if authHeader != "" {
				tokenParts := strings.Split(authHeader, " ")
				if len(tokenParts) == 2 && tokenParts[0] == "Bearer" {
					token = tokenParts[1]
				}
			}
		}

		// If no token provided, continue without authentication
		// Authentication will be handled via WebSocket messages
		if token == "" {
			c.Next()
			return
		}

		// Validate token if provided
		claims, err := authService.ValidateAccessToken(token)
		if err != nil {
			log.Printf("WebSocket pre-auth failed: %v", err)
			// Don't abort, let WebSocket handle authentication
			c.Next()
			return
		}

		// Set user information in context for potential use
		c.Set("ws_user_id", claims.UserID)
		c.Set("ws_user_email", claims.Email)
		c.Set("ws_user_username", claims.Username)
		c.Set("ws_user_role", claims.Role)
		c.Set("ws_jwt_claims", claims)

		c.Next()
	}
}

// WebSocketCORS handles CORS for WebSocket connections
func WebSocketCORS() gin.HandlerFunc {
	return func(c *gin.Context) {
		origin := c.GetHeader("Origin")
		
		// In production, implement proper origin validation
		// For now, allow common development origins
		allowedOrigins := []string{
			"http://localhost:3000",
			"http://localhost:3001",
			"http://127.0.0.1:3000",
			"http://127.0.0.1:3001",
		}
		
		isAllowed := false
		for _, allowed := range allowedOrigins {
			if origin == allowed {
				isAllowed = true
				break
			}
		}
		
		// In development, allow all origins
		// In production, be more restrictive
		if origin == "" || isAllowed {
			c.Header("Access-Control-Allow-Origin", origin)
		}
		
		c.Header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization, Sec-WebSocket-Protocol")
		c.Header("Access-Control-Allow-Credentials", "true")
		
		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}
		
		c.Next()
	}
}

// WebSocketLogging logs WebSocket connection attempts
func WebSocketLogging() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		clientIP := c.ClientIP()
		userAgent := c.GetHeader("User-Agent")
		
		log.Printf("WebSocket connection attempt from %s (User-Agent: %s)", clientIP, userAgent)
		
		c.Next()
		
		duration := time.Since(start)
		log.Printf("WebSocket connection processed in %v", duration)
	}
}

// WebSocketHealthCheck provides a simple health check for WebSocket endpoint
func WebSocketHealthCheck() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Check if this is a WebSocket upgrade request
		if c.GetHeader("Upgrade") != "websocket" {
			c.JSON(http.StatusOK, gin.H{
				"service":   "WebSocket",
				"status":    "available",
				"timestamp": time.Now(),
				"message":   "WebSocket endpoint is ready for connections",
			})
			return
		}
		
		c.Next()
	}
}