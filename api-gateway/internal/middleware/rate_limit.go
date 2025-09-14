package middleware

import (
	"net/http"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// RateLimiter represents a token bucket rate limiter
type RateLimiter struct {
	tokens    int
	capacity  int
	refillRate int
	lastRefill time.Time
	mutex     sync.Mutex
}

// NewRateLimiter creates a new rate limiter
func NewRateLimiter(capacity, refillRate int) *RateLimiter {
	return &RateLimiter{
		tokens:     capacity,
		capacity:   capacity,
		refillRate: refillRate,
		lastRefill: time.Now(),
	}
}

// Allow checks if a request should be allowed
func (rl *RateLimiter) Allow() bool {
	rl.mutex.Lock()
	defer rl.mutex.Unlock()

	now := time.Now()
	elapsed := now.Sub(rl.lastRefill)
	
	// Refill tokens based on elapsed time
	tokensToAdd := int(elapsed.Seconds()) * rl.refillRate
	if tokensToAdd > 0 {
		rl.tokens = min(rl.capacity, rl.tokens+tokensToAdd)
		rl.lastRefill = now
	}

	// Check if we have tokens available
	if rl.tokens > 0 {
		rl.tokens--
		return true
	}

	return false
}

// Global rate limiters for different endpoints
var (
	rateLimiters = make(map[string]*RateLimiter)
	limiterMutex sync.RWMutex
)

// RateLimitMiddleware creates a rate limiting middleware
func RateLimitMiddleware(requestsPerMinute int) gin.HandlerFunc {
	return func(c *gin.Context) {
		// Use IP address as the key for rate limiting
		clientIP := c.ClientIP()
		
		limiterMutex.RLock()
		limiter, exists := rateLimiters[clientIP]
		limiterMutex.RUnlock()

		if !exists {
			limiterMutex.Lock()
			// Double-check pattern
			if limiter, exists = rateLimiters[clientIP]; !exists {
				limiter = NewRateLimiter(requestsPerMinute, requestsPerMinute/60)
				rateLimiters[clientIP] = limiter
			}
			limiterMutex.Unlock()
		}

		if !limiter.Allow() {
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error": "Rate limit exceeded. Please try again later.",
				"retry_after": 60,
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// MathOperationRateLimit applies specific rate limiting for math operations
func MathOperationRateLimit() gin.HandlerFunc {
	return RateLimitMiddleware(30) // 30 requests per minute for math operations
}

// GeneralAPIRateLimit applies general rate limiting for API endpoints
func GeneralAPIRateLimit() gin.HandlerFunc {
	return RateLimitMiddleware(100) // 100 requests per minute for general API
}

// CleanupExpiredLimiters removes old rate limiters to prevent memory leaks
func CleanupExpiredLimiters() {
	ticker := time.NewTicker(10 * time.Minute)
	go func() {
		for {
			select {
			case <-ticker.C:
				limiterMutex.Lock()
				now := time.Now()
				for ip, limiter := range rateLimiters {
					// Remove limiters that haven't been used in the last hour
					if now.Sub(limiter.lastRefill) > time.Hour {
						delete(rateLimiters, ip)
					}
				}
				limiterMutex.Unlock()
			}
		}
	}()
}

// getOrCreateLimiter gets or creates a rate limiter for a specific key
func getOrCreateLimiter(key string, requestsPerMinute int, duration time.Duration) *RateLimiter {
	limiterMutex.RLock()
	limiter, exists := rateLimiters[key]
	limiterMutex.RUnlock()

	if !exists {
		limiterMutex.Lock()
		// Double-check pattern
		if limiter, exists = rateLimiters[key]; !exists {
			limiter = NewRateLimiter(requestsPerMinute, requestsPerMinute/60)
			rateLimiters[key] = limiter
		}
		limiterMutex.Unlock()
	}

	return limiter
}

// Helper function for min (Go 1.21 compatibility)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}