package main

import (
	"database/sql"
	"log"
	"net/http"
	"os"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"
	"ai-math-tutor/api-gateway/internal/math"
	"ai-math-tutor/api-gateway/internal/middleware"
	"ai-math-tutor/api-gateway/internal/websocket"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/joho/godotenv"
	_ "github.com/lib/pq"
)

// Health check response
type HealthResponse struct {
	Status   string            `json:"status"`
	Version  string            `json:"version"`
	Services map[string]string `json:"services"`
}

func main() {
	// Load environment variables from .env file
	if err := godotenv.Load("../.env"); err != nil {
		log.Printf("Warning: Could not load .env file: %v", err)
	}

	// Set Gin mode based on environment
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.DebugMode)
	}

	// Initialize database connection
	db, err := initDatabase()
	if err != nil {
		log.Fatal("Failed to connect to database:", err)
	}
	defer db.Close()

	// Initialize authentication service
	authService := auth.NewService(db)

	// Initialize WebSocket hub
	wsHub := websocket.NewHub()
	go wsHub.Run()

	// Start cleanup routine for expired sessions
	go func() {
		ticker := time.NewTicker(1 * time.Hour)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				if err := authService.CleanupExpiredSessions(); err != nil {
					log.Printf("Error cleaning up expired sessions: %v", err)
				}
			}
		}
	}()

	// Start rate limiter cleanup
	middleware.CleanupExpiredLimiters()

	// TODO: Initialize gRPC client when protobuf files are generated
	// if err := grpc.InitGRPCClient(); err != nil {
	//     log.Printf("gRPC client initialization failed, using HTTP fallback: %v", err)
	// }
	// defer grpc.CloseGRPCClient()

	// Create Gin router
	r := gin.Default()

	// Add CORS middleware
	r.Use(func(c *gin.Context) {
		c.Header("Access-Control-Allow-Origin", "*")
		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	// Health check endpoints
	r.GET("/", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"service": "AI Math Tutor - API Gateway",
			"status":  "running",
			"version": "1.0.0",
		})
	})

	r.GET("/health", func(c *gin.Context) {
		// Check database connection
		dbStatus := "connected"
		if err := db.Ping(); err != nil {
			dbStatus = "disconnected"
		}

		c.JSON(http.StatusOK, HealthResponse{
			Status:  "healthy",
			Version: "1.0.0",
			Services: map[string]string{
				"database":    dbStatus,
				"redis":       "connected", // TODO: Add Redis health check
				"math_engine": "available", // TODO: Add math engine health check
			},
		})
	})

	// API routes group
	api := r.Group("/api/v1")
	{
		// Authentication routes
		authGroup := api.Group("/auth")
		{
			authGroup.POST("/register", auth.RegisterHandler(authService))
			authGroup.POST("/login", auth.LoginHandler(authService))
			authGroup.POST("/logout", auth.LogoutHandler(authService))
			authGroup.POST("/refresh", auth.RefreshTokenHandler(authService))
			authGroup.POST("/forgot-password", auth.ForgotPasswordHandler(authService))
			authGroup.POST("/reset-password", auth.ResetPasswordHandler(authService))
		}

		// Protected user management routes
		users := api.Group("/users")
		users.Use(middleware.AuthMiddleware(authService))
		{
			users.GET("/profile", auth.GetProfileHandler(authService))
			users.PUT("/profile", auth.UpdateProfileHandler(authService))
			users.POST("/change-password", auth.ChangePasswordHandler(authService))
		}

		// Math problem routes
		problems := api.Group("/problems")
		problems.Use(middleware.OptionalAuth(authService)) // Optional auth for public access
		problems.Use(middleware.MathOperationRateLimit())  // Rate limiting for math operations
		{
			problems.POST("/parse", math.ParseProblemHandler())
			problems.POST("/solve", math.SolveProblemHandler())
			problems.POST("/validate", math.ValidateAnswerHandler())
			problems.POST("/visualize", math.GenerateVisualizationHandler())
			problems.POST("/hint", math.GenerateHintHandler())
			problems.POST("/explain", math.ExplainStepHandler())
		}

		// Quiz routes (placeholder)
		quizzes := api.Group("/quizzes")
		quizzes.Use(middleware.AuthMiddleware(authService))
		{
			quizzes.POST("/generate", func(c *gin.Context) {
				// TODO: Implement in task 5.1
				c.JSON(http.StatusNotImplemented, gin.H{"error": "Not implemented yet"})
			})
			quizzes.POST("/:id/submit", func(c *gin.Context) {
				// TODO: Implement in task 5.2
				c.JSON(http.StatusNotImplemented, gin.H{"error": "Not implemented yet"})
			})
		}

		// Progress tracking routes (placeholder)
		progress := api.Group("/progress")
		progress.Use(middleware.AuthMiddleware(authService))
		{
			progress.GET("/", func(c *gin.Context) {
				// TODO: Implement in task 6.1
				c.JSON(http.StatusNotImplemented, gin.H{"error": "Not implemented yet"})
			})
			progress.GET("/recommendations", func(c *gin.Context) {
				// TODO: Implement in task 6.2
				c.JSON(http.StatusNotImplemented, gin.H{"error": "Not implemented yet"})
			})
		}

		// Admin routes
		admin := api.Group("/admin")
		admin.Use(middleware.AuthMiddleware(authService))
		admin.Use(middleware.RequireRole(auth.RoleAdmin))
		{
			admin.GET("/users", func(c *gin.Context) {
				// TODO: Implement admin user management
				c.JSON(http.StatusNotImplemented, gin.H{"error": "Not implemented yet"})
			})
			admin.GET("/websocket/stats", func(c *gin.Context) {
				stats := wsHub.GetStats()
				c.JSON(http.StatusOK, gin.H{
					"success": true,
					"data":    stats,
				})
			})
		}
	}

	// WebSocket routes
	ws := r.Group("/ws")
	ws.Use(middleware.WebSocketCORS())
	ws.Use(middleware.WebSocketLogging())
	ws.Use(middleware.WebSocketRateLimit())
	ws.Use(middleware.WebSocketAuth(authService))
	{
		// Main WebSocket endpoint for real-time features
		ws.GET("/connect", websocket.HandleWebSocket(wsHub, authService))
		
		// WebSocket health check
		ws.GET("/health", middleware.WebSocketHealthCheck())
		
		// Session management endpoints (REST API for session creation)
		sessions := ws.Group("/sessions")
		sessions.Use(middleware.AuthMiddleware(authService))
		{
			sessions.POST("/create", func(c *gin.Context) {
				userID, exists := middleware.GetUserID(c)
				if !exists {
					c.JSON(http.StatusUnauthorized, gin.H{"error": "User ID not found"})
					return
				}

				var req struct {
					Name string `json:"name" binding:"required"`
				}
				if err := c.ShouldBindJSON(&req); err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
					return
				}

				session := wsHub.CreateSession(req.Name, userID)
				c.JSON(http.StatusCreated, gin.H{
					"success": true,
					"data": gin.H{
						"session_id": session.ID,
						"name":       session.Name,
						"created_by": session.CreatedBy,
						"created_at": session.CreatedAt,
					},
				})
			})

			sessions.GET("/:id", func(c *gin.Context) {
				sessionID := c.Param("id")
				if sessionID == "" {
					c.JSON(http.StatusBadRequest, gin.H{"error": "Session ID required"})
					return
				}

				sessionUUID, err := uuid.Parse(sessionID)
				if err != nil {
					c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid session ID"})
					return
				}

				session, exists := wsHub.GetSession(sessionUUID)
				if !exists {
					c.JSON(http.StatusNotFound, gin.H{"error": "Session not found"})
					return
				}

				// Get client info for session
				clients := make([]gin.H, 0, len(session.Clients))
				for _, client := range session.Clients {
					if client.IsAuthenticated() {
						clients = append(clients, gin.H{
							"user_id":  client.UserID,
							"username": client.Username,
							"role":     client.Role,
						})
					}
				}

				c.JSON(http.StatusOK, gin.H{
					"success": true,
					"data": gin.H{
						"session_id":      session.ID,
						"name":            session.Name,
						"created_by":      session.CreatedBy,
						"created_at":      session.CreatedAt,
						"is_active":       session.IsActive,
						"clients":         clients,
						"current_problem": session.CurrentProblem,
					},
				})
			})
		}
	}

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8000"
	}

	log.Printf("🚀 API Gateway starting on port %s", port)
	if err := r.Run(":" + port); err != nil {
		log.Fatal("Failed to start server:", err)
	}
}

// initDatabase initializes the database connection
func initDatabase() (*sql.DB, error) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://user:password@localhost/ai_math_tutor?sslmode=disable"
	}

	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, err
	}

	// Configure connection pool
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, err
	}

	log.Println("✅ Database connected successfully")
	return db, nil
}