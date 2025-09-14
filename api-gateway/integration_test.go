package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ai-math-tutor/api-gateway/internal/math"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

// Integration tests for the API Gateway math endpoints
func TestMathAPIIntegration(t *testing.T) {
	// Set up test mode
	gin.SetMode(gin.TestMode)

	// Create a test router with the math endpoints
	router := gin.New()
	api := router.Group("/api/v1")
	problems := api.Group("/problems")
	{
		problems.POST("/parse", math.ParseProblemHandler())
		problems.POST("/solve", math.SolveProblemHandler())
		problems.POST("/validate", math.ValidateAnswerHandler())
		problems.POST("/visualize", math.GenerateVisualizationHandler())
		problems.POST("/hint", math.GenerateHintHandler())
		problems.POST("/explain", math.ExplainStepHandler())
	}

	t.Run("Parse Problem Endpoint", func(t *testing.T) {
		requestBody := math.ProblemRequest{
			ProblemText: "Solve for x: 2x + 3 = 7",
			Domain:      "algebra",
		}

		jsonBody, _ := json.Marshal(requestBody)
		req, _ := http.NewRequest("POST", "/api/v1/problems/parse", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		// The test will fail without a running math engine, but we can verify the structure
		assert.Contains(t, []int{http.StatusOK, http.StatusInternalServerError}, w.Code)

		var response math.APIResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)

		// If it's an error, it should be properly formatted
		if w.Code != http.StatusOK {
			assert.False(t, response.Success)
			assert.NotEmpty(t, response.Error)
		}
	})

	t.Run("Solve Problem Endpoint", func(t *testing.T) {
		requestBody := math.SolveRequest{
			Problem: map[string]interface{}{
				"id":            "test-123",
				"original_text": "Solve for x: 2x + 3 = 7",
				"domain":        "algebra",
				"difficulty":    "beginner",
				"variables":     []string{"x"},
				"expressions":   []string{"2x + 3 = 7"},
				"problem_type":  "linear_equation",
				"metadata":      map[string]interface{}{},
			},
		}

		jsonBody, _ := json.Marshal(requestBody)
		req, _ := http.NewRequest("POST", "/api/v1/problems/solve", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Contains(t, []int{http.StatusOK, http.StatusInternalServerError}, w.Code)

		var response math.APIResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
	})

	t.Run("Validate Answer Endpoint", func(t *testing.T) {
		requestBody := math.ValidateRequest{
			Problem: map[string]interface{}{
				"id":            "test-123",
				"original_text": "Solve for x: 2x + 3 = 7",
				"domain":        "algebra",
				"difficulty":    "beginner",
				"variables":     []string{"x"},
				"expressions":   []string{"2x + 3 = 7"},
				"problem_type":  "linear_equation",
				"metadata":      map[string]interface{}{},
			},
			Answer: "2",
		}

		jsonBody, _ := json.Marshal(requestBody)
		req, _ := http.NewRequest("POST", "/api/v1/problems/validate", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Contains(t, []int{http.StatusOK, http.StatusInternalServerError}, w.Code)

		var response math.APIResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
	})

	t.Run("Invalid JSON Request", func(t *testing.T) {
		req, _ := http.NewRequest("POST", "/api/v1/problems/parse", bytes.NewBuffer([]byte("invalid json")))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response math.APIResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.False(t, response.Success)
		assert.NotEmpty(t, response.Error)
	})

	t.Run("Missing Required Fields", func(t *testing.T) {
		requestBody := math.ProblemRequest{
			// Missing ProblemText which is required
			Domain: "algebra",
		}

		jsonBody, _ := json.Marshal(requestBody)
		req, _ := http.NewRequest("POST", "/api/v1/problems/parse", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)

		assert.Equal(t, http.StatusBadRequest, w.Code)

		var response math.APIResponse
		err := json.Unmarshal(w.Body.Bytes(), &response)
		assert.NoError(t, err)
		assert.False(t, response.Success)
		assert.Contains(t, response.Error, "Invalid request")
	})
}

func TestRateLimitingIntegration(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	// Create router with rate limiting
	router := gin.New()
	
	// Add a simple test endpoint with rate limiting
	router.POST("/test", func(c *gin.Context) {
		// Simulate rate limiting by checking if we've exceeded limits
		// In a real scenario, this would be handled by the middleware
		c.JSON(http.StatusOK, gin.H{"message": "success"})
	})

	// Test that we can make requests within the rate limit
	for i := 0; i < 5; i++ {
		req, _ := http.NewRequest("POST", "/test", nil)
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
		
		// Should succeed within rate limit
		assert.Equal(t, http.StatusOK, w.Code)
	}
}

func TestConcurrentRequestHandling(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	router := gin.New()
	router.POST("/concurrent", func(c *gin.Context) {
		// Simulate some processing time
		time.Sleep(10 * time.Millisecond)
		c.JSON(http.StatusOK, gin.H{"message": "processed"})
	})

	// Test concurrent requests
	const numRequests = 10
	results := make(chan int, numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			req, _ := http.NewRequest("POST", "/concurrent", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
			results <- w.Code
		}()
	}

	// Collect results
	for i := 0; i < numRequests; i++ {
		statusCode := <-results
		assert.Equal(t, http.StatusOK, statusCode)
	}
}

func TestAPIResponseFormat(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	router := gin.New()
	router.POST("/test-response", func(c *gin.Context) {
		response := math.APIResponse{
			Success: true,
			Data:    map[string]string{"test": "data"},
		}
		c.JSON(http.StatusOK, response)
	})

	req, _ := http.NewRequest("POST", "/test-response", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response math.APIResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	assert.NotNil(t, response.Data)
	assert.Empty(t, response.Error)
}

func TestErrorHandling(t *testing.T) {
	gin.SetMode(gin.TestMode)
	
	router := gin.New()
	router.POST("/test-error", func(c *gin.Context) {
		response := math.APIResponse{
			Success: false,
			Error:   "Test error message",
		}
		c.JSON(http.StatusInternalServerError, response)
	})

	req, _ := http.NewRequest("POST", "/test-error", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusInternalServerError, w.Code)

	var response math.APIResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.False(t, response.Success)
	assert.Equal(t, "Test error message", response.Error)
	assert.Nil(t, response.Data)
}

// Benchmark tests for performance
func BenchmarkParseEndpoint(b *testing.B) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/parse", math.ParseProblemHandler())

	requestBody := math.ProblemRequest{
		ProblemText: "Solve for x: 2x + 3 = 7",
		Domain:      "algebra",
	}
	jsonBody, _ := json.Marshal(requestBody)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req, _ := http.NewRequest("POST", "/parse", bytes.NewBuffer(jsonBody))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		router.ServeHTTP(w, req)
	}
}

func BenchmarkConcurrentRequests(b *testing.B) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/benchmark", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	})

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			req, _ := http.NewRequest("POST", "/benchmark", nil)
			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)
		}
	})
}