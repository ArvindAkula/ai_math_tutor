package math

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/assert"
)

func TestParseProblemHandler(t *testing.T) {
	// Set up test server
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/parse", ParseProblemHandler())

	tests := []struct {
		name           string
		requestBody    ProblemRequest
		expectedStatus int
		expectError    bool
	}{
		{
			name: "Valid linear equation",
			requestBody: ProblemRequest{
				ProblemText: "Solve for x: 2x + 3 = 7",
				Domain:      "algebra",
			},
			expectedStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name: "Valid calculus problem",
			requestBody: ProblemRequest{
				ProblemText: "Find the derivative of x^2 + 3x",
				Domain:      "calculus",
			},
			expectedStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name: "Empty problem text",
			requestBody: ProblemRequest{
				ProblemText: "",
				Domain:      "algebra",
			},
			expectedStatus: http.StatusBadRequest,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create request
			jsonBody, _ := json.Marshal(tt.requestBody)
			req, _ := http.NewRequest("POST", "/parse", bytes.NewBuffer(jsonBody))
			req.Header.Set("Content-Type", "application/json")

			// Create response recorder
			w := httptest.NewRecorder()

			// Perform request
			router.ServeHTTP(w, req)

			// Check status code
			assert.Equal(t, tt.expectedStatus, w.Code)

			// Parse response
			var response APIResponse
			err := json.Unmarshal(w.Body.Bytes(), &response)
			assert.NoError(t, err)

			// Check error expectation
			if tt.expectError {
				assert.False(t, response.Success)
				assert.NotEmpty(t, response.Error)
			} else {
				// Note: This test will fail without a running math engine
				// In a real test environment, you would mock the math engine service
				t.Skip("Skipping test that requires running math engine service")
			}
		})
	}
}

func TestSolveProblemHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/solve", SolveProblemHandler())

	tests := []struct {
		name           string
		requestBody    SolveRequest
		expectedStatus int
		expectError    bool
	}{
		{
			name: "Valid problem to solve",
			requestBody: SolveRequest{
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
			},
			expectedStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name: "Invalid problem structure",
			requestBody: SolveRequest{
				Problem: map[string]interface{}{},
			},
			expectedStatus: http.StatusInternalServerError,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonBody, _ := json.Marshal(tt.requestBody)
			req, _ := http.NewRequest("POST", "/solve", bytes.NewBuffer(jsonBody))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			var response APIResponse
			err := json.Unmarshal(w.Body.Bytes(), &response)
			assert.NoError(t, err)

			if tt.expectError {
				assert.False(t, response.Success)
			} else {
				t.Skip("Skipping test that requires running math engine service")
			}
		})
	}
}

func TestValidateAnswerHandler(t *testing.T) {
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/validate", ValidateAnswerHandler())

	tests := []struct {
		name           string
		requestBody    ValidateRequest
		expectedStatus int
		expectError    bool
	}{
		{
			name: "Valid answer validation",
			requestBody: ValidateRequest{
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
			},
			expectedStatus: http.StatusOK,
			expectError:    false,
		},
		{
			name: "Missing answer",
			requestBody: ValidateRequest{
				Problem: map[string]interface{}{
					"id": "test-123",
				},
				Answer: "",
			},
			expectedStatus: http.StatusBadRequest,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			jsonBody, _ := json.Marshal(tt.requestBody)
			req, _ := http.NewRequest("POST", "/validate", bytes.NewBuffer(jsonBody))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			router.ServeHTTP(w, req)

			assert.Equal(t, tt.expectedStatus, w.Code)

			var response APIResponse
			err := json.Unmarshal(w.Body.Bytes(), &response)
			assert.NoError(t, err)

			if tt.expectError {
				assert.False(t, response.Success)
			} else {
				t.Skip("Skipping test that requires running math engine service")
			}
		})
	}
}

func TestRateLimitingIntegration(t *testing.T) {
	// This test would verify that rate limiting works correctly
	// It would require setting up the middleware and making multiple requests
	t.Skip("Integration test for rate limiting - implement when needed")
}

func TestConcurrentRequests(t *testing.T) {
	// This test would verify that the API can handle concurrent requests
	// It would spawn multiple goroutines making simultaneous requests
	t.Skip("Concurrent request test - implement when needed")
}

// Mock math engine server for testing
func setupMockMathEngine() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/parse-problem":
			response := ParsedProblem{
				ID:           "test-123",
				OriginalText: "Solve for x: 2x + 3 = 7",
				Domain:       "algebra",
				Difficulty:   "beginner",
				Variables:    []string{"x"},
				Expressions:  []string{"2x + 3 = 7"},
				ProblemType:  "linear_equation",
				Metadata:     map[string]interface{}{},
			}
			json.NewEncoder(w).Encode(response)

		case "/solve-step-by-step":
			response := StepSolution{
				ProblemID:   "test-123",
				FinalAnswer: "2",
				Steps: []SolutionStep{
					{
						StepNumber:             1,
						Operation:              "Subtract 3 from both sides",
						Explanation:            "To isolate the variable term",
						MathematicalExpression: "2x = 4",
						IntermediateResult:     "2x = 4",
					},
					{
						StepNumber:             2,
						Operation:              "Divide by 2",
						Explanation:            "To solve for x",
						MathematicalExpression: "x = 2",
						IntermediateResult:     "x = 2",
					},
				},
				SolutionMethod:  "Linear equation solving",
				ConfidenceScore: 0.95,
				ComputationTime: 0.1,
			}
			json.NewEncoder(w).Encode(response)

		case "/validate-answer":
			response := ValidationResult{
				IsCorrect:     true,
				UserAnswer:    "2",
				CorrectAnswer: "2",
				Explanation:   "Correct answer",
				PartialCredit: 1.0,
			}
			json.NewEncoder(w).Encode(response)

		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

// Example of how to use the mock server in tests
func TestWithMockMathEngine(t *testing.T) {
	// Start mock server
	mockServer := setupMockMathEngine()
	defer mockServer.Close()

	// Override the math engine URL for testing
	originalURL := mathEngineURL
	mathEngineURL = mockServer.URL
	defer func() { mathEngineURL = originalURL }()

	// Now run tests that will use the mock server
	gin.SetMode(gin.TestMode)
	router := gin.New()
	router.POST("/parse", ParseProblemHandler())

	requestBody := ProblemRequest{
		ProblemText: "Solve for x: 2x + 3 = 7",
		Domain:      "algebra",
	}

	jsonBody, _ := json.Marshal(requestBody)
	req, _ := http.NewRequest("POST", "/parse", bytes.NewBuffer(jsonBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	assert.Equal(t, http.StatusOK, w.Code)

	var response APIResponse
	err := json.Unmarshal(w.Body.Bytes(), &response)
	assert.NoError(t, err)
	assert.True(t, response.Success)
	assert.NotNil(t, response.Data)
}