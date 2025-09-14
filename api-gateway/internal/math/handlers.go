package math

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"time"

	"github.com/gin-gonic/gin"
)

// Math Engine service configuration
var mathEngineURL = getEnvOrDefault("MATH_ENGINE_URL", "http://localhost:8001")

// Request/Response structures for math operations
type ProblemRequest struct {
	ProblemText string `json:"problem_text" binding:"required"`
	Domain      string `json:"domain,omitempty"`
	UserID      string `json:"user_id,omitempty"`
}

type SolveRequest struct {
	Problem map[string]interface{} `json:"problem" binding:"required"`
}

type ValidateRequest struct {
	Problem map[string]interface{} `json:"problem" binding:"required"`
	Answer  string                 `json:"answer" binding:"required"`
}

type VisualizationRequest struct {
	Problem map[string]interface{} `json:"problem" binding:"required"`
	VizType string                 `json:"viz_type,omitempty"`
}

type HintRequest struct {
	Problem     map[string]interface{} `json:"problem" binding:"required"`
	CurrentStep int                    `json:"current_step,omitempty"`
}

type ExplanationRequest struct {
	Step      map[string]interface{} `json:"step" binding:"required"`
	UserLevel string                 `json:"user_level,omitempty"`
}

// Response structures
type APIResponse struct {
	Success bool        `json:"success"`
	Data    interface{} `json:"data,omitempty"`
	Error   string      `json:"error,omitempty"`
}

type ParsedProblem struct {
	ID                string                 `json:"id"`
	OriginalText      string                 `json:"original_text"`
	Domain            string                 `json:"domain"`
	Difficulty        string                 `json:"difficulty"`
	Variables         []string               `json:"variables"`
	Expressions       []string               `json:"expressions"`
	ProblemType       string                 `json:"problem_type"`
	Metadata          map[string]interface{} `json:"metadata"`
}

type StepSolution struct {
	ProblemID        string         `json:"problem_id"`
	Steps            []SolutionStep `json:"steps"`
	FinalAnswer      string         `json:"final_answer"`
	SolutionMethod   string         `json:"solution_method"`
	ConfidenceScore  float64        `json:"confidence_score"`
	ComputationTime  float64        `json:"computation_time"`
}

type SolutionStep struct {
	StepNumber             int    `json:"step_number"`
	Operation              string `json:"operation"`
	Explanation            string `json:"explanation"`
	MathematicalExpression string `json:"mathematical_expression"`
	IntermediateResult     string `json:"intermediate_result"`
	Reasoning              string `json:"reasoning,omitempty"`
}

type ValidationResult struct {
	IsCorrect     bool    `json:"is_correct"`
	UserAnswer    string  `json:"user_answer"`
	CorrectAnswer string  `json:"correct_answer"`
	Explanation   string  `json:"explanation"`
	PartialCredit float64 `json:"partial_credit"`
}

// HTTP client for math engine communication
var httpClient = &http.Client{
	Timeout: 30 * time.Second,
}

// ParseProblemHandler handles problem parsing requests
func ParseProblemHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ProblemRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/parse-problem", map[string]interface{}{
			"problem_text": req.ProblemText,
			"domain":       req.Domain,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var parsedProblem ParsedProblem
		if err := json.Unmarshal(response, &parsedProblem); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse math engine response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    parsedProblem,
		})
	}
}

// SolveProblemHandler handles step-by-step problem solving
func SolveProblemHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req SolveRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/solve-step-by-step", req.Problem)

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var solution StepSolution
		if err := json.Unmarshal(response, &solution); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse solution response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    solution,
		})
	}
}

// ValidateAnswerHandler handles answer validation
func ValidateAnswerHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ValidateRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/validate-answer", map[string]interface{}{
			"problem": req.Problem,
			"answer":  req.Answer,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var result ValidationResult
		if err := json.Unmarshal(response, &result); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse validation response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    result,
		})
	}
}

// GenerateVisualizationHandler handles visualization generation
func GenerateVisualizationHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req VisualizationRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/generate-visualization", map[string]interface{}{
			"problem":  req.Problem,
			"viz_type": req.VizType,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var vizData map[string]interface{}
		if err := json.Unmarshal(response, &vizData); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse visualization response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    vizData,
		})
	}
}

// GenerateHintHandler handles hint generation
func GenerateHintHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req HintRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/generate-hint", map[string]interface{}{
			"problem":      req.Problem,
			"current_step": req.CurrentStep,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var hintData map[string]interface{}
		if err := json.Unmarshal(response, &hintData); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse hint response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    hintData,
		})
	}
}

// ExplainStepHandler handles step explanation requests
func ExplainStepHandler() gin.HandlerFunc {
	return func(c *gin.Context) {
		var req ExplanationRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			c.JSON(http.StatusBadRequest, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Invalid request: %v", err),
			})
			return
		}

		// Forward request to math engine
		response, err := callMathEngine("POST", "/explain-step", map[string]interface{}{
			"step":       req.Step,
			"user_level": req.UserLevel,
		})

		if err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   fmt.Sprintf("Math engine error: %v", err),
			})
			return
		}

		var explanation map[string]interface{}
		if err := json.Unmarshal(response, &explanation); err != nil {
			c.JSON(http.StatusInternalServerError, APIResponse{
				Success: false,
				Error:   "Failed to parse explanation response",
			})
			return
		}

		c.JSON(http.StatusOK, APIResponse{
			Success: true,
			Data:    explanation,
		})
	}
}

// Helper function to call math engine service
func callMathEngine(method, endpoint string, data interface{}) ([]byte, error) {
	var body io.Reader
	if data != nil {
		jsonData, err := json.Marshal(data)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request data: %v", err)
		}
		body = bytes.NewBuffer(jsonData)
	}

	req, err := http.NewRequest(method, mathEngineURL+endpoint, body)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}

	if data != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to call math engine: %v", err)
	}
	defer resp.Body.Close()

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("math engine returned error %d: %s", resp.StatusCode, string(responseBody))
	}

	return responseBody, nil
}

// Helper function to get environment variable with default
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}