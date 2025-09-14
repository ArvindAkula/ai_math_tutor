package websocket

import (
	"strings"
	"testing"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"net/http/httptest"
)

// MockAuthService for testing
type MockAuthService struct {
	ValidTokens map[string]*auth.JWTClaims
}

func NewMockAuthService() *MockAuthService {
	return &MockAuthService{
		ValidTokens: make(map[string]*auth.JWTClaims),
	}
}

func (m *MockAuthService) ValidateAccessToken(token string) (*auth.JWTClaims, error) {
	if claims, exists := m.ValidTokens[token]; exists {
		return claims, nil
	}
	return nil, auth.ErrInvalidToken
}

func (m *MockAuthService) AddValidToken(token string, claims *auth.JWTClaims) {
	m.ValidTokens[token] = claims
}

func setupTestServer() (*httptest.Server, *Hub, *MockAuthService) {
	gin.SetMode(gin.TestMode)
	
	hub := NewHub()
	go hub.Run()
	
	authService := NewMockAuthService()
	
	r := gin.New()
	r.GET("/ws", HandleWebSocket(hub, authService))
	
	server := httptest.NewServer(r)
	return server, hub, authService
}

func TestWebSocketConnection(t *testing.T) {
	server, hub, _ := setupTestServer()
	defer server.Close()

	// Convert HTTP URL to WebSocket URL
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"

	// Connect to WebSocket
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Give some time for connection to be registered
	time.Sleep(100 * time.Millisecond)

	// Check that client is registered in hub
	stats := hub.GetStats()
	assert.Equal(t, 1, stats["total_clients"])
	assert.Equal(t, 0, stats["authenticated_clients"]) // Not authenticated yet
}

func TestWebSocketAuthentication(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Create valid token and claims
	userID := uuid.New()
	token := "valid-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Email:    "test@example.com",
		Username: "testuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Connect to WebSocket
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Send authentication message
	authMsg, err := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
	require.NoError(t, err)

	err = conn.WriteJSON(authMsg)
	require.NoError(t, err)

	// Read response
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)

	assert.Equal(t, MessageTypeAuthSuccess, response.Type)

	// Parse auth success data
	var authSuccess map[string]interface{}
	err = response.ParseData(&authSuccess)
	require.NoError(t, err)

	assert.Equal(t, userID.String(), authSuccess["user_id"])
	assert.Equal(t, "testuser", authSuccess["username"])
	assert.Equal(t, string(auth.RoleStudent), authSuccess["role"])

	// Give some time for authentication to be processed
	time.Sleep(100 * time.Millisecond)

	// Check that client is now authenticated
	stats := hub.GetStats()
	assert.Equal(t, 1, stats["authenticated_clients"])
}

func TestWebSocketAuthenticationFailure(t *testing.T) {
	server, _, _ := setupTestServer()
	defer server.Close()

	// Connect to WebSocket
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Send authentication message with invalid token
	authMsg, err := NewMessage(MessageTypeAuth, AuthMessage{Token: "invalid-token"})
	require.NoError(t, err)

	err = conn.WriteJSON(authMsg)
	require.NoError(t, err)

	// Read response
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)

	assert.Equal(t, MessageTypeError, response.Type)

	// Parse error data
	var errorData ErrorMessage
	err = response.ParseData(&errorData)
	require.NoError(t, err)

	assert.Equal(t, "AUTH_FAILED", errorData.Code)
	assert.Contains(t, errorData.Message, "Authentication failed")
}

func TestWebSocketHeartbeat(t *testing.T) {
	server, _, _ := setupTestServer()
	defer server.Close()

	// Connect to WebSocket
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Send heartbeat message
	heartbeatMsg, err := NewMessage(MessageTypeHeartbeat, nil)
	require.NoError(t, err)

	err = conn.WriteJSON(heartbeatMsg)
	require.NoError(t, err)

	// Read response
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)

	assert.Equal(t, MessageTypeHeartbeat, response.Type)

	// Parse heartbeat response
	var heartbeatData map[string]interface{}
	err = response.ParseData(&heartbeatData)
	require.NoError(t, err)

	assert.Contains(t, heartbeatData, "timestamp")
	assert.Equal(t, "alive", heartbeatData["status"])
}

func TestWebSocketProblemSubmission(t *testing.T) {
	server, _, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "valid-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Email:    "test@example.com",
		Username: "testuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Connect and authenticate
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Authenticate
	authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
	err = conn.WriteJSON(authMsg)
	require.NoError(t, err)

	// Read auth success response
	var authResponse Message
	err = conn.ReadJSON(&authResponse)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeAuthSuccess, authResponse.Type)

	// Submit problem
	problemMsg, err := NewMessage(MessageTypeProblemSubmit, ProblemSubmitMessage{
		ProblemText: "2x + 3 = 7",
		Domain:      "algebra",
	})
	require.NoError(t, err)

	err = conn.WriteJSON(problemMsg)
	require.NoError(t, err)

	// Read progress response
	var progressResponse Message
	err = conn.ReadJSON(&progressResponse)
	require.NoError(t, err)

	assert.Equal(t, MessageTypeProblemProgress, progressResponse.Type)

	// Parse progress data
	var progressData map[string]interface{}
	err = progressResponse.ParseData(&progressData)
	require.NoError(t, err)

	assert.Contains(t, progressData, "problem_id")
	assert.Equal(t, "processing", progressData["status"])
}

func TestWebSocketSessionManagement(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()
	_ = hub // Use hub variable to avoid unused variable error

	// Setup authentication for two users
	userID1 := uuid.New()
	userID2 := uuid.New()
	token1 := "valid-test-token-1"
	token2 := "valid-test-token-2"
	
	claims1 := &auth.JWTClaims{UserID: userID1, Username: "user1", Role: auth.RoleStudent, Type: "access"}
	claims2 := &auth.JWTClaims{UserID: userID2, Username: "user2", Role: auth.RoleStudent, Type: "access"}
	
	authService.AddValidToken(token1, claims1)
	authService.AddValidToken(token2, claims2)

	// Connect first client
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn1.Close()

	// Authenticate first client
	authMsg1, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token1})
	err = conn1.WriteJSON(authMsg1)
	require.NoError(t, err)

	var authResponse1 Message
	err = conn1.ReadJSON(&authResponse1)
	require.NoError(t, err)

	// Create session
	session := hub.CreateSession("Test Session", userID1)

	// Join session
	joinMsg, _ := NewMessage(MessageTypeJoinSession, JoinSessionMessage{SessionID: session.ID})
	err = conn1.WriteJSON(joinMsg)
	require.NoError(t, err)

	// Read join success response
	var joinResponse Message
	err = conn1.ReadJSON(&joinResponse)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeUserJoined, joinResponse.Type)

	// Connect second client
	conn2, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn2.Close()

	// Authenticate second client
	authMsg2, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token2})
	err = conn2.WriteJSON(authMsg2)
	require.NoError(t, err)

	var authResponse2 Message
	err = conn2.ReadJSON(&authResponse2)
	require.NoError(t, err)

	// Second client joins session
	err = conn2.WriteJSON(joinMsg)
	require.NoError(t, err)

	// First client should receive user joined notification
	var userJoinedMsg Message
	err = conn1.ReadJSON(&userJoinedMsg)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeUserJoined, userJoinedMsg.Type)

	// Second client should receive join success
	var joinResponse2 Message
	err = conn2.ReadJSON(&joinResponse2)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeUserJoined, joinResponse2.Type)

	// Verify session has both clients
	retrievedSession, exists := hub.GetSession(session.ID)
	require.True(t, exists)
	assert.Equal(t, 2, len(retrievedSession.Clients))
}

func TestWebSocketConnectionLimit(t *testing.T) {
	server, hub, _ := setupTestServer()
	defer server.Close()

	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"

	// Create multiple connections
	var connections []*websocket.Conn
	for i := 0; i < 5; i++ {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)
		connections = append(connections, conn)
	}

	// Give time for all connections to be registered
	time.Sleep(200 * time.Millisecond)

	// Check that all connections are registered
	stats := hub.GetStats()
	assert.Equal(t, 5, stats["total_clients"])

	// Close all connections
	for _, conn := range connections {
		conn.Close()
	}

	// Give time for cleanup
	time.Sleep(200 * time.Millisecond)

	// Check that connections are cleaned up
	stats = hub.GetStats()
	assert.Equal(t, 0, stats["total_clients"])
}

func TestWebSocketErrorHandling(t *testing.T) {
	server, _, _ := setupTestServer()
	defer server.Close()

	// Connect to WebSocket
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	// Send invalid JSON message
	err = conn.WriteMessage(websocket.TextMessage, []byte(`{"invalid": json}`))
	require.NoError(t, err)

	// Connection should remain open despite invalid message
	// Send a valid heartbeat to verify connection is still working
	heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
	err = conn.WriteJSON(heartbeatMsg)
	require.NoError(t, err)

	// Should receive heartbeat response
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeHeartbeat, response.Type)
}

func TestWebSocketConcurrentConnections(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup multiple valid tokens
	numClients := 10
	tokens := make([]string, numClients)
	for i := 0; i < numClients; i++ {
		userID := uuid.New()
		token := uuid.New().String()
		claims := &auth.JWTClaims{
			UserID:   userID,
			Username: "user" + string(rune(i)),
			Role:     auth.RoleStudent,
			Type:     "access",
		}
		authService.AddValidToken(token, claims)
		tokens[i] = token
	}

	// Connect multiple clients concurrently
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	var connections []*websocket.Conn

	for i := 0; i < numClients; i++ {
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)
		connections = append(connections, conn)

		// Authenticate each client
		authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: tokens[i]})
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)
	}

	// Give time for all connections to authenticate
	time.Sleep(500 * time.Millisecond)

	// Check that all clients are authenticated
	stats := hub.GetStats()
	assert.Equal(t, numClients, stats["total_clients"])
	assert.Equal(t, numClients, stats["authenticated_clients"])

	// Clean up connections
	for _, conn := range connections {
		conn.Close()
	}
}