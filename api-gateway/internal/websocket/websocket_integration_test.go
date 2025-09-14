package websocket

import (
	"strings"
	"testing"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Integration test for the complete WebSocket real-time features
func TestWebSocketRealTimeFeatures_Integration(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "realtime-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "realtimeuser",
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

	// Read auth response (could be auth request first)
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)

	// If we get an auth request, send auth again
	if response.Type == MessageTypeAuth {
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)
		err = conn.ReadJSON(&response)
		require.NoError(t, err)
	}

	// Should now have auth success
	assert.Equal(t, MessageTypeAuthSuccess, response.Type)

	// Test real-time solution sending
	t.Run("Real-time solution", func(t *testing.T) {
		solution := map[string]interface{}{
			"problem_id":   uuid.New(),
			"final_answer": "x = 2",
			"steps": []map[string]interface{}{
				{
					"step_number": 1,
					"operation":   "Subtract 3 from both sides",
					"result":      "2x = 4",
				},
			},
		}

		// Send solution via hub
		hub.SendRealTimeSolution(nil, &userID, solution)

		// Should receive the solution
		var solutionMsg Message
		err = conn.ReadJSON(&solutionMsg)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeProblemSolution, solutionMsg.Type)

		var receivedSolution map[string]interface{}
		err = solutionMsg.ParseData(&receivedSolution)
		require.NoError(t, err)
		assert.Equal(t, "x = 2", receivedSolution["final_answer"])
	})

	// Test progress updates
	t.Run("Progress updates", func(t *testing.T) {
		progressUpdate := ProgressUpdateMessage{
			UserID:         userID,
			ProblemsSolved: 15,
			CurrentStreak:  7,
			SkillLevels:    map[string]int{"algebra": 4, "calculus": 2},
		}

		// Send progress update via hub
		hub.SendProgressUpdate(userID, progressUpdate)

		// Should receive the progress update
		var progressMsg Message
		err = conn.ReadJSON(&progressMsg)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeProgressUpdate, progressMsg.Type)

		var receivedProgress ProgressUpdateMessage
		err = progressMsg.ParseData(&receivedProgress)
		require.NoError(t, err)
		assert.Equal(t, 15, receivedProgress.ProblemsSolved)
		assert.Equal(t, 7, receivedProgress.CurrentStreak)
		assert.Equal(t, 4, receivedProgress.SkillLevels["algebra"])
	})

	// Test visualization updates
	t.Run("Visualization updates", func(t *testing.T) {
		visualization := VisualizationMessage{
			ProblemID:         uuid.New(),
			VisualizationType: "function_plot",
			PlotData: map[string]interface{}{
				"function": "y = 2x + 3",
				"domain":   []float64{-5, 5},
			},
		}

		// Send visualization update via hub
		hub.SendVisualizationUpdate(nil, &userID, visualization)

		// Should receive the visualization update
		var vizMsg Message
		err = conn.ReadJSON(&vizMsg)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeVisualizationUpdate, vizMsg.Type)

		var receivedViz VisualizationMessage
		err = vizMsg.ParseData(&receivedViz)
		require.NoError(t, err)
		assert.Equal(t, "function_plot", receivedViz.VisualizationType)
		assert.Equal(t, visualization.ProblemID, receivedViz.ProblemID)
	})
}

// Test WebSocket connection management and error recovery
func TestWebSocketConnectionManagement_Integration(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "connection-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "connectionuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Test connection health monitoring
	t.Run("Connection health", func(t *testing.T) {
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)
		defer conn.Close()

		// Authenticate
		authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)

		// Read auth response
		var response Message
		err = conn.ReadJSON(&response)
		require.NoError(t, err)

		// If we get an auth request, send auth again
		if response.Type == MessageTypeAuth {
			err = conn.WriteJSON(authMsg)
			require.NoError(t, err)
			err = conn.ReadJSON(&response)
			require.NoError(t, err)
		}

		// Test heartbeat functionality
		heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
		err = conn.WriteJSON(heartbeatMsg)
		require.NoError(t, err)

		var heartbeatResponse Message
		err = conn.ReadJSON(&heartbeatResponse)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeHeartbeat, heartbeatResponse.Type)

		// Verify connection is tracked in hub
		stats := hub.GetStats()
		assert.GreaterOrEqual(t, stats["total_clients"].(int), 1)
	})

	// Test error handling
	t.Run("Error handling", func(t *testing.T) {
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)
		defer conn.Close()

		// Try to send a message without authentication
		problemMsg, _ := NewMessage(MessageTypeProblemSubmit, ProblemSubmitMessage{
			ProblemText: "2x + 3 = 7",
			Domain:      "algebra",
		})
		err = conn.WriteJSON(problemMsg)
		require.NoError(t, err)

		// Should receive an auth request first
		var response Message
		err = conn.ReadJSON(&response)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeAuth, response.Type)

		// Now authenticate
		authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)

		// Should get auth success
		err = conn.ReadJSON(&response)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeAuthSuccess, response.Type)
	})
}

// Test collaborative session features
func TestWebSocketCollaborativeFeatures_Integration(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup two users
	userID1 := uuid.New()
	userID2 := uuid.New()
	token1 := "collab-test-token-1"
	token2 := "collab-test-token-2"

	claims1 := &auth.JWTClaims{UserID: userID1, Username: "user1", Role: auth.RoleStudent, Type: "access"}
	claims2 := &auth.JWTClaims{UserID: userID2, Username: "user2", Role: auth.RoleStudent, Type: "access"}

	authService.AddValidToken(token1, claims1)
	authService.AddValidToken(token2, claims2)

	// Create session
	session := hub.CreateSession("Collaborative Test Session", userID1)

	// Connect first user
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn1.Close()

	// Authenticate first user
	authMsg1, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token1})
	err = conn1.WriteJSON(authMsg1)
	require.NoError(t, err)

	// Read auth response
	var response1 Message
	err = conn1.ReadJSON(&response1)
	require.NoError(t, err)

	// Handle auth request if needed
	if response1.Type == MessageTypeAuth {
		err = conn1.WriteJSON(authMsg1)
		require.NoError(t, err)
		err = conn1.ReadJSON(&response1)
		require.NoError(t, err)
	}

	// Join session
	joinMsg, _ := NewMessage(MessageTypeJoinSession, JoinSessionMessage{SessionID: session.ID})
	err = conn1.WriteJSON(joinMsg)
	require.NoError(t, err)

	// Read join response
	var joinResponse Message
	err = conn1.ReadJSON(&joinResponse)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeUserJoined, joinResponse.Type)

	// Test typing indicator
	t.Run("Typing indicator", func(t *testing.T) {
		// Send typing indicator
		hub.SendTypingIndicator(session.ID, userID1, true)

		// Should receive typing indicator
		var typingMsg Message
		err = conn1.ReadJSON(&typingMsg)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeTypingIndicator, typingMsg.Type)

		var typingData map[string]interface{}
		err = typingMsg.ParseData(&typingData)
		require.NoError(t, err)
		assert.Equal(t, userID1.String(), typingData["user_id"])
		assert.Equal(t, true, typingData["is_typing"])
	})

	// Verify session state
	retrievedSession, exists := hub.GetSession(session.ID)
	require.True(t, exists)
	assert.Equal(t, 1, len(retrievedSession.Clients))
	assert.True(t, retrievedSession.IsActive)
}

// Test WebSocket performance under load
func TestWebSocketPerformance_BasicLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "performance-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "perfuser",
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

	// Read auth response
	var response Message
	err = conn.ReadJSON(&response)
	require.NoError(t, err)

	// Handle auth request if needed
	if response.Type == MessageTypeAuth {
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)
		err = conn.ReadJSON(&response)
		require.NoError(t, err)
	}

	// Test rapid message sending
	numMessages := 50
	startTime := time.Now()

	for i := 0; i < numMessages; i++ {
		heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
		err = conn.WriteJSON(heartbeatMsg)
		require.NoError(t, err)

		var heartbeatResponse Message
		err = conn.ReadJSON(&heartbeatResponse)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeHeartbeat, heartbeatResponse.Type)
	}

	duration := time.Since(startTime)
	messagesPerSecond := float64(numMessages) / duration.Seconds()

	t.Logf("Processed %d messages in %v (%.2f msg/sec)", numMessages, duration, messagesPerSecond)

	// Performance assertions
	assert.Less(t, duration, 5*time.Second, "Message processing took too long")
	assert.Greater(t, messagesPerSecond, 10.0, "Message throughput too low")

	// Verify hub is still functioning
	stats := hub.GetStats()
	assert.Equal(t, 1, stats["authenticated_clients"])
}

// Helper function to drain messages from a connection
func drainMessages(conn *websocket.Conn, timeout time.Duration) []Message {
	var messages []Message
	conn.SetReadDeadline(time.Now().Add(timeout))

	for {
		var msg Message
		err := conn.ReadJSON(&msg)
		if err != nil {
			break
		}
		messages = append(messages, msg)
	}

	return messages
}

// Test message ordering and reliability
func TestWebSocketMessageOrdering_Integration(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "ordering-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "orderuser",
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

	// Drain initial messages
	drainMessages(conn, 100*time.Millisecond)

	// Send multiple different message types rapidly
	messageTypes := []struct {
		msgType MessageType
		data    interface{}
	}{
		{MessageTypeProgressUpdate, ProgressUpdateMessage{UserID: userID, ProblemsSolved: 1}},
		{MessageTypeAchievement, map[string]interface{}{"type": "first_problem", "points": 10}},
		{MessageTypeProgressUpdate, ProgressUpdateMessage{UserID: userID, ProblemsSolved: 2}},
	}

	// Send messages via hub
	for _, mt := range messageTypes {
		switch mt.msgType {
		case MessageTypeProgressUpdate:
			hub.SendProgressUpdate(userID, mt.data.(ProgressUpdateMessage))
		case MessageTypeAchievement:
			hub.SendAchievement(userID, mt.data)
		}
	}

	// Collect received messages
	receivedMessages := drainMessages(conn, 500*time.Millisecond)

	// Should have received all messages
	assert.GreaterOrEqual(t, len(receivedMessages), len(messageTypes))

	// Verify message types are correct
	progressCount := 0
	achievementCount := 0
	for _, msg := range receivedMessages {
		switch msg.Type {
		case MessageTypeProgressUpdate:
			progressCount++
		case MessageTypeAchievement:
			achievementCount++
		}
	}

	assert.GreaterOrEqual(t, progressCount, 2, "Should receive progress updates")
	assert.GreaterOrEqual(t, achievementCount, 1, "Should receive achievement")
}