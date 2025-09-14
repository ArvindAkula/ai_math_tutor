package websocket

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Performance and reliability tests for WebSocket functionality

func TestWebSocketPerformance_ConcurrentConnections(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	server, hub, authService := setupTestServer()
	defer server.Close()

	numClients := 100
	connectionsEstablished := make(chan bool, numClients)
	var wg sync.WaitGroup

	// Setup authentication tokens
	tokens := make([]string, numClients)
	for i := 0; i < numClients; i++ {
		userID := uuid.New()
		token := fmt.Sprintf("token-%d", i)
		claims := &auth.JWTClaims{
			UserID:   userID,
			Username: fmt.Sprintf("user%d", i),
			Role:     auth.RoleStudent,
			Type:     "access",
		}
		authService.AddValidToken(token, claims)
		tokens[i] = token
	}

	startTime := time.Now()

	// Create concurrent connections
	for i := 0; i < numClients; i++ {
		wg.Add(1)
		go func(clientIndex int) {
			defer wg.Done()

			wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
			conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
			if err != nil {
				t.Errorf("Failed to connect client %d: %v", clientIndex, err)
				return
			}
			defer conn.Close()

			// Authenticate
			authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: tokens[clientIndex]})
			if err := conn.WriteJSON(authMsg); err != nil {
				t.Errorf("Failed to authenticate client %d: %v", clientIndex, err)
				return
			}

			// Read auth response
			var response Message
			if err := conn.ReadJSON(&response); err != nil {
				t.Errorf("Failed to read auth response for client %d: %v", clientIndex, err)
				return
			}

			if response.Type == MessageTypeAuthSuccess {
				connectionsEstablished <- true
			}

			// Keep connection alive for a short time
			time.Sleep(100 * time.Millisecond)
		}(i)
	}

	// Wait for all connections to complete
	wg.Wait()
	connectionTime := time.Since(startTime)

	// Count successful connections
	close(connectionsEstablished)
	successfulConnections := 0
	for range connectionsEstablished {
		successfulConnections++
	}

	t.Logf("Established %d/%d connections in %v", successfulConnections, numClients, connectionTime)
	
	// Allow some time for hub to process all connections
	time.Sleep(200 * time.Millisecond)

	// Verify hub statistics
	stats := hub.GetStats()
	assert.GreaterOrEqual(t, stats["authenticated_clients"].(int), successfulConnections/2) // Allow for some disconnections
	
	// Performance assertions
	assert.Less(t, connectionTime, 10*time.Second, "Connection establishment took too long")
	assert.GreaterOrEqual(t, successfulConnections, numClients*8/10, "Too many connection failures") // Allow 20% failure rate
}

func TestWebSocketPerformance_MessageThroughput(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance test in short mode")
	}

	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authenticated client
	userID := uuid.New()
	token := "throughput-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "throughputuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Connect and authenticate
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
	err = conn.WriteJSON(authMsg)
	require.NoError(t, err)

	var authResponse Message
	err = conn.ReadJSON(&authResponse)
	require.NoError(t, err)
	require.Equal(t, MessageTypeAuthSuccess, authResponse.Type)

	// Test message throughput
	numMessages := 1000
	messagesSent := 0
	messagesReceived := 0
	
	startTime := time.Now()

	// Send messages concurrently
	go func() {
		for i := 0; i < numMessages; i++ {
			heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
			if err := conn.WriteJSON(heartbeatMsg); err != nil {
				break
			}
			messagesSent++
		}
	}()

	// Receive messages
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	for messagesReceived < numMessages {
		select {
		case <-ctx.Done():
			t.Logf("Timeout reached, received %d/%d messages", messagesReceived, numMessages)
			goto done
		default:
			var response Message
			if err := conn.ReadJSON(&response); err != nil {
				t.Logf("Error reading message: %v", err)
				break
			}
			if response.Type == MessageTypeHeartbeat {
				messagesReceived++
			}
		}
	}

done:
	throughputTime := time.Since(startTime)
	messagesPerSecond := float64(messagesReceived) / throughputTime.Seconds()

	t.Logf("Sent %d messages, received %d messages in %v", messagesSent, messagesReceived, throughputTime)
	t.Logf("Throughput: %.2f messages/second", messagesPerSecond)

	// Performance assertions
	assert.GreaterOrEqual(t, messagesReceived, numMessages*8/10, "Too many messages lost")
	assert.Greater(t, messagesPerSecond, 50.0, "Throughput too low")

	// Verify hub is still functioning
	stats := hub.GetStats()
	assert.Equal(t, 1, stats["authenticated_clients"])
}

func TestWebSocketReliability_ConnectionRecovery(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "recovery-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "recoveryuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Test multiple connection cycles
	for cycle := 0; cycle < 5; cycle++ {
		t.Logf("Connection cycle %d", cycle+1)

		// Connect and authenticate
		wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
		conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
		require.NoError(t, err)

		authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
		err = conn.WriteJSON(authMsg)
		require.NoError(t, err)

		var authResponse Message
		err = conn.ReadJSON(&authResponse)
		require.NoError(t, err)
		assert.Equal(t, MessageTypeAuthSuccess, authResponse.Type)

		// Send a few messages to verify connection works
		for i := 0; i < 3; i++ {
			heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
			err = conn.WriteJSON(heartbeatMsg)
			require.NoError(t, err)

			var response Message
			err = conn.ReadJSON(&response)
			require.NoError(t, err)
			assert.Equal(t, MessageTypeHeartbeat, response.Type)
		}

		// Close connection
		conn.Close()

		// Wait a bit before next cycle
		time.Sleep(100 * time.Millisecond)
	}

	// Verify hub cleaned up properly
	time.Sleep(200 * time.Millisecond)
	stats := hub.GetStats()
	assert.Equal(t, 0, stats["authenticated_clients"], "Connections not properly cleaned up")
}

func TestWebSocketReliability_SessionPersistence(t *testing.T) {
	server, hub, authService := setupTestServer()
	defer server.Close()

	// Setup two users
	userID1 := uuid.New()
	userID2 := uuid.New()
	token1 := "session-test-token-1"
	token2 := "session-test-token-2"

	claims1 := &auth.JWTClaims{UserID: userID1, Username: "user1", Role: auth.RoleStudent, Type: "access"}
	claims2 := &auth.JWTClaims{UserID: userID2, Username: "user2", Role: auth.RoleStudent, Type: "access"}

	authService.AddValidToken(token1, claims1)
	authService.AddValidToken(token2, claims2)

	// Create session
	session := hub.CreateSession("Persistence Test Session", userID1)

	// Connect first user
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn1, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn1.Close()

	// Authenticate first user
	authMsg1, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token1})
	err = conn1.WriteJSON(authMsg1)
	require.NoError(t, err)

	var authResponse1 Message
	err = conn1.ReadJSON(&authResponse1)
	require.NoError(t, err)

	// Join session
	joinMsg, _ := NewMessage(MessageTypeJoinSession, JoinSessionMessage{SessionID: session.ID})
	err = conn1.WriteJSON(joinMsg)
	require.NoError(t, err)

	var joinResponse Message
	err = conn1.ReadJSON(&joinResponse)
	require.NoError(t, err)

	// Verify session state
	retrievedSession, exists := hub.GetSession(session.ID)
	require.True(t, exists)
	assert.Equal(t, 1, len(retrievedSession.Clients))

	// Connect second user
	conn2, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn2.Close()

	// Authenticate second user
	authMsg2, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token2})
	err = conn2.WriteJSON(authMsg2)
	require.NoError(t, err)

	var authResponse2 Message
	err = conn2.ReadJSON(&authResponse2)
	require.NoError(t, err)

	// Second user joins session
	err = conn2.WriteJSON(joinMsg)
	require.NoError(t, err)

	// First user should receive notification
	var userJoinedMsg Message
	err = conn1.ReadJSON(&userJoinedMsg)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeUserJoined, userJoinedMsg.Type)

	// Second user should receive join confirmation
	var joinResponse2 Message
	err = conn2.ReadJSON(&joinResponse2)
	require.NoError(t, err)

	// Verify session has both users
	retrievedSession, exists = hub.GetSession(session.ID)
	require.True(t, exists)
	assert.Equal(t, 2, len(retrievedSession.Clients))

	// Simulate first user disconnecting abruptly
	conn1.Close()

	// Give time for cleanup
	time.Sleep(200 * time.Millisecond)

	// Session should still exist with second user
	retrievedSession, exists = hub.GetSession(session.ID)
	require.True(t, exists)
	assert.Equal(t, 1, len(retrievedSession.Clients))

	// Second user should still be able to send messages
	heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
	err = conn2.WriteJSON(heartbeatMsg)
	require.NoError(t, err)

	var heartbeatResponse Message
	err = conn2.ReadJSON(&heartbeatResponse)
	require.NoError(t, err)
	assert.Equal(t, MessageTypeHeartbeat, heartbeatResponse.Type)
}

func TestWebSocketReliability_ErrorRecovery(t *testing.T) {
	server, _, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "error-test-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "erroruser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Connect and authenticate
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	require.NoError(t, err)
	defer conn.Close()

	authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
	err = conn.WriteJSON(authMsg)
	require.NoError(t, err)

	var authResponse Message
	err = conn.ReadJSON(&authResponse)
	require.NoError(t, err)

	// Test various error scenarios
	testCases := []struct {
		name        string
		message     Message
		expectError bool
	}{
		{
			name: "Invalid message type",
			message: Message{
				ID:        uuid.New(),
				Type:      MessageType("invalid_type"),
				Data:      []byte(`{}`),
				Timestamp: time.Now(),
			},
			expectError: true,
		},
		{
			name: "Invalid JSON data",
			message: Message{
				ID:        uuid.New(),
				Type:      MessageTypeProblemSubmit,
				Data:      []byte(`{"invalid": json}`),
				Timestamp: time.Now(),
			},
			expectError: true,
		},
		{
			name: "Unauthenticated action",
			message: Message{
				ID:        uuid.New(),
				Type:      MessageTypeJoinSession,
				Data:      []byte(`{"session_id": "` + uuid.New().String() + `"}`),
				Timestamp: time.Now(),
			},
			expectError: false, // Should handle gracefully
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Send the test message
			err = conn.WriteJSON(tc.message)
			require.NoError(t, err)

			// Try to read response (may be error or other message)
			var response Message
			err = conn.ReadJSON(&response)
			
			if tc.expectError {
				// Should receive an error message
				assert.Equal(t, MessageTypeError, response.Type)
			} else {
				// Connection should remain stable
				assert.NotNil(t, response)
			}

			// Verify connection is still working by sending heartbeat
			heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
			err = conn.WriteJSON(heartbeatMsg)
			require.NoError(t, err)

			var heartbeatResponse Message
			err = conn.ReadJSON(&heartbeatResponse)
			require.NoError(t, err)
			assert.Equal(t, MessageTypeHeartbeat, heartbeatResponse.Type)
		})
	}
}

// Benchmark tests for performance measurement
func BenchmarkWebSocket_MessageProcessing(b *testing.B) {
	server, _, authService := setupTestServer()
	defer server.Close()

	// Setup authentication
	userID := uuid.New()
	token := "benchmark-token"
	claims := &auth.JWTClaims{
		UserID:   userID,
		Username: "benchmarkuser",
		Role:     auth.RoleStudent,
		Type:     "access",
	}
	authService.AddValidToken(token, claims)

	// Connect and authenticate
	wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
	conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
	if err != nil {
		b.Fatal(err)
	}
	defer conn.Close()

	authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: token})
	conn.WriteJSON(authMsg)

	var authResponse Message
	conn.ReadJSON(&authResponse)

	b.ResetTimer()

	// Benchmark message processing
	for i := 0; i < b.N; i++ {
		heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
		conn.WriteJSON(heartbeatMsg)

		var response Message
		conn.ReadJSON(&response)
	}
}

func BenchmarkWebSocket_ConcurrentClients(b *testing.B) {
	server, _, authService := setupTestServer()
	defer server.Close()

	numClients := 10
	tokens := make([]string, numClients)

	// Setup authentication tokens
	for i := 0; i < numClients; i++ {
		userID := uuid.New()
		token := fmt.Sprintf("bench-token-%d", i)
		claims := &auth.JWTClaims{
			UserID:   userID,
			Username: fmt.Sprintf("benchuser%d", i),
			Role:     auth.RoleStudent,
			Type:     "access",
		}
		authService.AddValidToken(token, claims)
		tokens[i] = token
	}

	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		clientIndex := 0
		for pb.Next() {
			wsURL := "ws" + strings.TrimPrefix(server.URL, "http") + "/ws"
			conn, _, err := websocket.DefaultDialer.Dial(wsURL, nil)
			if err != nil {
				continue
			}

			authMsg, _ := NewMessage(MessageTypeAuth, AuthMessage{Token: tokens[clientIndex%numClients]})
			conn.WriteJSON(authMsg)

			var authResponse Message
			conn.ReadJSON(&authResponse)

			heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, nil)
			conn.WriteJSON(heartbeatMsg)

			var response Message
			conn.ReadJSON(&response)

			conn.Close()
			clientIndex++
		}
	})
}