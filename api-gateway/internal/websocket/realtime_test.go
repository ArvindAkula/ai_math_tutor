package websocket

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHub_SendRealTimeSolution(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	sessionID := uuid.New()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	solution := map[string]interface{}{
		"problem_id": uuid.New(),
		"steps": []map[string]interface{}{
			{
				"step_number": 1,
				"operation":   "Subtract 3 from both sides",
				"result":      "2x = 4",
			},
		},
		"final_answer": "x = 2",
	}

	// Test sending to session
	hub.SendRealTimeSolution(&sessionID, nil, solution)

	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeProblemSolution, msg.Type)
		assert.Equal(t, &sessionID, msg.SessionID)
		assert.Nil(t, msg.UserID)
		
		var receivedSolution map[string]interface{}
		err := msg.ParseData(&receivedSolution)
		assert.NoError(t, err)
		assert.Equal(t, solution["final_answer"], receivedSolution["final_answer"])
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Test sending to user
	hub.SendRealTimeSolution(nil, &userID, solution)

	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeProblemSolution, msg.Type)
		assert.Nil(t, msg.SessionID)
		assert.Equal(t, &userID, msg.UserID)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestHub_SendVisualizationUpdate(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	visualization := VisualizationMessage{
		ProblemID:         uuid.New(),
		VisualizationType: "function_plot",
		PlotData: map[string]interface{}{
			"x_values": []float64{1, 2, 3, 4, 5},
			"y_values": []float64{2, 4, 6, 8, 10},
		},
		InteractiveElements: []interface{}{
			map[string]interface{}{
				"type": "hover_point",
				"x":    3,
				"y":    6,
			},
		},
	}

	// Send visualization update
	hub.SendVisualizationUpdate(nil, &userID, visualization)

	// Check that message was broadcast
	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeVisualizationUpdate, msg.Type)
		assert.Equal(t, &userID, msg.UserID)
		
		var receivedViz VisualizationMessage
		err := msg.ParseData(&receivedViz)
		assert.NoError(t, err)
		assert.Equal(t, visualization.VisualizationType, receivedViz.VisualizationType)
		assert.Equal(t, visualization.ProblemID, receivedViz.ProblemID)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestHub_SendTypingIndicator(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	sessionID := uuid.New()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	// Send typing indicator
	hub.SendTypingIndicator(sessionID, userID, true)

	// Check that message was broadcast
	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeTypingIndicator, msg.Type)
		assert.Equal(t, &sessionID, msg.SessionID)
		
		var typingData map[string]interface{}
		err := msg.ParseData(&typingData)
		assert.NoError(t, err)
		assert.Equal(t, userID.String(), typingData["user_id"])
		assert.Equal(t, true, typingData["is_typing"])
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestHub_BroadcastToAllUsers(t *testing.T) {
	hub := NewHub()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	announcement := map[string]interface{}{
		"title":   "System Maintenance",
		"message": "The system will be down for maintenance in 10 minutes",
		"type":    "warning",
	}

	// Send broadcast to all users
	hub.BroadcastToAllUsers(MessageTypeAchievement, announcement)

	// Check that message was broadcast
	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeAchievement, msg.Type)
		assert.Nil(t, msg.UserID)
		assert.Nil(t, msg.SessionID)
		
		var receivedAnnouncement map[string]interface{}
		err := msg.ParseData(&receivedAnnouncement)
		assert.NoError(t, err)
		assert.Equal(t, announcement["title"], receivedAnnouncement["title"])
		assert.Equal(t, announcement["type"], receivedAnnouncement["type"])
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestClient_ConnectionHealth(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub)

	// Initially connection should be healthy
	assert.True(t, client.isConnectionHealthy())

	// Simulate old ping time
	client.LastPing = time.Now().Add(-2 * time.Hour)
	assert.False(t, client.isConnectionHealthy())

	// Update ping time
	client.updateLastPing()
	assert.True(t, client.isConnectionHealthy())
}

func TestClient_ReconnectCount(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub)

	// Initially should be 0
	assert.Equal(t, 0, client.ReconnectCount)

	// Increment reconnect count
	client.incrementReconnectCount()
	assert.Equal(t, 1, client.ReconnectCount)

	client.incrementReconnectCount()
	assert.Equal(t, 2, client.ReconnectCount)

	// Reset reconnect count
	client.resetReconnectCount()
	assert.Equal(t, 0, client.ReconnectCount)
}

func TestClient_ConnectionDuration(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub)

	// Should have a connection duration
	duration := client.getConnectionDuration()
	assert.True(t, duration >= 0)
	assert.True(t, duration < time.Second) // Should be very recent

	// Wait a bit and check again
	time.Sleep(10 * time.Millisecond)
	newDuration := client.getConnectionDuration()
	assert.True(t, newDuration > duration)
}

func TestTypingIndicatorMessage(t *testing.T) {
	userID := uuid.New()
	sessionID := uuid.New()

	typingMsg := TypingIndicatorMessage{
		UserID:    userID,
		Username:  "testuser",
		IsTyping:  true,
		SessionID: sessionID,
	}

	// Test message creation and parsing
	msg, err := NewMessage(MessageTypeTypingIndicator, typingMsg)
	require.NoError(t, err)

	var parsedMsg TypingIndicatorMessage
	err = msg.ParseData(&parsedMsg)
	require.NoError(t, err)

	assert.Equal(t, typingMsg.UserID, parsedMsg.UserID)
	assert.Equal(t, typingMsg.Username, parsedMsg.Username)
	assert.Equal(t, typingMsg.IsTyping, parsedMsg.IsTyping)
	assert.Equal(t, typingMsg.SessionID, parsedMsg.SessionID)
}

func TestVisualizationMessage(t *testing.T) {
	problemID := uuid.New()

	vizMsg := VisualizationMessage{
		ProblemID:         problemID,
		VisualizationType: "2d_plot",
		PlotData: map[string]interface{}{
			"function": "y = 2x + 3",
			"domain":   []float64{-10, 10},
		},
		InteractiveElements: []interface{}{
			map[string]string{"type": "zoom"},
			map[string]string{"type": "pan"},
		},
	}

	// Test message creation and parsing
	msg, err := NewMessage(MessageTypeVisualizationUpdate, vizMsg)
	require.NoError(t, err)

	var parsedMsg VisualizationMessage
	err = msg.ParseData(&parsedMsg)
	require.NoError(t, err)

	assert.Equal(t, vizMsg.ProblemID, parsedMsg.ProblemID)
	assert.Equal(t, vizMsg.VisualizationType, parsedMsg.VisualizationType)
	assert.Equal(t, 2, len(parsedMsg.InteractiveElements))
}

func TestConnectionStatusMessage(t *testing.T) {
	clientID := uuid.New()

	statusMsg := ConnectionStatusMessage{
		Status:    "connected",
		Timestamp: time.Now(),
		ClientID:  clientID,
		Reason:    "initial_connection",
	}

	// Test message creation and parsing
	msg, err := NewMessage(MessageTypeConnectionStatus, statusMsg)
	require.NoError(t, err)

	var parsedMsg ConnectionStatusMessage
	err = msg.ParseData(&parsedMsg)
	require.NoError(t, err)

	assert.Equal(t, statusMsg.Status, parsedMsg.Status)
	assert.Equal(t, statusMsg.ClientID, parsedMsg.ClientID)
	assert.Equal(t, statusMsg.Reason, parsedMsg.Reason)
	assert.WithinDuration(t, statusMsg.Timestamp, parsedMsg.Timestamp, time.Second)
}

func TestReconnectMessage(t *testing.T) {
	reconnectMsg := ReconnectMessage{
		Reason:      "connection_lost",
		RetryAfter:  5,
		MaxRetries:  3,
		BackoffType: "exponential",
	}

	// Test message creation and parsing
	msg, err := NewMessage(MessageTypeReconnect, reconnectMsg)
	require.NoError(t, err)

	var parsedMsg ReconnectMessage
	err = msg.ParseData(&parsedMsg)
	require.NoError(t, err)

	assert.Equal(t, reconnectMsg.Reason, parsedMsg.Reason)
	assert.Equal(t, reconnectMsg.RetryAfter, parsedMsg.RetryAfter)
	assert.Equal(t, reconnectMsg.MaxRetries, parsedMsg.MaxRetries)
	assert.Equal(t, reconnectMsg.BackoffType, parsedMsg.BackoffType)
}

func TestNewMessageTypes(t *testing.T) {
	// Test that all new message types are properly defined
	newMessageTypes := []MessageType{
		MessageTypeVisualizationUpdate,
		MessageTypeTypingIndicator,
		MessageTypeConnectionStatus,
		MessageTypeReconnect,
	}

	for _, msgType := range newMessageTypes {
		assert.NotEmpty(t, string(msgType))
		
		// Test that we can create messages with these types
		msg, err := NewMessage(msgType, map[string]string{"test": "data"})
		assert.NoError(t, err)
		assert.Equal(t, msgType, msg.Type)
	}
}