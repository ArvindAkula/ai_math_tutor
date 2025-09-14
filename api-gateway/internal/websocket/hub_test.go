package websocket

import (
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestHub_CreateSession(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	sessionName := "Test Session"

	session := hub.CreateSession(sessionName, userID)

	assert.NotNil(t, session)
	assert.NotEqual(t, uuid.Nil, session.ID)
	assert.Equal(t, sessionName, session.Name)
	assert.Equal(t, userID, session.CreatedBy)
	assert.True(t, session.IsActive)
	assert.Equal(t, 0, len(session.Clients))

	// Check that session is stored in hub
	storedSession, exists := hub.Sessions[session.ID]
	assert.True(t, exists)
	assert.Equal(t, session, storedSession)
}

func TestHub_GetSession(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	session := hub.CreateSession("Test Session", userID)

	// Test existing session
	retrievedSession, exists := hub.GetSession(session.ID)
	assert.True(t, exists)
	assert.Equal(t, session, retrievedSession)

	// Test non-existing session
	nonExistentID := uuid.New()
	retrievedSession, exists = hub.GetSession(nonExistentID)
	assert.False(t, exists)
	assert.Nil(t, retrievedSession)
}

func TestHub_JoinSession(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	session := hub.CreateSession("Test Session", userID)

	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()

	// Create authenticated client
	client := NewClient(nil, hub)
	client.UserID = &userID
	client.Username = "testuser"

	// Test successful join
	err := hub.JoinSession(client, session.ID)
	assert.NoError(t, err)

	// Check client is in session
	assert.True(t, client.Sessions[session.ID])
	assert.Equal(t, client, session.Clients[client.ID])

	// Test joining non-existent session
	nonExistentID := uuid.New()
	err = hub.JoinSession(client, nonExistentID)
	assert.Error(t, err)

	// Test unauthenticated client
	unauthClient := NewClient(nil, hub)
	err = hub.JoinSession(unauthClient, session.ID)
	assert.Error(t, err)
}

func TestHub_LeaveSession(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	session := hub.CreateSession("Test Session", userID)

	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()

	// Create and join client
	client := NewClient(nil, hub)
	client.UserID = &userID
	client.Username = "testuser"
	
	err := hub.JoinSession(client, session.ID)
	require.NoError(t, err)

	// Verify client is in session
	assert.True(t, client.Sessions[session.ID])
	assert.Equal(t, client, session.Clients[client.ID])

	// Leave session
	hub.LeaveSession(client, session.ID)

	// Verify client is removed from session
	assert.False(t, client.Sessions[session.ID])
	assert.Nil(t, session.Clients[client.ID])
}

func TestHub_UpdateProblemState(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	session := hub.CreateSession("Test Session", userID)

	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()

	problemState := &ProblemState{
		ProblemID:   uuid.New(),
		ProblemText: "2x + 3 = 7",
		Domain:      "algebra",
		Steps:       []SolutionStep{},
		CurrentStep: 0,
		IsComplete:  false,
		UpdatedBy:   userID,
		UpdatedAt:   time.Now(),
	}

	// Test successful update
	err := hub.UpdateProblemState(session.ID, problemState)
	assert.NoError(t, err)
	assert.Equal(t, problemState, session.CurrentProblem)

	// Test update non-existent session
	nonExistentID := uuid.New()
	err = hub.UpdateProblemState(nonExistentID, problemState)
	assert.Error(t, err)
}

func TestHub_GetClientSessions(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	
	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()
	
	// Create multiple sessions
	session1 := hub.CreateSession("Session 1", userID)
	session2 := hub.CreateSession("Session 2", userID)
	session3 := hub.CreateSession("Session 3", userID)

	// Create client and join some sessions
	client := NewClient(nil, hub)
	client.UserID = &userID
	client.Username = "testuser"
	hub.Clients[client.ID] = client

	err := hub.JoinSession(client, session1.ID)
	require.NoError(t, err)
	err = hub.JoinSession(client, session2.ID)
	require.NoError(t, err)

	// Get client sessions
	clientSessions := hub.GetClientSessions(client.ID)
	assert.Equal(t, 2, len(clientSessions))

	// Verify sessions are correct
	sessionIDs := make(map[uuid.UUID]bool)
	for _, session := range clientSessions {
		sessionIDs[session.ID] = true
	}
	assert.True(t, sessionIDs[session1.ID])
	assert.True(t, sessionIDs[session2.ID])
	assert.False(t, sessionIDs[session3.ID])

	// Test non-existent client
	nonExistentID := uuid.New()
	clientSessions = hub.GetClientSessions(nonExistentID)
	assert.Nil(t, clientSessions)
}

func TestHub_SendProgressUpdate(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	update := ProgressUpdateMessage{
		UserID:         userID,
		ProblemsSolved: 10,
		CurrentStreak:  5,
		SkillLevels:    map[string]int{"algebra": 3},
	}

	// Send progress update
	hub.SendProgressUpdate(userID, update)

	// Check that message was broadcast
	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeProgressUpdate, msg.Type)
		assert.Equal(t, &userID, msg.UserID)
		
		var receivedUpdate ProgressUpdateMessage
		err := msg.ParseData(&receivedUpdate)
		assert.NoError(t, err)
		assert.Equal(t, update, receivedUpdate)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestHub_SendAchievement(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()

	// Create a channel to capture broadcast messages
	broadcastReceived := make(chan Message, 1)
	originalBroadcast := hub.Broadcast
	hub.Broadcast = broadcastReceived

	achievement := map[string]interface{}{
		"type":        "streak",
		"title":       "Problem Solver",
		"description": "Solved 10 problems in a row",
		"points":      100,
	}

	// Send achievement
	hub.SendAchievement(userID, achievement)

	// Check that message was broadcast
	select {
	case msg := <-broadcastReceived:
		assert.Equal(t, MessageTypeAchievement, msg.Type)
		assert.Equal(t, &userID, msg.UserID)
		
		var receivedAchievement map[string]interface{}
		err := msg.ParseData(&receivedAchievement)
		assert.NoError(t, err)
		assert.Equal(t, achievement, receivedAchievement)
	case <-time.After(100 * time.Millisecond):
		t.Fatal("Expected broadcast message not received")
	}

	// Restore original broadcast channel
	hub.Broadcast = originalBroadcast
}

func TestHub_GetStats(t *testing.T) {
	hub := NewHub()
	userID1 := uuid.New()
	userID2 := uuid.New()

	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()

	// Create clients
	client1 := NewClient(nil, hub)
	client1.UserID = &userID1
	client1.Username = "user1"
	hub.Clients[client1.ID] = client1

	client2 := NewClient(nil, hub)
	client2.UserID = &userID2
	client2.Username = "user2"
	hub.Clients[client2.ID] = client2

	// Create unauthenticated client
	client3 := NewClient(nil, hub)
	hub.Clients[client3.ID] = client3

	// Create sessions
	session1 := hub.CreateSession("Session 1", userID1)
	session2 := hub.CreateSession("Session 2", userID2)
	
	// Join clients to sessions
	hub.JoinSession(client1, session1.ID)
	hub.JoinSession(client2, session2.ID)

	// Get stats
	stats := hub.GetStats()

	assert.Equal(t, 3, stats["total_clients"])
	assert.Equal(t, 2, stats["authenticated_clients"])
	assert.Equal(t, 2, stats["active_sessions"])
	assert.Equal(t, 2, stats["total_sessions"])
}

func TestHub_CleanupInactiveSessions(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()

	// Create session with old timestamp
	session := NewSession("Old Session", userID)
	session.CreatedAt = time.Now().Add(-1 * time.Hour) // 1 hour ago
	hub.Sessions[session.ID] = session

	// Create recent session
	recentSession := hub.CreateSession("Recent Session", userID)

	// Run cleanup
	hub.cleanupInactiveSessions()

	// Old session should be removed (no clients and old)
	_, exists := hub.Sessions[session.ID]
	assert.False(t, exists)

	// Recent session should remain
	_, exists = hub.Sessions[recentSession.ID]
	assert.True(t, exists)
}

func TestHub_SessionCleanupWithClients(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()

	// Create old session with clients
	session := NewSession("Old Session with Clients", userID)
	session.CreatedAt = time.Now().Add(-1 * time.Hour)
	hub.Sessions[session.ID] = session

	// Add client to session
	client := NewClient(nil, hub)
	client.UserID = &userID
	session.Clients[client.ID] = client

	// Run cleanup
	hub.cleanupInactiveSessions()

	// Session should remain because it has clients
	_, exists := hub.Sessions[session.ID]
	assert.True(t, exists)
}

func TestHub_RemoveClientFromSession_EmptySession(t *testing.T) {
	hub := NewHub()
	userID := uuid.New()
	session := hub.CreateSession("Test Session", userID)

	// Start a goroutine to consume broadcast messages to prevent blocking
	go func() {
		for range hub.Broadcast {
			// Consume messages to prevent blocking
		}
	}()

	// Create and join client
	client := NewClient(nil, hub)
	client.UserID = &userID
	client.Username = "testuser"
	
	err := hub.JoinSession(client, session.ID)
	require.NoError(t, err)

	// Verify session exists
	_, exists := hub.Sessions[session.ID]
	assert.True(t, exists)

	// Remove client from session
	hub.removeClientFromSession(client, session.ID)

	// Session should be deleted because it's empty
	_, exists = hub.Sessions[session.ID]
	assert.False(t, exists)
}