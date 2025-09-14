package websocket

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewMessage(t *testing.T) {
	tests := []struct {
		name     string
		msgType  MessageType
		data     interface{}
		wantErr  bool
	}{
		{
			name:    "valid message with string data",
			msgType: MessageTypeProblemSubmit,
			data:    "test data",
			wantErr: false,
		},
		{
			name:    "valid message with struct data",
			msgType: MessageTypeProgressUpdate,
			data: ProgressUpdateMessage{
				UserID:         uuid.New(),
				ProblemsSolved: 5,
				CurrentStreak:  3,
				SkillLevels:    map[string]int{"algebra": 2},
			},
			wantErr: false,
		},
		{
			name:    "valid message with nil data",
			msgType: MessageTypeHeartbeat,
			data:    nil,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg, err := NewMessage(tt.msgType, tt.data)
			
			if tt.wantErr {
				assert.Error(t, err)
				assert.Nil(t, msg)
			} else {
				assert.NoError(t, err)
				assert.NotNil(t, msg)
				assert.Equal(t, tt.msgType, msg.Type)
				assert.NotEqual(t, uuid.Nil, msg.ID)
				assert.WithinDuration(t, time.Now(), msg.Timestamp, time.Second)
			}
		})
	}
}

func TestMessage_ParseData(t *testing.T) {
	// Create a message with known data
	originalData := ProblemSubmitMessage{
		ProblemText: "2x + 3 = 7",
		Domain:      "algebra",
		SessionID:   nil,
	}

	msg, err := NewMessage(MessageTypeProblemSubmit, originalData)
	require.NoError(t, err)

	// Parse the data back
	var parsedData ProblemSubmitMessage
	err = msg.ParseData(&parsedData)
	
	assert.NoError(t, err)
	assert.Equal(t, originalData.ProblemText, parsedData.ProblemText)
	assert.Equal(t, originalData.Domain, parsedData.Domain)
	assert.Equal(t, originalData.SessionID, parsedData.SessionID)
}

func TestMessage_ParseDataInvalidJSON(t *testing.T) {
	msg := &Message{
		ID:        uuid.New(),
		Type:      MessageTypeProblemSubmit,
		Data:      json.RawMessage(`{"invalid": json}`),
		Timestamp: time.Now(),
	}

	var parsedData ProblemSubmitMessage
	err := msg.ParseData(&parsedData)
	
	assert.Error(t, err)
}

func TestNewClient(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub) // nil conn for testing

	assert.NotEqual(t, uuid.Nil, client.ID)
	assert.Nil(t, client.UserID)
	assert.Equal(t, "", client.Username)
	assert.Equal(t, "", client.Role)
	assert.Equal(t, hub, client.Hub)
	assert.NotNil(t, client.Send)
	assert.NotNil(t, client.Sessions)
	assert.Equal(t, 0, len(client.Sessions))
}

func TestNewSession(t *testing.T) {
	name := "Test Session"
	createdBy := uuid.New()
	
	session := NewSession(name, createdBy)

	assert.NotEqual(t, uuid.Nil, session.ID)
	assert.Equal(t, name, session.Name)
	assert.Equal(t, createdBy, session.CreatedBy)
	assert.WithinDuration(t, time.Now(), session.CreatedAt, time.Second)
	assert.NotNil(t, session.Clients)
	assert.Equal(t, 0, len(session.Clients))
	assert.Nil(t, session.CurrentProblem)
	assert.True(t, session.IsActive)
}

func TestNewHub(t *testing.T) {
	hub := NewHub()

	assert.NotNil(t, hub.Clients)
	assert.NotNil(t, hub.Sessions)
	assert.NotNil(t, hub.Register)
	assert.NotNil(t, hub.Unregister)
	assert.NotNil(t, hub.Broadcast)
	assert.Equal(t, 0, len(hub.Clients))
	assert.Equal(t, 0, len(hub.Sessions))
}

func TestClient_IsAuthenticated(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub)

	// Initially not authenticated
	assert.False(t, client.IsAuthenticated())

	// Set user ID to authenticate
	userID := uuid.New()
	client.UserID = &userID
	assert.True(t, client.IsAuthenticated())
}

func TestClient_GetUserInfo(t *testing.T) {
	hub := NewHub()
	client := NewClient(nil, hub)

	// Not authenticated - should return nil
	userInfo := client.GetUserInfo()
	assert.Nil(t, userInfo)

	// Authenticated - should return user info
	userID := uuid.New()
	client.UserID = &userID
	client.Username = "testuser"
	client.Role = "student"

	userInfo = client.GetUserInfo()
	assert.NotNil(t, userInfo)
	assert.Equal(t, userID, userInfo["user_id"])
	assert.Equal(t, "testuser", userInfo["username"])
	assert.Equal(t, "student", userInfo["role"])
}

func TestMessageTypes(t *testing.T) {
	// Test that all message types are defined
	messageTypes := []MessageType{
		MessageTypeProblemSubmit,
		MessageTypeProblemSolution,
		MessageTypeProblemProgress,
		MessageTypeHintRequest,
		MessageTypeHintResponse,
		MessageTypeProgressUpdate,
		MessageTypeStreakUpdate,
		MessageTypeAchievement,
		MessageTypeJoinSession,
		MessageTypeLeaveSession,
		MessageTypeUserJoined,
		MessageTypeUserLeft,
		MessageTypeCollabUpdate,
		MessageTypeError,
		MessageTypeHeartbeat,
		MessageTypeAuth,
		MessageTypeAuthSuccess,
		MessageTypeAuthError,
	}

	for _, msgType := range messageTypes {
		assert.NotEmpty(t, string(msgType))
	}
}

func TestProblemState(t *testing.T) {
	userID := uuid.New()
	problemState := &ProblemState{
		ProblemID:   uuid.New(),
		ProblemText: "Solve for x: 2x + 3 = 7",
		Domain:      "algebra",
		Steps: []SolutionStep{
			{
				StepNumber:  1,
				Operation:   "Subtract 3 from both sides",
				Explanation: "To isolate the variable term",
				Expression:  "2x = 4",
				Result:      "2x = 4",
			},
		},
		CurrentStep: 1,
		IsComplete:  false,
		UpdatedBy:   userID,
		UpdatedAt:   time.Now(),
	}

	assert.NotEqual(t, uuid.Nil, problemState.ProblemID)
	assert.Equal(t, "Solve for x: 2x + 3 = 7", problemState.ProblemText)
	assert.Equal(t, "algebra", problemState.Domain)
	assert.Equal(t, 1, len(problemState.Steps))
	assert.Equal(t, 1, problemState.CurrentStep)
	assert.False(t, problemState.IsComplete)
	assert.Equal(t, userID, problemState.UpdatedBy)
}

func TestErrorMessages(t *testing.T) {
	// Test that error messages are proper error types
	errors := []error{
		ErrSessionNotFound,
		ErrNotAuthenticated,
		ErrInvalidMessage,
		ErrPermissionDenied,
	}

	for _, err := range errors {
		assert.NotNil(t, err)
		assert.NotEmpty(t, err.Error())
	}
}