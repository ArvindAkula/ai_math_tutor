package websocket

import (
	"encoding/json"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// MessageType represents different types of WebSocket messages
type MessageType string

const (
	// Problem solving session messages
	MessageTypeProblemSubmit    MessageType = "problem_submit"
	MessageTypeProblemSolution  MessageType = "problem_solution"
	MessageTypeProblemProgress  MessageType = "problem_progress"
	MessageTypeHintRequest      MessageType = "hint_request"
	MessageTypeHintResponse     MessageType = "hint_response"
	MessageTypeVisualizationUpdate MessageType = "visualization_update"
	
	// Progress tracking messages
	MessageTypeProgressUpdate   MessageType = "progress_update"
	MessageTypeStreakUpdate     MessageType = "streak_update"
	MessageTypeAchievement      MessageType = "achievement"
	
	// Collaborative features
	MessageTypeJoinSession      MessageType = "join_session"
	MessageTypeLeaveSession     MessageType = "leave_session"
	MessageTypeUserJoined       MessageType = "user_joined"
	MessageTypeUserLeft         MessageType = "user_left"
	MessageTypeCollabUpdate     MessageType = "collab_update"
	MessageTypeTypingIndicator  MessageType = "typing_indicator"
	
	// System messages
	MessageTypeError            MessageType = "error"
	MessageTypeHeartbeat        MessageType = "heartbeat"
	MessageTypeAuth             MessageType = "auth"
	MessageTypeAuthSuccess      MessageType = "auth_success"
	MessageTypeAuthError        MessageType = "auth_error"
	MessageTypeConnectionStatus MessageType = "connection_status"
	MessageTypeReconnect        MessageType = "reconnect"
)

// Message represents a WebSocket message
type Message struct {
	ID        uuid.UUID       `json:"id"`
	Type      MessageType     `json:"type"`
	Data      json.RawMessage `json:"data"`
	Timestamp time.Time       `json:"timestamp"`
	UserID    *uuid.UUID      `json:"user_id,omitempty"`
	SessionID *uuid.UUID      `json:"session_id,omitempty"`
}

// Client represents a WebSocket client connection
type Client struct {
	ID           uuid.UUID
	UserID       *uuid.UUID
	Username     string
	Role         string
	Conn         *websocket.Conn
	Send         chan Message
	Hub          *Hub
	Sessions     map[uuid.UUID]bool // Track which sessions this client is part of
	LastPing     time.Time          // Track last ping for connection health
	ConnectedAt  time.Time          // Track connection time
	ReconnectCount int              // Track reconnection attempts
	IsReconnecting bool             // Flag for reconnection state
}

// Session represents a collaborative problem-solving session
type Session struct {
	ID          uuid.UUID
	Name        string
	CreatedBy   uuid.UUID
	CreatedAt   time.Time
	Clients     map[uuid.UUID]*Client
	CurrentProblem *ProblemState
	IsActive    bool
}

// ProblemState represents the current state of a problem being solved
type ProblemState struct {
	ProblemID   uuid.UUID       `json:"problem_id"`
	ProblemText string          `json:"problem_text"`
	Domain      string          `json:"domain"`
	Steps       []SolutionStep  `json:"steps"`
	CurrentStep int             `json:"current_step"`
	IsComplete  bool            `json:"is_complete"`
	UpdatedBy   uuid.UUID       `json:"updated_by"`
	UpdatedAt   time.Time       `json:"updated_at"`
}

// SolutionStep represents a step in the problem solution
type SolutionStep struct {
	StepNumber    int    `json:"step_number"`
	Operation     string `json:"operation"`
	Explanation   string `json:"explanation"`
	Expression    string `json:"mathematical_expression"`
	Result        string `json:"intermediate_result"`
	CompletedBy   *uuid.UUID `json:"completed_by,omitempty"`
	CompletedAt   *time.Time `json:"completed_at,omitempty"`
}

// Hub maintains active clients and sessions
type Hub struct {
	Clients    map[uuid.UUID]*Client
	Sessions   map[uuid.UUID]*Session
	Register   chan *Client
	Unregister chan *Client
	Broadcast  chan Message
}

// AuthMessage represents authentication message data
type AuthMessage struct {
	Token string `json:"token"`
}

// ProblemSubmitMessage represents problem submission data
type ProblemSubmitMessage struct {
	ProblemText string `json:"problem_text"`
	Domain      string `json:"domain"`
	SessionID   *uuid.UUID `json:"session_id,omitempty"`
}

// ProgressUpdateMessage represents progress update data
type ProgressUpdateMessage struct {
	UserID         uuid.UUID `json:"user_id"`
	ProblemsSolved int       `json:"problems_solved"`
	CurrentStreak  int       `json:"current_streak"`
	SkillLevels    map[string]int `json:"skill_levels"`
}

// JoinSessionMessage represents session join request
type JoinSessionMessage struct {
	SessionID uuid.UUID `json:"session_id"`
}

// ErrorMessage represents error message data
type ErrorMessage struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details interface{} `json:"details,omitempty"`
}

// VisualizationMessage represents visualization data
type VisualizationMessage struct {
	ProblemID      uuid.UUID   `json:"problem_id"`
	VisualizationType string   `json:"visualization_type"`
	PlotData       interface{} `json:"plot_data"`
	InteractiveElements []interface{} `json:"interactive_elements,omitempty"`
}

// TypingIndicatorMessage represents typing indicator data
type TypingIndicatorMessage struct {
	UserID    uuid.UUID `json:"user_id"`
	Username  string    `json:"username"`
	IsTyping  bool      `json:"is_typing"`
	SessionID uuid.UUID `json:"session_id"`
}

// ConnectionStatusMessage represents connection status updates
type ConnectionStatusMessage struct {
	Status    string    `json:"status"` // "connected", "disconnected", "reconnecting"
	Timestamp time.Time `json:"timestamp"`
	ClientID  uuid.UUID `json:"client_id"`
	Reason    string    `json:"reason,omitempty"`
}

// ReconnectMessage represents reconnection instructions
type ReconnectMessage struct {
	Reason       string `json:"reason"`
	RetryAfter   int    `json:"retry_after_seconds"`
	MaxRetries   int    `json:"max_retries"`
	BackoffType  string `json:"backoff_type"` // "linear", "exponential"
}

// NewMessage creates a new WebSocket message
func NewMessage(msgType MessageType, data interface{}) (*Message, error) {
	dataBytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	return &Message{
		ID:        uuid.New(),
		Type:      msgType,
		Data:      dataBytes,
		Timestamp: time.Now(),
	}, nil
}

// ParseData parses the message data into the provided struct
func (m *Message) ParseData(v interface{}) error {
	return json.Unmarshal(m.Data, v)
}

// NewClient creates a new WebSocket client
func NewClient(conn *websocket.Conn, hub *Hub) *Client {
	return &Client{
		ID:             uuid.New(),
		Conn:           conn,
		Send:           make(chan Message, 256),
		Hub:            hub,
		Sessions:       make(map[uuid.UUID]bool),
		LastPing:       time.Now(),
		ConnectedAt:    time.Now(),
		ReconnectCount: 0,
		IsReconnecting: false,
	}
}

// NewSession creates a new collaborative session
func NewSession(name string, createdBy uuid.UUID) *Session {
	return &Session{
		ID:        uuid.New(),
		Name:      name,
		CreatedBy: createdBy,
		CreatedAt: time.Now(),
		Clients:   make(map[uuid.UUID]*Client),
		IsActive:  true,
	}
}

// NewHub creates a new WebSocket hub
func NewHub() *Hub {
	return &Hub{
		Clients:    make(map[uuid.UUID]*Client),
		Sessions:   make(map[uuid.UUID]*Session),
		Register:   make(chan *Client),
		Unregister: make(chan *Client),
		Broadcast:  make(chan Message),
	}
}