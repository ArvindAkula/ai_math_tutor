package websocket

import (
	"log"
	"net/http"
	"time"

	"ai-math-tutor/api-gateway/internal/auth"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// AuthServiceInterface defines the interface needed for WebSocket authentication
type AuthServiceInterface interface {
	ValidateAccessToken(token string) (*auth.JWTClaims, error)
}

const (
	// Time allowed to write a message to the peer
	writeWait = 10 * time.Second

	// Time allowed to read the next pong message from the peer
	pongWait = 60 * time.Second

	// Send pings to peer with this period. Must be less than pongWait
	pingPeriod = (pongWait * 9) / 10

	// Maximum message size allowed from peer
	maxMessageSize = 512
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		// Allow connections from any origin in development
		// In production, implement proper origin checking
		return true
	},
}

// HandleWebSocket handles WebSocket connection upgrades
func HandleWebSocket(hub *Hub, authService AuthServiceInterface) gin.HandlerFunc {
	return func(c *gin.Context) {
		conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
		if err != nil {
			log.Printf("WebSocket upgrade failed: %v", err)
			return
		}

		client := NewClient(conn, hub)
		client.Hub.Register <- client

		// Start goroutines for reading and writing
		go client.writePump()
		go client.readPump(authService)
	}
}

// readPump pumps messages from the WebSocket connection to the hub
func (c *Client) readPump(authService AuthServiceInterface) {
	defer func() {
		c.Hub.Unregister <- c
		c.Conn.Close()
	}()

	c.Conn.SetReadLimit(maxMessageSize)
	c.Conn.SetReadDeadline(time.Now().Add(pongWait))
	c.Conn.SetPongHandler(func(string) error {
		c.Conn.SetReadDeadline(time.Now().Add(pongWait))
		c.updateLastPing() // Update ping time when pong received
		return nil
	})

	for {
		var message Message
		err := c.Conn.ReadJSON(&message)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error for client %s (User: %v): %v", c.ID, c.UserID, err)
				
				// Send reconnection instruction if connection was healthy before
				if c.isConnectionHealthy() && c.ReconnectCount < 5 {
					c.incrementReconnectCount()
					retryAfter := c.ReconnectCount * 2 // Exponential backoff
					c.sendReconnectInstruction("connection_lost", retryAfter)
				}
			}
			break
		}

		// Update last activity time
		c.updateLastPing()

		// Handle the message with error recovery
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Panic in message handler for client %s: %v", c.ID, r)
					c.sendError("INTERNAL_ERROR", "Message processing failed")
				}
			}()
			c.handleMessage(message, authService)
		}()
	}
}

// writePump pumps messages from the hub to the WebSocket connection
func (c *Client) writePump() {
	ticker := time.NewTicker(pingPeriod)
	healthTicker := time.NewTicker(30 * time.Second) // Check connection health every 30 seconds
	defer func() {
		ticker.Stop()
		healthTicker.Stop()
		c.Conn.Close()
	}()

	for {
		select {
		case message, ok := <-c.Send:
			c.Conn.SetWriteDeadline(time.Now().Add(writeWait))
			if !ok {
				c.Conn.WriteMessage(websocket.CloseMessage, []byte{})
				return
			}

			// Attempt to write message with error recovery
			if err := c.Conn.WriteJSON(message); err != nil {
				log.Printf("WebSocket write error for client %s (User: %v): %v", c.ID, c.UserID, err)
				
				// If this is a critical message, attempt to send reconnect instruction
				if message.Type == MessageTypeError || message.Type == MessageTypeReconnect {
					log.Printf("Failed to send critical message to client %s", c.ID)
				}
				return
			}

		case <-ticker.C:
			c.Conn.SetWriteDeadline(time.Now().Add(writeWait))
			if err := c.Conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				log.Printf("Failed to send ping to client %s (User: %v): %v", c.ID, c.UserID, err)
				return
			}

		case <-healthTicker.C:
			// Check connection health and send status if needed
			if !c.isConnectionHealthy() {
				log.Printf("Connection unhealthy for client %s (User: %v), last ping: %v", 
					c.ID, c.UserID, c.LastPing)
				
				// Send connection status warning
				statusMsg, _ := NewMessage(MessageTypeConnectionStatus, ConnectionStatusMessage{
					Status:    "unhealthy",
					Timestamp: time.Now(),
					ClientID:  c.ID,
					Reason:    "ping_timeout",
				})
				
				c.Conn.SetWriteDeadline(time.Now().Add(writeWait))
				if err := c.Conn.WriteJSON(statusMsg); err != nil {
					log.Printf("Failed to send health status to client %s: %v", c.ID, err)
					return
				}
			}
		}
	}
}

// handleMessage processes incoming WebSocket messages
func (c *Client) handleMessage(message Message, authService AuthServiceInterface) {
	switch message.Type {
	case MessageTypeAuth:
		c.handleAuth(message, authService)
	case MessageTypeHeartbeat:
		c.handleHeartbeat()
	case MessageTypeProblemSubmit:
		c.handleProblemSubmit(message)
	case MessageTypeHintRequest:
		c.handleHintRequest(message)
	case MessageTypeJoinSession:
		c.handleJoinSession(message)
	case MessageTypeLeaveSession:
		c.handleLeaveSession(message)
	case MessageTypeCollabUpdate:
		c.handleCollabUpdate(message)
	case MessageTypeTypingIndicator:
		c.handleTypingIndicator(message)
	case MessageTypeConnectionStatus:
		c.handleConnectionStatus(message)
	default:
		c.sendError("UNKNOWN_MESSAGE_TYPE", "Unknown message type: "+string(message.Type))
	}
}

// handleAuth handles authentication messages
func (c *Client) handleAuth(message Message, authService AuthServiceInterface) {
	var authMsg AuthMessage
	if err := message.ParseData(&authMsg); err != nil {
		c.sendError("INVALID_AUTH_DATA", "Invalid authentication data")
		return
	}

	// Validate JWT token
	claims, err := authService.ValidateAccessToken(authMsg.Token)
	if err != nil {
		c.sendError("AUTH_FAILED", "Authentication failed: "+err.Error())
		return
	}

	// Set client authentication info
	c.UserID = &claims.UserID
	c.Username = claims.Username
	c.Role = string(claims.Role)

	// Send success response
	successMsg, _ := NewMessage(MessageTypeAuthSuccess, map[string]interface{}{
		"user_id":  claims.UserID,
		"username": claims.Username,
		"role":     claims.Role,
	})
	c.Send <- *successMsg

	log.Printf("Client %s authenticated as user %s (%s)", c.ID, claims.Username, claims.UserID)
}

// handleHeartbeat handles heartbeat messages
func (c *Client) handleHeartbeat() {
	heartbeatMsg, _ := NewMessage(MessageTypeHeartbeat, map[string]interface{}{
		"timestamp": time.Now(),
		"status":    "alive",
	})
	c.Send <- *heartbeatMsg
}

// handleProblemSubmit handles problem submission messages
func (c *Client) handleProblemSubmit(message Message) {
	if c.UserID == nil {
		c.sendError("NOT_AUTHENTICATED", "Authentication required")
		return
	}

	var problemMsg ProblemSubmitMessage
	if err := message.ParseData(&problemMsg); err != nil {
		c.sendError("INVALID_PROBLEM_DATA", "Invalid problem data")
		return
	}

	// Create problem state
	problemState := &ProblemState{
		ProblemID:   uuid.New(),
		ProblemText: problemMsg.ProblemText,
		Domain:      problemMsg.Domain,
		Steps:       []SolutionStep{},
		CurrentStep: 0,
		IsComplete:  false,
		UpdatedBy:   *c.UserID,
		UpdatedAt:   time.Now(),
	}

	// If this is part of a session, update the session
	if problemMsg.SessionID != nil {
		if err := c.Hub.UpdateProblemState(*problemMsg.SessionID, problemState); err != nil {
			c.sendError("SESSION_UPDATE_FAILED", "Failed to update session")
			return
		}
	}

	// Send progress message (this would typically trigger math engine processing)
	progressMsg, _ := NewMessage(MessageTypeProblemProgress, map[string]interface{}{
		"problem_id": problemState.ProblemID,
		"status":     "processing",
		"message":    "Problem submitted for solving",
	})
	
	if problemMsg.SessionID != nil {
		progressMsg.SessionID = problemMsg.SessionID
	} else {
		progressMsg.UserID = c.UserID
	}
	
	c.Hub.Broadcast <- *progressMsg
}

// handleHintRequest handles hint request messages
func (c *Client) handleHintRequest(message Message) {
	if c.UserID == nil {
		c.sendError("NOT_AUTHENTICATED", "Authentication required")
		return
	}

	// This would typically call the math engine for hint generation
	hintMsg, _ := NewMessage(MessageTypeHintResponse, map[string]interface{}{
		"hint":       "Try isolating the variable on one side of the equation",
		"step_hint":  true,
		"difficulty": "beginner",
	})
	hintMsg.UserID = c.UserID
	c.Hub.Broadcast <- *hintMsg
}

// handleJoinSession handles session join requests
func (c *Client) handleJoinSession(message Message) {
	if c.UserID == nil {
		c.sendError("NOT_AUTHENTICATED", "Authentication required")
		return
	}

	var joinMsg JoinSessionMessage
	if err := message.ParseData(&joinMsg); err != nil {
		c.sendError("INVALID_JOIN_DATA", "Invalid join session data")
		return
	}

	if err := c.Hub.JoinSession(c, joinMsg.SessionID); err != nil {
		c.sendError("JOIN_FAILED", "Failed to join session: "+err.Error())
		return
	}

	// Send success response
	successMsg, _ := NewMessage(MessageTypeUserJoined, map[string]interface{}{
		"session_id": joinMsg.SessionID,
		"user_id":    *c.UserID,
		"username":   c.Username,
		"message":    "Successfully joined session",
	})
	c.Send <- *successMsg
}

// handleLeaveSession handles session leave requests
func (c *Client) handleLeaveSession(message Message) {
	var leaveMsg JoinSessionMessage // Same structure as join
	if err := message.ParseData(&leaveMsg); err != nil {
		c.sendError("INVALID_LEAVE_DATA", "Invalid leave session data")
		return
	}

	c.Hub.LeaveSession(c, leaveMsg.SessionID)

	// Send success response
	successMsg, _ := NewMessage(MessageTypeUserLeft, map[string]interface{}{
		"session_id": leaveMsg.SessionID,
		"user_id":    c.UserID,
		"username":   c.Username,
		"message":    "Successfully left session",
	})
	c.Send <- *successMsg
}

// handleCollabUpdate handles collaborative update messages
func (c *Client) handleCollabUpdate(message Message) {
	if c.UserID == nil {
		c.sendError("NOT_AUTHENTICATED", "Authentication required")
		return
	}

	var problemState ProblemState
	if err := message.ParseData(&problemState); err != nil {
		c.sendError("INVALID_UPDATE_DATA", "Invalid collaboration update data")
		return
	}

	// Update the problem state with current user info
	problemState.UpdatedBy = *c.UserID
	problemState.UpdatedAt = time.Now()

	// If message has session ID, update that session
	if message.SessionID != nil {
		if err := c.Hub.UpdateProblemState(*message.SessionID, &problemState); err != nil {
			c.sendError("UPDATE_FAILED", "Failed to update session")
			return
		}
	}
}

// sendError sends an error message to the client
func (c *Client) sendError(code, message string) {
	errorMsg, _ := NewMessage(MessageTypeError, ErrorMessage{
		Code:    code,
		Message: message,
	})
	c.Send <- *errorMsg
}

// IsAuthenticated checks if the client is authenticated
func (c *Client) IsAuthenticated() bool {
	return c.UserID != nil
}

// GetUserInfo returns user information for the client
func (c *Client) GetUserInfo() map[string]interface{} {
	if c.UserID == nil {
		return nil
	}

	return map[string]interface{}{
		"user_id":  *c.UserID,
		"username": c.Username,
		"role":     c.Role,
	}
}

// handleTypingIndicator handles typing indicator messages
func (c *Client) handleTypingIndicator(message Message) {
	if c.UserID == nil {
		c.sendError("NOT_AUTHENTICATED", "Authentication required")
		return
	}

	var typingMsg TypingIndicatorMessage
	if err := message.ParseData(&typingMsg); err != nil {
		c.sendError("INVALID_TYPING_DATA", "Invalid typing indicator data")
		return
	}

	// Update typing indicator with current user info
	typingMsg.UserID = *c.UserID
	typingMsg.Username = c.Username

	// Broadcast to session members
	broadcastMsg, _ := NewMessage(MessageTypeTypingIndicator, typingMsg)
	broadcastMsg.SessionID = &typingMsg.SessionID
	c.Hub.Broadcast <- *broadcastMsg
}

// handleConnectionStatus handles connection status messages
func (c *Client) handleConnectionStatus(message Message) {
	var statusMsg ConnectionStatusMessage
	if err := message.ParseData(&statusMsg); err != nil {
		c.sendError("INVALID_STATUS_DATA", "Invalid connection status data")
		return
	}

	// Update client connection info
	c.LastPing = time.Now()
	
	// Send status acknowledgment
	ackMsg, _ := NewMessage(MessageTypeConnectionStatus, map[string]interface{}{
		"status":    "acknowledged",
		"timestamp": time.Now(),
		"client_id": c.ID,
	})
	c.Send <- *ackMsg
}

// sendReconnectInstruction sends reconnection instructions to client
func (c *Client) sendReconnectInstruction(reason string, retryAfter int) {
	reconnectMsg, _ := NewMessage(MessageTypeReconnect, ReconnectMessage{
		Reason:      reason,
		RetryAfter:  retryAfter,
		MaxRetries:  5,
		BackoffType: "exponential",
	})
	c.Send <- *reconnectMsg
}

// updateLastPing updates the client's last ping time
func (c *Client) updateLastPing() {
	c.LastPing = time.Now()
}

// isConnectionHealthy checks if the connection is healthy based on ping timing
func (c *Client) isConnectionHealthy() bool {
	return time.Since(c.LastPing) < pongWait*2
}

// getConnectionDuration returns how long the client has been connected
func (c *Client) getConnectionDuration() time.Duration {
	return time.Since(c.ConnectedAt)
}

// incrementReconnectCount increments the reconnection counter
func (c *Client) incrementReconnectCount() {
	c.ReconnectCount++
}

// resetReconnectCount resets the reconnection counter
func (c *Client) resetReconnectCount() {
	c.ReconnectCount = 0
}