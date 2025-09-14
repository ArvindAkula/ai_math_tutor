package websocket

import (
	"errors"
	"log"
	"time"

	"github.com/google/uuid"
)

// Custom errors
var (
	ErrSessionNotFound   = errors.New("session not found")
	ErrNotAuthenticated  = errors.New("client not authenticated")
	ErrInvalidMessage    = errors.New("invalid message format")
	ErrPermissionDenied  = errors.New("permission denied")
)

// Run starts the WebSocket hub
func (h *Hub) Run() {
	// Start cleanup routine for inactive sessions
	go h.cleanupRoutine()

	for {
		select {
		case client := <-h.Register:
			h.registerClient(client)

		case client := <-h.Unregister:
			h.unregisterClient(client)

		case message := <-h.Broadcast:
			h.broadcastMessage(message)
		}
	}
}

// registerClient registers a new client
func (h *Hub) registerClient(client *Client) {
	h.Clients[client.ID] = client
	log.Printf("Client %s connected (User: %v)", client.ID, client.UserID)

	// Send authentication request if not authenticated
	if client.UserID == nil {
		authMsg, _ := NewMessage(MessageTypeAuth, map[string]string{
			"message": "Please authenticate with a valid JWT token",
		})
		select {
		case client.Send <- *authMsg:
		default:
			close(client.Send)
			delete(h.Clients, client.ID)
		}
	}
}

// unregisterClient unregisters a client
func (h *Hub) unregisterClient(client *Client) {
	if _, ok := h.Clients[client.ID]; ok {
		// Remove client from all sessions
		for sessionID := range client.Sessions {
			h.removeClientFromSession(client, sessionID)
		}

		// Close client connection
		delete(h.Clients, client.ID)
		close(client.Send)
		
		log.Printf("Client %s disconnected (User: %v)", client.ID, client.UserID)
	}
}

// broadcastMessage broadcasts a message to relevant clients
func (h *Hub) broadcastMessage(message Message) {
	// If message has a session ID, broadcast only to session members
	if message.SessionID != nil {
		h.broadcastToSession(*message.SessionID, message)
		return
	}

	// If message has a user ID, send only to that user's clients
	if message.UserID != nil {
		h.sendToUser(*message.UserID, message)
		return
	}

	// Otherwise, broadcast to all authenticated clients
	for _, client := range h.Clients {
		if client.UserID != nil {
			select {
			case client.Send <- message:
			default:
				h.unregisterClient(client)
			}
		}
	}
}

// broadcastToSession broadcasts a message to all clients in a session
func (h *Hub) broadcastToSession(sessionID uuid.UUID, message Message) {
	session, exists := h.Sessions[sessionID]
	if !exists {
		return
	}

	for _, client := range session.Clients {
		select {
		case client.Send <- message:
		default:
			h.unregisterClient(client)
		}
	}
}

// sendToUser sends a message to all clients of a specific user
func (h *Hub) sendToUser(userID uuid.UUID, message Message) {
	for _, client := range h.Clients {
		if client.UserID != nil && *client.UserID == userID {
			select {
			case client.Send <- message:
			default:
				h.unregisterClient(client)
			}
		}
	}
}

// CreateSession creates a new collaborative session
func (h *Hub) CreateSession(name string, createdBy uuid.UUID) *Session {
	session := NewSession(name, createdBy)
	h.Sessions[session.ID] = session
	
	log.Printf("Session %s created by user %s", session.ID, createdBy)
	return session
}

// JoinSession adds a client to a session
func (h *Hub) JoinSession(client *Client, sessionID uuid.UUID) error {
	session, exists := h.Sessions[sessionID]
	if !exists {
		return ErrSessionNotFound
	}

	if client.UserID == nil {
		return ErrNotAuthenticated
	}

	// Add client to session
	session.Clients[client.ID] = client
	client.Sessions[sessionID] = true

	// Notify other session members
	joinMsg, _ := NewMessage(MessageTypeUserJoined, map[string]interface{}{
		"user_id":    *client.UserID,
		"username":   client.Username,
		"session_id": sessionID,
	})
	joinMsg.SessionID = &sessionID
	h.Broadcast <- *joinMsg

	log.Printf("Client %s (User: %s) joined session %s", client.ID, *client.UserID, sessionID)
	return nil
}

// LeaveSession removes a client from a session
func (h *Hub) LeaveSession(client *Client, sessionID uuid.UUID) {
	h.removeClientFromSession(client, sessionID)
}

// removeClientFromSession removes a client from a session
func (h *Hub) removeClientFromSession(client *Client, sessionID uuid.UUID) {
	session, exists := h.Sessions[sessionID]
	if !exists {
		return
	}

	// Remove client from session
	delete(session.Clients, client.ID)
	delete(client.Sessions, sessionID)

	// Notify other session members if client was authenticated
	if client.UserID != nil {
		leaveMsg, _ := NewMessage(MessageTypeUserLeft, map[string]interface{}{
			"user_id":    *client.UserID,
			"username":   client.Username,
			"session_id": sessionID,
		})
		leaveMsg.SessionID = &sessionID
		h.Broadcast <- *leaveMsg
	}

	// Clean up empty sessions
	if len(session.Clients) == 0 {
		session.IsActive = false
		delete(h.Sessions, sessionID)
		log.Printf("Session %s closed (no active clients)", sessionID)
	}

	log.Printf("Client %s (User: %v) left session %s", client.ID, client.UserID, sessionID)
}

// UpdateProblemState updates the problem state in a session
func (h *Hub) UpdateProblemState(sessionID uuid.UUID, problemState *ProblemState) error {
	session, exists := h.Sessions[sessionID]
	if !exists {
		return ErrSessionNotFound
	}

	session.CurrentProblem = problemState
	
	// Broadcast update to session members
	updateMsg, _ := NewMessage(MessageTypeCollabUpdate, problemState)
	updateMsg.SessionID = &sessionID
	h.Broadcast <- *updateMsg

	return nil
}

// GetSession returns a session by ID
func (h *Hub) GetSession(sessionID uuid.UUID) (*Session, bool) {
	session, exists := h.Sessions[sessionID]
	return session, exists
}

// GetClientSessions returns all sessions a client is part of
func (h *Hub) GetClientSessions(clientID uuid.UUID) []*Session {
	client, exists := h.Clients[clientID]
	if !exists {
		return nil
	}

	var sessions []*Session
	for sessionID := range client.Sessions {
		if session, exists := h.Sessions[sessionID]; exists {
			sessions = append(sessions, session)
		}
	}

	return sessions
}

// SendProgressUpdate sends a progress update to a user
func (h *Hub) SendProgressUpdate(userID uuid.UUID, update ProgressUpdateMessage) {
	msg, err := NewMessage(MessageTypeProgressUpdate, update)
	if err != nil {
		log.Printf("Error creating progress update message: %v", err)
		return
	}

	msg.UserID = &userID
	h.Broadcast <- *msg
}

// SendRealTimeSolution sends a real-time solution update to a session or user
func (h *Hub) SendRealTimeSolution(sessionID *uuid.UUID, userID *uuid.UUID, solution interface{}) {
	msg, err := NewMessage(MessageTypeProblemSolution, solution)
	if err != nil {
		log.Printf("Error creating solution message: %v", err)
		return
	}

	msg.SessionID = sessionID
	msg.UserID = userID
	h.Broadcast <- *msg
}

// SendHintToUser sends a hint to a specific user
func (h *Hub) SendHintToUser(userID uuid.UUID, hint interface{}) {
	msg, err := NewMessage(MessageTypeHintResponse, hint)
	if err != nil {
		log.Printf("Error creating hint message: %v", err)
		return
	}

	msg.UserID = &userID
	h.Broadcast <- *msg
}

// SendVisualizationUpdate sends visualization data to session or user
func (h *Hub) SendVisualizationUpdate(sessionID *uuid.UUID, userID *uuid.UUID, visualization interface{}) {
	msg, err := NewMessage(MessageTypeVisualizationUpdate, visualization)
	if err != nil {
		log.Printf("Error creating visualization message: %v", err)
		return
	}

	msg.SessionID = sessionID
	msg.UserID = userID
	h.Broadcast <- *msg
}

// BroadcastToAllUsers sends a message to all authenticated users
func (h *Hub) BroadcastToAllUsers(msgType MessageType, data interface{}) {
	msg, err := NewMessage(msgType, data)
	if err != nil {
		log.Printf("Error creating broadcast message: %v", err)
		return
	}

	h.Broadcast <- *msg
}

// SendTypingIndicator sends typing indicator to session members
func (h *Hub) SendTypingIndicator(sessionID uuid.UUID, userID uuid.UUID, isTyping bool) {
	msg, err := NewMessage(MessageTypeTypingIndicator, map[string]interface{}{
		"user_id":   userID,
		"is_typing": isTyping,
	})
	if err != nil {
		log.Printf("Error creating typing indicator message: %v", err)
		return
	}

	msg.SessionID = &sessionID
	h.Broadcast <- *msg
}

// SendAchievement sends an achievement notification to a user
func (h *Hub) SendAchievement(userID uuid.UUID, achievement interface{}) {
	msg, err := NewMessage(MessageTypeAchievement, achievement)
	if err != nil {
		log.Printf("Error creating achievement message: %v", err)
		return
	}

	msg.UserID = &userID
	h.Broadcast <- *msg
}

// cleanupRoutine periodically cleans up inactive sessions and connections
func (h *Hub) cleanupRoutine() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			h.cleanupInactiveSessions()
		}
	}
}

// cleanupInactiveSessions removes sessions that have been inactive for too long
func (h *Hub) cleanupInactiveSessions() {
	now := time.Now()
	inactiveThreshold := 30 * time.Minute

	for sessionID, session := range h.Sessions {
		if len(session.Clients) == 0 && now.Sub(session.CreatedAt) > inactiveThreshold {
			delete(h.Sessions, sessionID)
			log.Printf("Cleaned up inactive session %s", sessionID)
		}
	}
}

// GetStats returns hub statistics
func (h *Hub) GetStats() map[string]interface{} {
	authenticatedClients := 0
	for _, client := range h.Clients {
		if client.UserID != nil {
			authenticatedClients++
		}
	}

	activeSessions := 0
	for _, session := range h.Sessions {
		if session.IsActive && len(session.Clients) > 0 {
			activeSessions++
		}
	}

	return map[string]interface{}{
		"total_clients":        len(h.Clients),
		"authenticated_clients": authenticatedClients,
		"active_sessions":      activeSessions,
		"total_sessions":       len(h.Sessions),
	}
}

