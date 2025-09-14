package grpc

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// MathGRPCClient wraps the gRPC client for math operations
type MathGRPCClient struct {
	conn   *grpc.ClientConn
	client MathServiceClient
}

// NewMathGRPCClient creates a new gRPC client for math operations
func NewMathGRPCClient() (*MathGRPCClient, error) {
	// Get gRPC server address from environment
	grpcAddr := os.Getenv("MATH_GRPC_ADDR")
	if grpcAddr == "" {
		grpcAddr = "localhost:8002" // Default gRPC port for math engine
	}

	// Create connection with timeout
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, grpcAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to math gRPC service: %v", err)
	}

	client := NewMathServiceClient(conn)

	return &MathGRPCClient{
		conn:   conn,
		client: client,
	}, nil
}

// Close closes the gRPC connection
func (c *MathGRPCClient) Close() error {
	return c.conn.Close()
}

// ParseProblemGRPC parses a problem using gRPC
func (c *MathGRPCClient) ParseProblemGRPC(ctx context.Context, problemText, domain, userID string) (*ParsedProblem, error) {
	req := &ParseProblemRequest{
		ProblemText: problemText,
		Domain:      domain,
		UserId:      userID,
	}

	resp, err := c.client.ParseProblem(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("gRPC parse problem failed: %v", err)
	}

	if resp.Error != "" {
		return nil, fmt.Errorf("math engine error: %s", resp.Error)
	}

	return resp.Problem, nil
}

// SolveProblemGRPC solves a problem using gRPC
func (c *MathGRPCClient) SolveProblemGRPC(ctx context.Context, problem *ParsedProblem) (*StepSolution, error) {
	req := &SolveProblemRequest{
		Problem: problem,
	}

	resp, err := c.client.SolveProblem(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("gRPC solve problem failed: %v", err)
	}

	if resp.Error != "" {
		return nil, fmt.Errorf("math engine error: %s", resp.Error)
	}

	return resp.Solution, nil
}

// ValidateAnswerGRPC validates an answer using gRPC
func (c *MathGRPCClient) ValidateAnswerGRPC(ctx context.Context, problem *ParsedProblem, answer string) (*ValidationResult, error) {
	req := &ValidateAnswerRequest{
		Problem: problem,
		Answer:  answer,
	}

	resp, err := c.client.ValidateAnswer(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("gRPC validate answer failed: %v", err)
	}

	if resp.Error != "" {
		return nil, fmt.Errorf("math engine error: %s", resp.Error)
	}

	return resp.Result, nil
}

// Global gRPC client instance
var globalGRPCClient *MathGRPCClient

// InitGRPCClient initializes the global gRPC client
func InitGRPCClient() error {
	client, err := NewMathGRPCClient()
	if err != nil {
		log.Printf("Warning: Failed to initialize gRPC client, falling back to HTTP: %v", err)
		return err
	}

	globalGRPCClient = client
	log.Println("✅ gRPC client initialized successfully")
	return nil
}

// GetGRPCClient returns the global gRPC client
func GetGRPCClient() *MathGRPCClient {
	return globalGRPCClient
}

// CloseGRPCClient closes the global gRPC client
func CloseGRPCClient() {
	if globalGRPCClient != nil {
		globalGRPCClient.Close()
	}
}

// NOTE: This file requires the generated protobuf files from math_service.proto
// To generate them, run:
// protoc --go_out=. --go_opt=paths=source_relative --go-grpc_out=. --go-grpc_opt=paths=source_relative internal/grpc/math_service.proto
//
// You'll need to install the protobuf compiler and Go plugins:
// go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
// go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest