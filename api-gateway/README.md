# AI Math Tutor - API Gateway

This is the Go-based API Gateway for the AI Math Tutor system. It provides REST API endpoints for mathematical operations and handles authentication, rate limiting, and service communication.

## Features

- **REST API endpoints** for mathematical problem solving
- **Rate limiting** to prevent abuse
- **Authentication middleware** with JWT tokens
- **Request validation** and error handling
- **Service communication** with Python math engine
- **gRPC support** (when protobuf files are generated)
- **Comprehensive testing** with unit and integration tests

## API Endpoints

### Mathematical Operations

All math endpoints are under `/api/v1/problems/` and include rate limiting (30 requests/minute).

#### Parse Problem
```http
POST /api/v1/problems/parse
Content-Type: application/json

{
  "problem_text": "Solve for x: 2x + 3 = 7",
  "domain": "algebra",
  "user_id": "optional-user-id"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "uuid",
    "original_text": "Solve for x: 2x + 3 = 7",
    "domain": "algebra",
    "difficulty": "beginner",
    "variables": ["x"],
    "expressions": ["2x + 3 = 7"],
    "problem_type": "linear_equation",
    "metadata": {}
  }
}
```

#### Solve Problem Step-by-Step
```http
POST /api/v1/problems/solve
Content-Type: application/json

{
  "problem": {
    "id": "uuid",
    "original_text": "Solve for x: 2x + 3 = 7",
    "domain": "algebra",
    "difficulty": "beginner",
    "variables": ["x"],
    "expressions": ["2x + 3 = 7"],
    "problem_type": "linear_equation",
    "metadata": {}
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "problem_id": "uuid",
    "steps": [
      {
        "step_number": 1,
        "operation": "Subtract 3 from both sides",
        "explanation": "To isolate the variable term",
        "mathematical_expression": "2x = 4",
        "intermediate_result": "2x = 4",
        "reasoning": "Moving constants to the right side"
      },
      {
        "step_number": 2,
        "operation": "Divide both sides by 2",
        "explanation": "To solve for x",
        "mathematical_expression": "x = 2",
        "intermediate_result": "x = 2"
      }
    ],
    "final_answer": "2",
    "solution_method": "Linear equation solving",
    "confidence_score": 0.95,
    "computation_time": 0.1
  }
}
```

#### Validate Answer
```http
POST /api/v1/problems/validate
Content-Type: application/json

{
  "problem": { /* parsed problem object */ },
  "answer": "2"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_correct": true,
    "user_answer": "2",
    "correct_answer": "2",
    "explanation": "Correct! The answer is 2.",
    "partial_credit": 1.0
  }
}
```

#### Generate Visualization
```http
POST /api/v1/problems/visualize
Content-Type: application/json

{
  "problem": { /* parsed problem object */ },
  "viz_type": "auto"
}
```

#### Generate Hint
```http
POST /api/v1/problems/hint
Content-Type: application/json

{
  "problem": { /* parsed problem object */ },
  "current_step": 1
}
```

#### Explain Step
```http
POST /api/v1/problems/explain
Content-Type: application/json

{
  "step": { /* solution step object */ },
  "user_level": "standard"
}
```

### Authentication Endpoints

#### Register
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "username": "username",
  "password": "password"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password"
}
```

#### Logout
```http
POST /api/v1/auth/logout
Authorization: Bearer <jwt-token>
```

### Health Check
```http
GET /health
```

## Rate Limiting

- **Math operations**: 30 requests per minute per IP
- **General API**: 100 requests per minute per IP
- Rate limiters automatically clean up expired entries

## Environment Variables

- `PORT`: Server port (default: 8000)
- `DATABASE_URL`: PostgreSQL connection string
- `MATH_ENGINE_URL`: Math engine service URL (default: http://localhost:8001)
- `MATH_GRPC_ADDR`: Math engine gRPC address (default: localhost:8002)
- `GIN_MODE`: Gin framework mode (debug/release)

## Running the Service

### Prerequisites
- Go 1.21+
- PostgreSQL database
- Running math engine service (Python)

### Development
```bash
# Install dependencies
go mod tidy

# Run tests
go test ./...

# Run integration tests
go test -v integration_test.go

# Build
go build -o api-gateway .

# Run
./api-gateway
```

### Docker
```bash
docker build -t ai-math-tutor-gateway .
docker run -p 8000:8000 ai-math-tutor-gateway
```

## Testing

### Unit Tests
```bash
go test ./internal/math -v
go test ./internal/auth -v
go test ./internal/middleware -v
```

### Integration Tests
```bash
go test -v integration_test.go
```

### Load Testing
```bash
go test -bench=. -v
```

## gRPC Support

gRPC support is implemented but requires generating protobuf files:

```bash
# Install protobuf compiler and Go plugins
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# Generate protobuf files
protoc --go_out=. --go_opt=paths=source_relative \
       --go-grpc_out=. --go-grpc_opt=paths=source_relative \
       internal/grpc/math_service.proto
```

## Error Handling

All endpoints return consistent error responses:

```json
{
  "success": false,
  "error": "Error description",
  "data": null
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │───▶│   API Gateway    │───▶│  Math Engine    │
│   (React/JS)    │    │   (Go/Gin)       │    │   (Python)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   PostgreSQL     │
                       │   Database       │
                       └──────────────────┘
```

## Contributing

1. Follow Go best practices and conventions
2. Add tests for new functionality
3. Update documentation for API changes
4. Use structured logging
5. Handle errors gracefully

## Security

- JWT-based authentication
- Input validation and sanitization
- Rate limiting per IP address
- CORS configuration
- SQL injection prevention
- Secure headers