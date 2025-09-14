# Multi-stage build for Go API Gateway
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download && go mod verify

# Copy source code
COPY . .

# Build the application with optimizations
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags='-w -s -extldflags "-static"' \
    -a -installsuffix cgo \
    -o main .

# Production stage
FROM alpine:latest AS production

# Install ca-certificates and create non-root user
RUN apk --no-cache add ca-certificates tzdata && \
    addgroup -g 1001 -S apigateway && \
    adduser -S apigateway -u 1001 -G apigateway

# Copy the binary from builder stage
COPY --from=builder /app/main /main

# Set proper permissions
RUN chown apigateway:apigateway /main && \
    chmod +x /main

# Switch to non-root user
USER apigateway

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8000/health || exit 1

# Run the application
ENTRYPOINT ["/main"]

# Development stage
FROM golang:1.21-alpine AS development

# Install development tools
RUN apk add --no-cache git ca-certificates tzdata curl

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Install air for hot reloading (compatible version)
RUN go install github.com/cosmtrek/air@v1.49.0

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start with hot reloading
CMD ["air"]