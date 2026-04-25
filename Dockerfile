# Stage 1: Industrial-Grade Build Environment
FROM golang:1.26-alpine AS builder

# Install build dependencies
RUN apk add --no-cache git ca-certificates tzdata && update-ca-certificates

WORKDIR /app

# Leverage Docker cache for dependencies
COPY go.mod go.sum ./
RUN go mod download

# Copy source and build
COPY . .
# CGO_ENABLED=0 ensures a statically linked binary for the 'scratch' image
# -ldflags="-s -w" strips debug information to minimize binary size
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w" \
    -o maknoon ./cmd/maknoon

# Stage 2: Final Secure Sandbox (Zero-OS Attack Surface)
FROM scratch

# Import system artifacts from builder
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /usr/share/zoneinfo /usr/share/zoneinfo
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/passwd

# Copy the Maknoon engine
COPY --from=builder /app/maknoon /usr/local/bin/maknoon

# Standard Maknoon workspace directories
WORKDIR /home/maknoon
ENV HOME=/home/maknoon

# Run as a non-privileged user (Security Best Practice)
USER 1000:1000

# Default to the MCP server entry point for AI Agent integration
ENTRYPOINT ["/usr/local/bin/maknoon"]
CMD ["mcp"]
