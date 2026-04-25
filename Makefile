.PHONY: build test docker-build clean completion man

# Build parameters
BINARY_NAME=maknoon
PKG=./cmd/maknoon
LDFLAGS=-s -w

# Standard Production Build (Statically linked, Stripped symbols)
build:
	@echo "🛠️  Building production-grade binary..."
	CGO_ENABLED=0 go build -ldflags="$(LDFLAGS)" -o $(BINARY_NAME) $(PKG)
	@ls -lh $(BINARY_NAME)

# Fast local test suite (Skips network/flaky tests)
test:
	@echo "🧪  Running optimized test suite..."
	go test -v -short ./...

# Full containerized sandbox build using BuildKit
docker-build:
	@echo "📦  Building secure scratch container..."
	docker buildx build -t maknoon-sandbox --load .

# Generate shell completions for the current session
completion:
	@echo "🐚  Generating bash completions..."
	./$(BINARY_NAME) completion bash > maknoon.completion
	@echo "Source 'maknoon.completion' to enable."

# Verify and update the manual page
man:
	@echo "📖  Verifying manual page integrity..."
	go run $(PKG) man --verify

# Cleanup build artifacts
clean:
	@echo "🧹  Cleaning up..."
	rm -f $(BINARY_NAME) maknoon_lean maknoon.completion
	go clean
