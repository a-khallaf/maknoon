package main

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestMCPServerTools(t *testing.T) {
	// Build the maknoon binary for the MCP server to use
	tmpDir := t.TempDir()
	maknoonPath := filepath.Join(tmpDir, "maknoon")
	buildCmd := exec.Command("go", "build", "-o", maknoonPath, "../../cmd/maknoon")
	if err := buildCmd.Run(); err != nil {
		t.Fatalf("Failed to build maknoon for tests: %v", err)
	}

	// Set environment for the server to find the binary and a mock home
	os.Setenv("MAKNOON_BINARY", maknoonPath)
	os.Setenv("HOME", tmpDir)
	defer os.Unsetenv("MAKNOON_BINARY")

	s := createServer()
	ctx := context.Background()

	t.Run("Tool List", func(t *testing.T) {
		tools := s.ListTools()
		if len(tools) == 0 {
			t.Fatal("Expected at least one tool, got zero")
		}
		expectedTools := []string{"vault_get", "vault_set", "encrypt_file", "identity_active"}
		for _, name := range expectedTools {
			if _, ok := tools[name]; !ok {
				t.Errorf("Missing expected tool: %s", name)
			}
		}
	})

	t.Run("Identity Active Success", func(t *testing.T) {
		// Generate an identity first
		keygenCmd := exec.Command(maknoonPath, "keygen", "-o", "test-id", "--no-password")
		keygenCmd.Env = append(os.Environ(), "MAKNOON_JSON=1")
		if err := keygenCmd.Run(); err != nil {
			t.Fatalf("Failed to generate test identity: %v", err)
		}

		req := json.RawMessage(`{
			"jsonrpc": "2.0",
			"id": "1",
			"method": "tools/call",
			"params": {
				"name": "identity_active",
				"arguments": {}
			}
		}`)

		res := s.HandleMessage(ctx, req)
		resRaw, _ := json.Marshal(res)
		if !strings.Contains(string(resRaw), "test-id.kem.pub") {
			t.Errorf("Identity discovery failed. Result: %s", string(resRaw))
		}
	})

	t.Run("Vault Get Error (Missing Master Key)", func(t *testing.T) {
		req := json.RawMessage(`{
			"jsonrpc": "2.0",
			"id": "2",
			"method": "tools/call",
			"params": {
				"name": "vault_get",
				"arguments": {
					"service": "test-service"
				}
			}
		}`)

		res := s.HandleMessage(ctx, req)
		resRaw, _ := json.Marshal(res)
		if !strings.Contains(string(resRaw), "master passphrase required") {
			t.Errorf("Expected master key error, got: %s", string(resRaw))
		}
	})
}
