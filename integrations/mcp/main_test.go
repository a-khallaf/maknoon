package main

import (
	"os"
	"testing"
)

func TestMain(m *testing.M) {
	os.Exit(m.Run())
}

func TestMCPServerTools_Registration(t *testing.T) {
	s := setupServer()
	tools := s.ListTools()

	expected := []string{
		"encrypt_file", "decrypt_file", "inspect_file",
		"p2p_send", "p2p_receive", "chat_start",
		"tunnel_start",
	}

	for _, name := range expected {
		if _, ok := tools[name]; !ok {
			t.Errorf("Tool %s not registered", name)
		}
	}
}

func TestMCPServerTools_P2PSend(t *testing.T) {
	s := setupServer()
	tools := s.ListTools()

	// Verify p2p_send exists
	_, ok := tools["p2p_send"]
	if !ok {
		t.Fatal("p2p_send tool not found")
	}
}
