package main

import (
	"testing"
)

func TestTransport_Setup(t *testing.T) {
	s := setupServer()
	tools := s.ListTools()
	if len(tools) == 0 {
		t.Error("Expected at least one tool to be registered")
	}
}
