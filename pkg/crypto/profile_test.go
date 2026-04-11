package crypto

import (
	"crypto/cipher"
	"testing"

	"golang.org/x/crypto/chacha20poly1305"
)

// MockProfileV2 is a lightweight profile for testing.
type MockProfileV2 struct {
	ProfileV1
}

func (p *MockProfileV2) ID() byte { return 2 }

func (p *MockProfileV2) NewAEAD(key []byte) (cipher.AEAD, error) {
	// Use standard ChaCha20-Poly1305 (12-byte nonce) instead of XChaCha20
	return chacha20poly1305.New(key)
}

func TestProfileRegistry(t *testing.T) {
	RegisterProfile(&MockProfileV2{})

	p1, err := GetProfile(1)
	if err != nil {
		t.Fatalf("Failed to get profile 1: %v", err)
	}
	if p1.ID() != 1 {
		t.Errorf("Expected ID 1, got %d", p1.ID())
	}

	p2, err := GetProfile(2)
	if err != nil {
		t.Fatalf("Failed to get profile 2: %v", err)
	}
	if p2.ID() != 2 {
		t.Errorf("Expected ID 2, got %d", p2.ID())
	}

	_, err = GetProfile(99)
	if err == nil {
		t.Error("Expected error for non-existent profile ID 99")
	}
}

func TestDefaultProfile(t *testing.T) {
	p := DefaultProfile()
	if p == nil {
		t.Fatal("Default profile should not be nil")
	}
	if p.ID() != 1 {
		t.Errorf("Expected default ID 1, got %d", p.ID())
	}
}
