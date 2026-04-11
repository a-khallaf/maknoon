package crypto

import (
	"crypto/cipher"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
)

// CryptoProfile defines the cryptographic primitives and parameters for a Maknoon version.
type CryptoProfile interface {
	ID() byte

	// KDF & Symmetric Encryption
	SaltSize() int
	NonceSize() int
	DeriveKey(passphrase, salt []byte) []byte
	NewAEAD(key []byte) (cipher.AEAD, error)

	// Asymmetric KEM (Key Encapsulation Mechanism)
	KEMName() string
	GenerateKEMKeyPair() (pub, priv []byte, err error)
	KEMEncapsulate(pubKey []byte) (ct, ss []byte, err error)
	KEMDecapsulate(privKey, ct []byte) (ss []byte, err error)
	KEMCiphertextSize() int

	// Digital Signatures
	SIGName() string
	GenerateSIGKeyPair() (pub, priv []byte, err error)
	Sign(data, privKey []byte) ([]byte, error)
	Verify(data, sig, pubKey []byte) bool
}

var (
	profiles = make(map[byte]CryptoProfile)
	mu       sync.RWMutex
)

// RegisterProfile adds a new cryptographic profile to the registry.
func RegisterProfile(p CryptoProfile) {
	mu.Lock()
	defer mu.Unlock()
	profiles[p.ID()] = p
}

// GetProfile retrieves a cryptographic profile by its ID.
// 1. Checks memory registry.
// 2. If ID < 128, attempts to auto-load from ~/.maknoon/profiles/ID.json.
// 3. If ID >= 128, reads 7 packed bytes from r to unpack a dynamic profile.
func GetProfile(id byte, r io.Reader) (CryptoProfile, error) {
	mu.RLock()
	p, ok := profiles[id]
	mu.RUnlock()
	if ok {
		return p, nil
	}

	// Automatic Discovery for Secret Profiles (3-127)
	if id > 2 && id < 128 {
		home, _ := os.UserHomeDir()
		profilePath := filepath.Join(home, MaknoonDir, ProfilesDir, fmt.Sprintf("%d.json", id))
		if _, err := os.Stat(profilePath); err == nil {
			raw, err := os.ReadFile(profilePath)
			if err == nil {
				var dp DynamicProfile
				if err := json.Unmarshal(raw, &dp); err == nil {
					if err := dp.Validate(); err == nil {
						RegisterProfile(&dp)
						return &dp, nil
					}
				}
			}
		}
	}

	if id >= 128 {
		if r == nil {
			return nil, fmt.Errorf("reader required for unknown dynamic profile ID: %d", id)
		}
		packed := make([]byte, 7)
		if _, err := io.ReadFull(r, packed); err != nil {
			return nil, fmt.Errorf("failed to read packed profile: %w", err)
		}
		dp, err := UnpackDynamicProfile(id, packed)
		if err != nil {
			return nil, err
		}
		if err := dp.Validate(); err != nil {
			return nil, fmt.Errorf("embedded profile validation failed: %w", err)
		}
		return dp, nil
	}

	return nil, fmt.Errorf("unsupported cryptographic profile ID: %d", id)
}

// DefaultProfile returns the standard NIST PQC profile (v1).
func DefaultProfile() CryptoProfile {
	p, _ := GetProfile(1, nil)
	return p
}
