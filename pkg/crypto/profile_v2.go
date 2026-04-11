package crypto

import (
	"crypto/aes"
	"crypto/cipher"

	"golang.org/x/crypto/argon2"
)

func init() {
	RegisterProfile(&ProfileV2{})
}

// ProfileV2 implements a "High-Compatibility" suite using AES-256-GCM and a faster KDF.
// Note: This is mainly to demonstrate cryptographic agility.
type ProfileV2 struct {
	ProfileV1 // Inherit KEM/SIG from V1 for now
}

func (p *ProfileV2) ID() byte { return 2 }

func (p *ProfileV2) SaltSize() int { return 16 } // Smaller salt for faster KDF

func (p *ProfileV2) NonceSize() int { return 12 } // AES-GCM standard nonce

func (p *ProfileV2) DeriveKey(passphrase, salt []byte) []byte {
	// Faster Argon2 settings (1 iteration, 16MB)
	return argon2.IDKey(passphrase, salt, 1, 16*1024, 4, 32)
}

func (p *ProfileV2) NewAEAD(key []byte) (cipher.AEAD, error) {
	block, err := aes.NewCipher(key)
	if err != nil {
		return nil, err
	}
	return cipher.NewGCM(block)
}

func (p *ProfileV2) GenerateKEMKeyPair() (pub, priv []byte, err error) {
	// For V2, we'll keep using the same KEM as V1 for identity compatibility
	return p.ProfileV1.GenerateKEMKeyPair()
}
