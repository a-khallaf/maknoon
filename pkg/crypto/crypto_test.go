package crypto

import (
	"bytes"
	"testing"
)

func TestSymmetricRoundTrip(t *testing.T) {
	data := []byte("This is a secret message for symmetric test.")
	passphrase := []byte("correct-passphrase-123")

	// 1. Encrypt
	var encrypted bytes.Buffer
	if err := EncryptStream(bytes.NewReader(data), &encrypted, passphrase, FlagNone, 0, 0); err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// 2. Decrypt
	var decrypted bytes.Buffer
	if _, err := DecryptStream(bytes.NewReader(encrypted.Bytes()), &decrypted, passphrase, 0); err != nil {
		t.Fatalf("Decryption failed: %v", err)
	}

	if !bytes.Equal(data, decrypted.Bytes()) {
		t.Errorf("Decrypted data mismatch. Got %s, want %s", decrypted.String(), string(data))
	}

	// 3. Wrong passphrase should fail
	var decryptedWrong bytes.Buffer
	if _, err := DecryptStream(bytes.NewReader(encrypted.Bytes()), &decryptedWrong, []byte("wrong-pass"), 0); err == nil {
		t.Error("Expected error with wrong passphrase, got nil")
	}
}

func TestAsymmetricRoundTrip(t *testing.T) {
	data := []byte("Post-Quantum Asymmetric Encryption Test Data")
	profile := DefaultProfile()
	pub, priv, err := profile.GenerateKEMKeyPair()
	if err != nil {
		t.Fatal(err)
	}

	// 1. Encrypt
	var encrypted bytes.Buffer
	if err := EncryptStreamWithPublicKey(bytes.NewReader(data), &encrypted, pub, FlagNone, 0, 0); err != nil {
		t.Fatalf("Asymmetric encryption failed: %v", err)
	}

	// 2. Decrypt
	var decrypted bytes.Buffer
	if _, err := DecryptStreamWithPrivateKey(bytes.NewReader(encrypted.Bytes()), &decrypted, priv, 0); err != nil {
		t.Fatalf("Asymmetric decryption failed: %v", err)
	}

	if !bytes.Equal(data, decrypted.Bytes()) {
		t.Errorf("Asymmetric mismatch. Got %s, want %s", decrypted.String(), string(data))
	}
}

func TestIntegratedSignThenEncryptUnit(t *testing.T) {
	data := []byte("Sign-then-Encrypt Unit Test")
	profile := DefaultProfile()
	
	// Recipient keys
	pub, priv, _ := profile.GenerateKEMKeyPair()
	// Sender keys
	spub, spriv, _ := profile.GenerateSIGKeyPair()

	// 1. Encrypt with integrated signature
	var encrypted bytes.Buffer
	if err := EncryptStreamWithPublicKeysAndSigner(bytes.NewReader(data), &encrypted, [][]byte{pub}, spriv, FlagNone, 0, 0); err != nil {
		t.Fatalf("Integrated encryption failed: %v", err)
	}

	// 2. Decrypt and Verify
	var decrypted bytes.Buffer
	// Test failure without sender key
	_, err := DecryptStreamWithPrivateKey(bytes.NewReader(encrypted.Bytes()), &decrypted, priv, 0)
	if err == nil || !bytes.Contains([]byte(err.Error()), []byte("sender public key not provided")) {
		t.Errorf("Expected error for missing sender key, got: %v", err)
	}

	// Test success with sender key
	decrypted.Reset()
	_, err = DecryptStreamWithPrivateKeyAndVerifier(bytes.NewReader(encrypted.Bytes()), &decrypted, priv, spub, 0)
	if err != nil {
		t.Fatalf("Integrated decryption failed: %v", err)
	}

	if !bytes.Equal(data, decrypted.Bytes()) {
		t.Errorf("Content mismatch. Got %q", decrypted.String())
	}
}
