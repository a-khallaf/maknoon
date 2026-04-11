// Package crypto provides the core cryptographic primitives and streaming
// encryption logic for Maknoon.
package crypto

// GeneratePQKeyPair generates a fresh ML-KEM and ML-DSA keypair using the default profile.
func GeneratePQKeyPair() (kemPub, kemPriv, sigPub, sigPriv []byte, err error) {
	profile := DefaultProfile()

	kemPub, kemPriv, err = profile.GenerateKEMKeyPair()
	if err != nil {
		return nil, nil, nil, nil, err
	}

	sigPub, sigPriv, err = profile.GenerateSIGKeyPair()
	if err != nil {
		// Zero out KEM keys before failing
		for i := range kemPriv {
			kemPriv[i] = 0
		}
		return nil, nil, nil, nil, err
	}

	return
}

// SignData signs a message using a Post-Quantum private key.
func SignData(message []byte, privKeyBytes []byte) ([]byte, error) {
	// Zero out the input bytes after unmarshaling to protect memory
	defer func() {
		for i := range privKeyBytes {
			privKeyBytes[i] = 0
		}
	}()

	return DefaultProfile().Sign(message, privKeyBytes)
}

// VerifySignature verifies a Post-Quantum signature against a message and public key.
func VerifySignature(message []byte, signature []byte, pubKeyBytes []byte) bool {
	return DefaultProfile().Verify(message, signature, pubKeyBytes)
}
