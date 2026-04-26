package crypto

import (
	"testing"
)

func TestP2PChat_Stub(t *testing.T) {
	// P2P Chat requires a full libp2p stack which is tested in smoke tests.
	// This stub keeps the package tests passing.
	if testing.Short() {
		t.Skip("skipping P2P test in short mode")
	}
}
