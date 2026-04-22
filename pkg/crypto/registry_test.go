package crypto

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestNostrRegistry(t *testing.T) {
	// 1. Generate keys for an identity
	kpub, kpriv, spub, spriv, npub, npriv, err := GeneratePQKeyPair()
	if err != nil {
		t.Fatalf("GeneratePQKeyPair failed: %v", err)
	}
	defer SafeClear(kpriv)
	defer SafeClear(spriv)
	defer SafeClear(npriv)

	handle := fmt.Sprintf("@nostr:%s", string(npub))
	record := &IdentityRecord{
		Handle:    handle,
		KEMPubKey: kpub,
		SIGPubKey: spub,
		Timestamp: time.Time{},
	}
	if err := record.Sign(spriv); err != nil {
		t.Fatalf("Sign failed: %v", err)
	}

	// 2. Mock Nostr Relay behavior
	// We verify the serialization and parsing logic that would run after fetching from a relay.
	recordStr, _ := GetCompactDNSRecordString(record)
	dataIdx := strings.Index(recordStr, "data=")
	if dataIdx == -1 {
		t.Fatal("Failed to generate compact DNS record")
	}
	maknoonData := recordStr[dataIdx+5:]

	metadata := map[string]interface{}{
		"maknoon": maknoonData,
	}
	content, _ := json.Marshal(metadata)

	// Verify our parsing logic works on this content
	var parsedMetadata map[string]interface{}
	if err := json.Unmarshal(content, &parsedMetadata); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	val, ok := parsedMetadata["maknoon"].(string)
	if !ok || val != maknoonData {
		t.Fatalf("Metadata 'maknoon' field mismatch")
	}

	// 3. Verify parseMaknoonTXT can handle the data with compression (z=1)
	parsedRecord, err := parseMaknoonTXT("v=maknoon1;z=1;data=" + val)
	if err != nil {
		t.Fatalf("parseMaknoonTXT failed: %v", err)
	}

	if !bytes.Equal(parsedRecord.KEMPubKey, record.KEMPubKey) {
		t.Error("KEM public key mismatch after roundtrip")
	}

	if !parsedRecord.Verify() {
		t.Error("Parsed record failed signature verification")
	}
}
