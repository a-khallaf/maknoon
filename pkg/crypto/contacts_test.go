package crypto

import (
	"bytes"
	"path/filepath"
	"testing"
	"time"

	"go.etcd.io/bbolt"
)

func TestContactManager(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "contacts.db")

	// 1. Initialize
	db, err := bbolt.Open(dbPath, 0600, nil)
	if err != nil {
		t.Fatalf("failed to open db: %v", err)
	}
	err = db.Update(func(tx *bbolt.Tx) error {
		_, err := tx.CreateBucketIfNotExists([]byte(contactBucket))
		return err
	})
	if err != nil {
		t.Fatalf("failed to create bucket: %v", err)
	}

	m := &ContactManager{db: db, path: dbPath}
	defer m.Close()

	// 2. Add Contact
	c := &Contact{
		Petname:   "@alice",
		KEMPubKey: []byte("kem-pub-data"),
		SIGPubKey: []byte("sig-pub-data"),
		AddedAt:   time.Now(),
		Notes:     "Testing contact",
	}

	if err := m.Add(c); err != nil {
		t.Fatalf("failed to add contact: %v", err)
	}

	// 3. Get Contact
	got, err := m.Get("@alice")
	if err != nil {
		t.Fatalf("failed to get contact: %v", err)
	}
	if got.Petname != c.Petname || !bytes.Equal(got.KEMPubKey, c.KEMPubKey) {
		t.Errorf("contact mismatch")
	}

	// Case-insensitivity check
	got2, err := m.Get("@ALICE")
	if err != nil || got2.Petname != "@alice" {
		t.Errorf("case-insensitive get failed")
	}

	// 4. List Contacts
	list, err := m.List()
	if err != nil || len(list) != 1 {
		t.Errorf("list failed: got %d contacts", len(list))
	}

	// 5. Delete Contact
	if err := m.Delete("@alice"); err != nil {
		t.Fatalf("delete failed: %v", err)
	}
	_, err = m.Get("@alice")
	if err == nil {
		t.Errorf("expected error getting deleted contact")
	}
}

func TestContactInvalidPetname(t *testing.T) {
	tmpDir := t.TempDir()
	dbPath := filepath.Join(tmpDir, "contacts_err.db")
	db, _ := bbolt.Open(dbPath, 0600, nil)
	m := &ContactManager{db: db, path: dbPath}
	defer m.Close()

	err := m.Add(&Contact{Petname: "no-at-sign"})
	if err == nil {
		t.Errorf("expected error for invalid petname (missing @)")
	}
}
