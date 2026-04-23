package crypto

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/awnumar/memguard"
)

const (
	// MaknoonDir is the default directory name for Maknoon data.
	MaknoonDir = ".maknoon"
	// KeysDir is the subdirectory for storing keys.
	KeysDir = "keys"
	// VaultsDir is the subdirectory for storing vaults.
	VaultsDir = "vaults"
	// ProfilesDir is the subdirectory for custom profiles.
	ProfilesDir = "profiles"
)

// IsAgentMode returns true if the application is running in non-interactive agent/JSON mode.
func IsAgentMode() bool {
	return os.Getenv("MAKNOON_AGENT_MODE") == "1" || os.Getenv("MAKNOON_JSON") == "1"
}

// Identity represents a full PQC keypair (KEM + SIG) + DHT metadata.
type Identity struct {
	Name      string
	KEMPub    []byte
	KEMPriv   []byte
	SIGPub    []byte
	SIGPriv   []byte
	NostrPub  []byte
	NostrPriv []byte
}

// IdentityManager handles local key storage and resolution.
type IdentityManager struct {
	KeysDir string
}

// NewIdentityManager creates an IdentityManager with default paths.
func NewIdentityManager() *IdentityManager {
	home := GetUserHomeDir()
	return &IdentityManager{
		KeysDir: filepath.Join(home, MaknoonDir, KeysDir),
	}
}

// ResolveKeyPath checks if a key exists locally, in ~/.maknoon/keys/, or in environment variables.
func ResolveKeyPath(path string, envVar string) string {
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	if envVar != "" {
		if env := os.Getenv(envVar); env != "" {
			if _, err := os.Stat(env); err == nil {
				return env
			}
		}
	}
	// Check default keys directory
	if path != "" {
		home := GetUserHomeDir()
		defaultPath := filepath.Join(home, MaknoonDir, KeysDir, path)
		if _, err := os.Stat(defaultPath); err == nil {
			return defaultPath
		}
	}
	return ""
}

// ResolveKeyPath is a convenience method on IdentityManager.
func (m *IdentityManager) ResolveKeyPath(path, envVar string) string {
	if path != "" {
		if _, err := os.Stat(path); err == nil {
			return path
		}
		// Check manager's KeysDir
		managedPath := filepath.Join(m.KeysDir, path)
		if _, err := os.Stat(managedPath); err == nil {
			return managedPath
		}
	}
	if envVar != "" {
		if env := os.Getenv(envVar); env != "" {
			if _, err := os.Stat(env); err == nil {
				return env
			}
		}
	}
	return ""
}

func (m *IdentityManager) ResolveBaseKeyPath(name string) (string, string, error) {
	if name == "" {
		return "", "", &ErrState{Reason: "identity name required"}
	}

	// 1. If it's an absolute path or contains path separators, use it directly
	if filepath.IsAbs(name) || strings.Contains(name, string(os.PathSeparator)) {
		base := strings.TrimSuffix(name, ".kem.key")
		base = strings.TrimSuffix(base, ".sig.key")
		base = strings.TrimSuffix(base, ".key")
		return base, filepath.Base(base), nil
	}

	// 2. Resolve via name in the managed KeysDir
	return filepath.Join(m.KeysDir, name), name, nil
}

// LoadIdentity handles the full flow of resolving and unlocking an identity.
func (m *IdentityManager) LoadIdentity(name string, passphrase []byte, pin string, isStdin bool) (*Identity, error) {
	if name == "" {
		name = GetGlobalConfig().DefaultIdentity
	}
	if name == "" {
		name = "default"
	}
	basePath, _, err := m.ResolveBaseKeyPath(name)

	if err != nil {
		return nil, err
	}

	id := &Identity{Name: name}

	// Load Public Keys
	id.KEMPub, _ = os.ReadFile(basePath + ".kem.pub")
	id.SIGPub, _ = os.ReadFile(basePath + ".sig.pub")
	id.NostrPub, _ = os.ReadFile(basePath + ".nostr.pub")

	// Load and Unlock KEM Private Key
	id.KEMPriv, err = m.LoadPrivateKey(basePath+".kem.key", passphrase, pin, isStdin)
	if err != nil {
		return nil, err
	}

	// Load and Unlock SIG Private Key
	id.SIGPriv, err = m.LoadPrivateKey(basePath+".sig.key", passphrase, pin, isStdin)
	if err != nil {
		id.Wipe()
		return nil, err
	}

	// Load and Unlock Nostr Private Key (Optional)
	nostrPath := basePath + ".nostr.key"
	if _, err := os.Stat(nostrPath); err == nil {
		id.NostrPriv, _ = m.LoadPrivateKey(nostrPath, passphrase, pin, isStdin)
	}

	return id, nil
}

// LoadPrivateKey resolves, reads, and unlocks a single private key.
func (m *IdentityManager) LoadPrivateKey(path string, passphrase []byte, pin string, isStdin bool) ([]byte, error) {
	resolvedPath := m.ResolveKeyPath(path, "")
	if _, err := os.Stat(resolvedPath); err != nil {
		return nil, &ErrState{Reason: fmt.Sprintf("private key not found: %s", path)}
	}

	keyBytes, err := os.ReadFile(resolvedPath)
	if err != nil {
		return nil, err
	}

	if len(keyBytes) > 4 && string(keyBytes[:4]) == MagicHeader {
		unlockedPass, err := m.UnlockPrivateKeyWithFIDOOrPass(passphrase, pin, resolvedPath, isStdin)
		if err != nil {
			return nil, err
		}
		// Decrypt the key stream
		var unlocked bytes.Buffer
		if _, _, err := DecryptStream(bytes.NewReader(keyBytes), &unlocked, unlockedPass, 1, false); err != nil {
			return nil, &ErrCrypto{Reason: fmt.Sprintf("failed to decrypt private key: %v", err)}
		}
		return unlocked.Bytes(), nil
	}

	return keyBytes, nil
}

// UnlockPrivateKeyWithFIDOOrPass handles the logic of getting the unlocking secret.
func (m *IdentityManager) UnlockPrivateKeyWithFIDOOrPass(password []byte, pin string, resolvedPath string, isStdin bool) ([]byte, error) {
	fido2Path := strings.TrimSuffix(resolvedPath, ".key")
	fido2Path = strings.TrimSuffix(fido2Path, ".kem")
	fido2Path = strings.TrimSuffix(fido2Path, ".sig")
	fido2Path += ".fido2"

	if _, err := os.Stat(fido2Path); err == nil {
		raw, err := os.ReadFile(fido2Path)
		if err != nil {
			return nil, fmt.Errorf("failed to read fido2 metadata: %w", err)
		}
		var meta Fido2Metadata
		if err := json.Unmarshal(raw, &meta); err != nil {
			return nil, fmt.Errorf("failed to unmarshal fido2 metadata: %w", err)
		}
		secret, err := Fido2Derive(meta.RPID, meta.CredentialID, pin)
		if err != nil {
			return nil, &ErrAuthentication{Reason: fmt.Sprintf("FIDO2 derivation failed: %v", err)}
		}
		return secret, nil
	}

	if len(password) == 0 {
		return nil, &ErrAuthentication{Reason: "passphrase required to unlock private key"}
	}
	return password, nil
}

// ResolvePublicKey handles handle resolution (@name) and local file paths.
func (m *IdentityManager) ResolvePublicKey(input string, tofu bool) ([]byte, error) {
	if strings.HasPrefix(input, "@") {
		// 1. Check local contacts (Petnames)
		cm, err := NewContactManager()
		if err == nil {
			contacts, _ := cm.List()
			var found []byte
			for _, c := range contacts {
				if c.Petname == input {
					found = c.KEMPubKey
					break
				}
			}
			cm.Close()
			if found != nil {
				return found, nil
			}
		}

		// 2. Check Global Discovery Registry
		reg := NewIdentityRegistry()
		record, err := reg.Resolve(context.Background(), input)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve identity handle: %w", err)
		}
		return record.KEMPubKey, nil
	}

	// 3. Fallback: Direct file path
	resolved := m.ResolveKeyPath(input, "")
	if resolved == "" {
		return nil, &ErrState{Reason: fmt.Sprintf("public key file not found: %s", input)}
	}
	return os.ReadFile(resolved)
}

// ListActiveIdentities returns a list of public key files in the KeysDir.
func (m *IdentityManager) ListActiveIdentities() ([]string, error) {
	files, err := os.ReadDir(m.KeysDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var identities []string
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		name := f.Name()
		if strings.HasSuffix(name, ".pub") {
			identities = append(identities, name)
		}
	}
	return identities, nil
}

// EnsureMaknoonDirs creates the default configuration and key directories.
func EnsureMaknoonDirs() error {
	home := GetUserHomeDir()
	base := filepath.Join(home, MaknoonDir)
	dirs := []string{
		base,
		filepath.Join(base, KeysDir),
		filepath.Join(base, VaultsDir),
		filepath.Join(base, ProfilesDir),
	}
	for _, dir := range dirs {
		if err := os.MkdirAll(dir, 0700); err != nil {
			return err
		}
	}
	return nil
}

func (m *IdentityManager) GetIdentityInfo(name string) (string, error) {
	basePath, _, err := m.ResolveBaseKeyPath(name)
	if err != nil {
		return "", err
	}
	// Check if at least the KEM key exists
	if _, err := os.Stat(basePath + ".kem.key"); err != nil {
		return "", fmt.Errorf("identity '%s' not found", name)
	}
	return basePath, nil
}

func (m *IdentityManager) RenameIdentity(oldName, newName string) error {
	oldBase, _, err := m.ResolveBaseKeyPath(oldName)
	if err != nil {
		return err
	}
	newBase, _, err := m.ResolveBaseKeyPath(newName)
	if err != nil {
		return err
	}

	suffixes := []string{".kem.key", ".kem.pub", ".sig.key", ".sig.pub", ".fido2", ".nostr.key", ".nostr.pub"}
	renamed := 0
	for _, s := range suffixes {
		oldPath := oldBase + s
		newPath := newBase + s
		if _, err := os.Stat(oldPath); err == nil {
			if err := os.Rename(oldPath, newPath); err != nil {
				return err
			}
			renamed++
		}
	}

	if renamed == 0 {
		return fmt.Errorf("identity '%s' not found", oldName)
	}

	// Rename in Contacts too
	cm, err := NewContactManager()
	if err == nil {
		contact, err := cm.Get(oldName)
		if err == nil {
			contact.Petname = newName
			cm.Add(contact)
			cm.Delete(oldName)
		}
		cm.Close()
	}

	return nil
}

func (m *IdentityManager) SplitIdentity(name string, threshold, shares int, passphrase string) ([]string, error) {
	// 1. Load the identity
	// For simplicity in the library, we assume interaction-less PIN if fido2 exists
	id, err := m.LoadIdentity(name, []byte(passphrase), "", false)
	if err != nil {
		return nil, err
	}
	defer id.Wipe()

	// 2. Pack the keys
	blob := make([]byte, 12+len(id.KEMPriv)+len(id.SIGPriv)+len(id.NostrPriv))
	offset := 0
	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.KEMPriv)))
	copy(blob[offset+4:offset+4+len(id.KEMPriv)], id.KEMPriv)
	offset += 4 + len(id.KEMPriv)

	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.SIGPriv)))
	copy(blob[offset+4:offset+4+len(id.SIGPriv)], id.SIGPriv)
	offset += 4 + len(id.SIGPriv)

	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.NostrPriv)))
	copy(blob[offset+4:], id.NostrPriv)

	defer SafeClear(blob)

	// 3. Split
	shards, err := SplitSecret(blob, threshold, shares)
	if err != nil {
		return nil, err
	}

	var results []string
	for _, s := range shards {
		results = append(results, s.ToMnemonic())
	}
	return results, nil
}

func (m *IdentityManager) CombineIdentity(mnemonics []string, output, passphrase string, noPassword bool) (string, error) {
	var shards []Share
	for _, mn := range mnemonics {
		s, err := FromMnemonic(mn)
		if err != nil {
			return "", fmt.Errorf("invalid mnemonic: %w", err)
		}
		shards = append(shards, *s)
	}

	blob, err := CombineShares(shards)
	if err != nil {
		return "", err
	}
	defer SafeClear(blob)

	if len(blob) < 12 {
		return "", fmt.Errorf("reconstructed blob too short")
	}

	// Unpack
	kemLen := binary.BigEndian.Uint32(blob[0:4])
	kemPriv := blob[4 : 4+kemLen]
	offset := 4 + kemLen

	sigLen := binary.BigEndian.Uint32(blob[offset : offset+4])
	sigPriv := blob[offset+4 : offset+4+sigLen]
	offset += 4 + sigLen

	var nostrPriv []byte
	if uint32(len(blob)) >= offset+4 {
		nostrLen := binary.BigEndian.Uint32(blob[offset : offset+4])
		if uint32(len(blob)) == offset+4+nostrLen {
			nostrPriv = blob[offset+4:]
		}
	}

	// Derive public keys
	kemPub, _ := DeriveKEMPublic(kemPriv)
	sigPub, _ := DeriveSIGPublic(sigPriv)
	var nostrPub []byte
	if len(nostrPriv) > 0 {
		nostrPub, _ = DeriveNostrPublic(nostrPriv)
	}

	basePath, _, err := m.ResolveBaseKeyPath(output)
	if err != nil {
		return "", err
	}

	// Write keys
	err = writeIdentityKeys(basePath, output, kemPub, kemPriv, sigPub, sigPriv, nostrPub, nostrPriv, []byte(passphrase), 1)
	return basePath, err
}

func writeIdentityKeys(basePath, baseName string, kemPub, kemPriv, sigPub, sigPriv, nostrPub, nostrPriv, password []byte, profileID byte) error {
	writeKey := func(path string, data []byte, isPrivate bool) error {
		if len(data) == 0 {
			return nil
		}
		finalData := data
		if isPrivate && len(password) > 0 {
			var b bytes.Buffer
			if err := EncryptStream(bytes.NewReader(data), &b, password, FlagNone, 1, profileID); err != nil {
				return err
			}
			finalData = b.Bytes()
		}
		mode := os.FileMode(0644)
		if isPrivate {
			mode = 0600
		}
		return os.WriteFile(path, finalData, mode)
	}

	if err := writeKey(basePath+".kem.key", kemPriv, true); err != nil {
		return err
	}
	if err := writeKey(basePath+".kem.pub", kemPub, false); err != nil {
		return err
	}
	if err := writeKey(basePath+".sig.key", sigPriv, true); err != nil {
		return err
	}
	if err := writeKey(basePath+".sig.pub", sigPub, false); err != nil {
		return err
	}
	if err := writeKey(basePath+".nostr.key", nostrPriv, true); err != nil {
		return err
	}
	if err := writeKey(basePath+".nostr.pub", nostrPub, false); err != nil {
		return err
	}
	return nil
}

type IdentityPublishOptions struct {
	Passphrase string
	Name       string
	Nostr      bool
	DNS        bool
	Local      bool
	Desec      bool
	DesecToken string
}

func (m *IdentityManager) IdentityPublish(ctx context.Context, handle string, opts IdentityPublishOptions) error {
	if !strings.HasPrefix(handle, "@") {
		return fmt.Errorf("handle must start with @")
	}

	name := "default"
	if opts.Name != "" {
		name = opts.Name
	}

	// 1. Get active identity
	basePath, _, _ := m.ResolveBaseKeyPath(name)
	var pin string
	if _, err := os.Stat(basePath + ".fido2"); err == nil {
		// PIN might be required, but library-first assumes non-interactive for now
	}

	id, err := m.LoadIdentity(name, []byte(opts.Passphrase), pin, false)
	if err != nil {
		return err
	}
	defer id.Wipe()

	// 2. Create and sign record
	record := &IdentityRecord{
		Handle:    handle,
		KEMPubKey: id.KEMPub,
		SIGPubKey: id.SIGPub,
		Timestamp: time.Now(),
	}

	if err := record.Sign(id.SIGPriv); err != nil {
		return fmt.Errorf("failed to sign identity record: %w", err)
	}

	// 3. Dispatch to registries
	if opts.Local {
		cm, err := NewContactManager()
		if err != nil {
			return err
		}
		defer cm.Close()

		return cm.Add(&Contact{
			Petname:   handle,
			KEMPubKey: record.KEMPubKey,
			SIGPubKey: record.SIGPubKey,
			AddedAt:   time.Now(),
		})
	}

	if opts.Desec {
		token := opts.DesecToken
		if token == "" {
			token = os.Getenv("DESEC_TOKEN")
		}
		if token == "" {
			return fmt.Errorf("deSEC token required")
		}

		dnsReg := NewDNSRegistry()
		return dnsReg.PublishWithKey(ctx, record, []byte(token))
	}

	// Default to Nostr
	if opts.Nostr || (!opts.DNS && !opts.Desec) {
		nostrReg := NewNostrRegistry()
		if len(id.NostrPriv) == 0 {
			return fmt.Errorf("nostr private key not found")
		}
		return nostrReg.PublishWithKey(ctx, record, id.NostrPriv)
	}

	return nil
}

func (m *IdentityManager) IdentitySplit(name string, threshold, shares int, passphrase string) ([]string, error) {
	// 1. Load the identity
	id, err := m.LoadIdentity(name, []byte(passphrase), "", false)
	if err != nil {
		return nil, err
	}
	defer id.Wipe()

	// 2. Pack the keys
	blob := make([]byte, 12+len(id.KEMPriv)+len(id.SIGPriv)+len(id.NostrPriv))
	offset := 0
	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.KEMPriv)))
	copy(blob[offset+4:offset+4+len(id.KEMPriv)], id.KEMPriv)
	offset += 4 + len(id.KEMPriv)

	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.SIGPriv)))
	copy(blob[offset+4:offset+4+len(id.SIGPriv)], id.SIGPriv)
	offset += 4 + len(id.SIGPriv)

	binary.BigEndian.PutUint32(blob[offset:offset+4], uint32(len(id.NostrPriv)))
	copy(blob[offset+4:], id.NostrPriv)

	defer SafeClear(blob)

	// 3. Split
	shards, err := SplitSecret(blob, threshold, shares)
	if err != nil {
		return nil, err
	}

	var results []string
	for _, s := range shards {
		results = append(results, s.ToMnemonic())
	}
	return results, nil
}

func (id *Identity) Wipe() {
	SafeClear(id.KEMPriv)
	SafeClear(id.SIGPriv)
	SafeClear(id.NostrPriv)
}

func SafeClear(b []byte) {
	if b != nil {
		memguard.WipeBytes(b)
	}
}
