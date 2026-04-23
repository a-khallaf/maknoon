package crypto

import (
	"context"
	"encoding/hex"
	"fmt"
	"io"
	"path/filepath"
	"time"
)

// EngineEvent is the base interface for all telemetry emitted by the Engine.
type EngineEvent interface {
	String() string
}

// EventEncryptionStarted is emitted when the protection pipeline begins.
type EventEncryptionStarted struct {
	TotalBytes int64
}

func (e EventEncryptionStarted) String() string { return "encryption started" }

// EventDecryptionStarted is emitted when the unprotection pipeline begins.
type EventDecryptionStarted struct {
	TotalBytes int64
}

func (e EventDecryptionStarted) String() string { return "decryption started" }

// EventChunkProcessed is emitted when a data chunk has been successfully processed.
type EventChunkProcessed struct {
	BytesProcessed int64
	TotalProcessed int64
}

func (e EventChunkProcessed) String() string { return "chunk processed" }

// EventHandshakeComplete is emitted when the cryptographic handshake (KEM) finishes.
type EventHandshakeComplete struct{}

func (e EventHandshakeComplete) String() string { return "handshake complete" }

// EventEmitter defines a safe way to push telemetry events.
type EventEmitter interface {
	Emit(ev EngineEvent)
}

// EngineContext carries the execution state, telemetry stream, and policy for an operation.
type EngineContext struct {
	context.Context
	Events chan<- EngineEvent
	Policy SecurityPolicy
}

// NewEngineContext creates a new context with an optional event stream.
func NewEngineContext(ctx context.Context, events chan<- EngineEvent, policy SecurityPolicy) *EngineContext {
	if ctx == nil {
		ctx = context.Background()
	}
	return &EngineContext{
		Context: ctx,
		Events:  events,
		Policy:  policy,
	}
}

// Emit safely sends an event to the telemetry stream, preventing panics on closed channels.
func (c *EngineContext) Emit(ev EngineEvent) {
	if c == nil || c.Events == nil {
		return
	}
	defer func() { _ = recover() }()
	c.Events <- ev
}

// Protector handles encryption and decryption pipelines.
type Protector interface {
	Protect(inputName string, r io.Reader, w io.Writer, opts Options) (byte, error)
	Unprotect(r io.Reader, w io.Writer, outPath string, opts Options) (byte, error)
	LoadCustomProfile(path string) (*DynamicProfile, error)
	GenerateRandomProfile(id byte) *DynamicProfile
	ValidateProfile(p *DynamicProfile) error
}

// IdentityService handles identity lifecycle and discovery.
type IdentityService interface {
	IdentityActive() ([]string, error)
	IdentityInfo(name string) (string, error)
	IdentityRename(oldName, newName string) error
	IdentitySplit(name string, threshold, shares int, passphrase string) ([]string, error)
	IdentityCombine(mnemonics []string, output string, passphrase string, noPassword bool) (string, error)
	IdentityPublish(ctx context.Context, handle string, opts IdentityPublishOptions) error
	ContactAdd(petname, kemPub, sigPub, note string) error
	ContactList() ([]*Contact, error)
}

// VaultManager handles secure secret storage.
type VaultManager interface {
	VaultGet(vaultPath string, service string, passphrase []byte, pin string) (*VaultEntry, error)
	VaultSet(vaultPath string, entry *VaultEntry, passphrase []byte, pin string) error
	VaultRename(oldName, newName string) error
	VaultDelete(name string) error
	VaultList(vaultPath string) ([]string, error)
	VaultSplit(vaultPath string, threshold, shares int, passphrase string) ([]string, error)
	VaultRecover(mnemonics []string, vaultPath string, output string, passphrase string) (string, error)
}

// P2PService handles peer-to-peer transfers.
type P2PService interface {
	P2PSend(ctx context.Context, inputName string, r io.Reader, opts P2PSendOptions) (string, <-chan P2PStatus, error)
	P2PReceive(ctx context.Context, code string, opts P2PReceiveOptions) (<-chan P2PStatus, error)
	ValidateWormholeURL(u string) error
}

// Utils provides secure generation helpers.
type Utils interface {
	GeneratePassword(length int, noSymbols bool) (string, error)
	GeneratePassphrase(words int, separator string) (string, error)
}

// StateProvider exposes the engine's internal configuration and policy.
type StateProvider interface {
	GetPolicy() SecurityPolicy
	GetConfig() *Config
}

// MaknoonEngine is the unified facade for the Maknoon system, composing all specialized services.
type MaknoonEngine interface {
	Protector
	IdentityService
	VaultManager
	P2PService
	Utils
	StateProvider
}

// Engine is the central stateful service for Maknoon operations.
type Engine struct {
	Policy     SecurityPolicy
	Config     *Config
	Identities *IdentityManager
}

func (e *Engine) GetPolicy() SecurityPolicy { return e.Policy }
func (e *Engine) GetConfig() *Config        { return e.Config }

func (e *Engine) VaultList(vaultPath string) ([]string, error) {
	if vaultPath == "" {
		vaultPath = filepath.Join(e.Config.Paths.VaultsDir, "default.vault")
	}
	if err := e.Policy.ValidatePath(vaultPath); err != nil {
		return nil, err
	}
	return ListVaultEntries(vaultPath)
}

func (e *Engine) GeneratePassword(length int, noSymbols bool) (string, error) {
	return GeneratePassword(length, noSymbols)
}

func (e *Engine) GeneratePassphrase(words int, separator string) (string, error) {
	return GeneratePassphrase(words, separator)
}

func (e *Engine) IdentityActive() ([]string, error) {
	return e.Identities.ListActiveIdentities()
}

func (e *Engine) IdentityInfo(name string) (string, error) {
	return e.Identities.GetIdentityInfo(name)
}

func (e *Engine) IdentityRename(oldName, newName string) error {
	return e.Identities.RenameIdentity(oldName, newName)
}

func (e *Engine) IdentitySplit(name string, threshold, shares int, passphrase string) ([]string, error) {
	return e.Identities.SplitIdentity(name, threshold, shares, passphrase)
}

func (e *Engine) IdentityCombine(mnemonics []string, output string, passphrase string, noPassword bool) (string, error) {
	return e.Identities.CombineIdentity(mnemonics, output, passphrase, noPassword)
}

func (e *Engine) IdentityPublish(ctx context.Context, handle string, opts IdentityPublishOptions) error {
	return e.Identities.IdentityPublish(ctx, handle, opts)
}

func (e *Engine) ContactAdd(petname, kemPub, sigPub, note string) error {
	cm, err := NewContactManager()
	if err != nil {
		return err
	}
	defer cm.Close()

	// Parse keys from hex
	kp, _ := hex.DecodeString(kemPub)
	sp, _ := hex.DecodeString(sigPub)

	return cm.Add(&Contact{
		Petname:   petname,
		KEMPubKey: kp,
		SIGPubKey: sp,
		Notes:     note,
		AddedAt:   time.Now(),
	})
}

func (e *Engine) ContactList() ([]*Contact, error) {
	cm, err := NewContactManager()
	if err != nil {
		return nil, err
	}
	defer cm.Close()
	return cm.List()
}

func (e *Engine) VaultSplit(vaultPath string, threshold, shares int, passphrase string) ([]string, error) {
	if vaultPath == "" {
		vaultPath = filepath.Join(e.Config.Paths.VaultsDir, "default.vault")
	}
	return SplitVault(vaultPath, threshold, shares, passphrase)
}

func (e *Engine) VaultRecover(mnemonics []string, vaultPath string, output string, passphrase string) (string, error) {
	if vaultPath == "" {
		vaultPath = filepath.Join(e.Config.Paths.VaultsDir, "default.vault")
	}
	return RecoverVault(mnemonics, vaultPath, output, passphrase)
}

// NewEngine creates a new Engine with the specified policy and loaded config.
func NewEngine(policy SecurityPolicy) (*Engine, error) {
	conf, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize engine config: %w", err)
	}

	return &Engine{
		Policy:     policy,
		Config:     conf,
		Identities: NewIdentityManager(),
	}, nil
}

// ValidateWormholeURL enforces network boundaries.
func (e *Engine) ValidateWormholeURL(u string) error {
	return e.Policy.ValidateWormholeURL(u, e.Config.AgentLimits.AllowedURLs)
}

// ValidateProfile performs both technical sanity and policy-driven validation.
func (e *Engine) ValidateProfile(p *DynamicProfile) error {
	// 1. Technical Sanity
	if err := p.Validate(); err != nil {
		return err
	}

	// 2. Policy Enforcement
	return e.Policy.ValidateProfileResource(p.ArgonMem, p.ArgonTime, p.ArgonThrd, e.Config.AgentLimits)
}

// LoadCustomProfile reads, validates, and registers a profile under the active policy.
func (e *Engine) LoadCustomProfile(path string) (*DynamicProfile, error) {
	dp, err := LoadCustomProfile(path)
	if err != nil {
		return nil, err
	}

	if err := e.ValidateProfile(dp); err != nil {
		return nil, err
	}

	return dp, nil
}
