package crypto

import (
	"os"
	"path/filepath"
	"strings"
)

// KeyStore defines the interface for persisting and retrieving cryptographic keys.
type KeyStore interface {
	ReadKey(path string) ([]byte, error)
	WriteKey(path string, data []byte, perm uint32) error
	Exists(path string) bool
	ListKeys(dir string) ([]string, error)
	EnsureDir(dir string) error
	ResolvePath(name string) (string, error)
	GetBaseDir() string
}

// ConfigStore defines the interface for managing engine configuration.
type ConfigStore interface {
	Load() (*Config, error)
	Save(conf *Config) error
}

// VaultStore defines the interface for persisting secure vaults.
type VaultStore interface {
	ReadVault(path string) ([]byte, error)
	WriteVault(path string, data []byte) error
	DeleteVault(path string) error
	ListVaults() ([]string, error)
}

// FileSystemKeyStore is the default implementation that uses the local disk.
type FileSystemKeyStore struct {
	BaseDir string
}

func (s *FileSystemKeyStore) ReadKey(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (s *FileSystemKeyStore) WriteKey(path string, data []byte, perm uint32) error {
	return os.WriteFile(path, data, os.FileMode(perm))
}

func (s *FileSystemKeyStore) Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func (s *FileSystemKeyStore) ListKeys(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}
	var keys []string
	for _, e := range entries {
		keys = append(keys, e.Name())
	}
	return keys, nil
}

func (s *FileSystemKeyStore) EnsureDir(dir string) error {
	return os.MkdirAll(dir, 0700)
}

func (s *FileSystemKeyStore) ResolvePath(name string) (string, error) {
	if filepath.IsAbs(name) || strings.Contains(name, string(os.PathSeparator)) {
		return name, nil
	}
	return filepath.Join(s.BaseDir, name), nil
}

func (s *FileSystemKeyStore) GetBaseDir() string {
	return s.BaseDir
}

// FileSystemConfigStore manages engine configuration on disk.
type FileSystemConfigStore struct {
	Path string
}

func (s *FileSystemConfigStore) Load() (*Config, error) {
	return LoadConfig()
}

func (s *FileSystemConfigStore) Save(conf *Config) error {
	return conf.Save()
}

// FileSystemVaultStore manages secure vaults on disk.
type FileSystemVaultStore struct {
	BaseDir string
}

func (s *FileSystemVaultStore) ReadVault(path string) ([]byte, error) {
	return os.ReadFile(path)
}

func (s *FileSystemVaultStore) WriteVault(path string, data []byte) error {
	return os.WriteFile(path, data, 0600)
}

func (s *FileSystemVaultStore) DeleteVault(path string) error {
	return os.Remove(path)
}

func (s *FileSystemVaultStore) ListVaults() ([]string, error) {
	entries, err := os.ReadDir(s.BaseDir)
	if err != nil {
		if os.IsNotExist(err) {
			return []string{}, nil
		}
		return nil, err
	}
	var vaults []string
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".vault") {
			vaults = append(vaults, e.Name())
		}
	}
	return vaults, nil
}
