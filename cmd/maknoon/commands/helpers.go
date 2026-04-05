package commands

import (
	"os"
	"path/filepath"
)

// resolveKeyPath checks if a key exists locally, and if not, looks in ~/.maknoon/keys/
func resolveKeyPath(path string) string {
	if _, err := os.Stat(path); err == nil {
		return path
	}
	// Check in ~/.maknoon/keys/
	home, _ := os.UserHomeDir()
	maknoonPath := filepath.Join(home, ".maknoon", "keys", path)
	if _, err := os.Stat(maknoonPath); err == nil {
		return maknoonPath
	}
	return path // Fallback to original
}
