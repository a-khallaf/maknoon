package commands

import (
	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
)

// resolveKeyPath checks if a key exists locally, in ~/.maknoon/keys/, or in environment variables.
func resolveKeyPath(path string, envVar string) string {
	return crypto.ResolveKeyPath(path, envVar)
}
