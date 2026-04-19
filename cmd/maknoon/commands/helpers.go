package commands

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
)

// JSONOutput triggers JSON-formatted output and suppresses all interactive prompts.
var JSONOutput bool

// JSONWriter is where printJSON sends its output.
var JSONWriter io.Writer = os.Stdout

// printJSON outputs an interface as a JSON string to JSONWriter.
func printJSON(v interface{}) {
	raw, _ := json.Marshal(v)
	fmt.Fprintln(JSONWriter, string(raw))
}

// printErrorJSON outputs an error as a JSON object to stderr.
func printErrorJSON(err error) {
	raw, _ := json.Marshal(map[string]string{"error": err.Error()})
	fmt.Fprintln(os.Stderr, string(raw))
}

// resolveKeyPath checks if a key exists locally, in ~/.maknoon/keys/, or in environment variables.
func resolveKeyPath(path string, envVar string) string {
	return crypto.ResolveKeyPath(path, envVar)
}

// validatePath ensures a path is safe to use.
// In JSON mode, it restricts all file operations to the user's home directory.
func validatePath(path string) error {
	if path == "-" || path == "" {
		return nil
	}

	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("invalid path: %w", err)
	}

	// Always resolve symlinks for final validation.
	// If the file doesn't exist, resolve symlinks of the parent directory.
	evalPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		parentEval, err2 := filepath.EvalSymlinks(filepath.Dir(absPath))
		if err2 == nil {
			evalPath = filepath.Join(parentEval, filepath.Base(absPath))
		} else {
			evalPath = absPath
		}
	}

	if JSONOutput {
		home, err := os.UserHomeDir()
		if err != nil {
			return fmt.Errorf("failed to get home directory: %w", err)
		}
		evalHome, _ := filepath.EvalSymlinks(home)

		tmp := os.TempDir()
		evalTmp, _ := filepath.EvalSymlinks(tmp)

		// Ensure the path is within the home directory or system temp directory
		if !strings.HasPrefix(evalPath, evalHome) && !strings.HasPrefix(evalPath, evalTmp) {
			return fmt.Errorf("security policy: arbitrary file paths outside home or temp are prohibited in JSON mode")
		}

		// Additional check for path traversal even within home if it contains ".."
		// filepath.Abs already resolves ".." in the absolute path representation.
	}

	return nil
}
