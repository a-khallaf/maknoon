package commands

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
)

// JSONOutput triggers JSON-formatted output and suppresses all interactive prompts.
var JSONOutput bool

// printJSON outputs an interface as a JSON string to stdout.
func printJSON(v interface{}) {
	raw, _ := json.Marshal(v)
	fmt.Println(string(raw))
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
