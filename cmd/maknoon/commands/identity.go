package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/spf13/cobra"
)

// IdentityCmd returns the cobra command for managing cryptographic identities.
func IdentityCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "identity",
		Short: "Manage Post-Quantum cryptographic identities",
	}

	cmd.AddCommand(identityListCmd())
	cmd.AddCommand(identityActiveCmd())
	cmd.AddCommand(identityShowCmd())
	cmd.AddCommand(identityRenameCmd())

	return cmd
}

func identityActiveCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "active",
		Short: "List absolute paths of available public keys for encryption",
		RunE: func(_ *cobra.Command, _ []string) error {
			home, _ := os.UserHomeDir()
			keysDir := filepath.Join(home, crypto.MaknoonDir, crypto.KeysDir)

			files, err := os.ReadDir(keysDir)
			if err != nil {
				if os.IsNotExist(err) {
					if JSONOutput {
						printJSON(map[string]interface{}{"active_keys": []string{}})
						return nil
					}
					fmt.Println("No identities found.")
					return nil
				}
				return err
			}

			var keys []string
			for _, f := range files {
				name := f.Name()
				if strings.HasSuffix(name, ".kem.pub") {
					keys = append(keys, filepath.Join(keysDir, name))
				}
			}

			if JSONOutput {
				printJSON(map[string]interface{}{
					"active_keys": keys,
				})
			} else {
				fmt.Println("🛡️  Active Public Keys (Absolute Paths):")
				for _, k := range keys {
					fmt.Printf("  - %s\n", k)
				}
			}
			return nil
		},
	}
	return cmd
}

func identityListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all local identities",
		RunE: func(_ *cobra.Command, _ []string) error {
			home, _ := os.UserHomeDir()
			keysDir := filepath.Join(home, crypto.MaknoonDir, crypto.KeysDir)

			files, err := os.ReadDir(keysDir)
			if err != nil {
				if os.IsNotExist(err) {
					if JSONOutput {
						printJSON([]string{})
						return nil
					}
					fmt.Println("No identities found.")
					return nil
				}
				return err
			}

			var identities []string
			seen := make(map[string]bool)
			for _, f := range files {
				name := f.Name()
				if strings.HasSuffix(name, ".kem.pub") {
					base := strings.TrimSuffix(name, ".kem.pub")
					if !seen[base] {
						identities = append(identities, base)
						seen[base] = true
					}
				}
			}

			if JSONOutput {
				printJSON(identities)
			} else {
				fmt.Println("🛡️  Maknoon Identities:")
				for _, id := range identities {
					fmt.Printf("  - %s\n", id)
				}
			}
			return nil
		},
	}
	return cmd
}

func identityShowCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "show [name]",
		Short: "Show details for a specific identity",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			name := args[0]
			home, _ := os.UserHomeDir()
			keysDir := filepath.Join(home, crypto.MaknoonDir, crypto.KeysDir)

			var basePath string
			if strings.Contains(name, string(os.PathSeparator)) {
				if JSONOutput {
					absPath, _ := filepath.Abs(name)
					evalPath, _ := filepath.EvalSymlinks(absPath)
					evalHome, _ := filepath.EvalSymlinks(home)
					if !strings.HasPrefix(evalPath, evalHome) {
						return fmt.Errorf("security policy: arbitrary key paths outside home are prohibited in JSON mode")
					}
				}
				basePath = name
			} else {
				bn := filepath.Base(name)
				if bn == ".." || bn == "." || bn == "/" {
					return fmt.Errorf("invalid identity name")
				}
				basePath = filepath.Join(keysDir, bn)
			}

			pubKeyPath := basePath + ".kem.pub"
			if _, err := os.Stat(pubKeyPath); err != nil {
				return fmt.Errorf("identity '%s' not found", name)
			}

			// Check for hardware binding
			hasFido2 := false
			if _, err := os.Stat(basePath + ".fido2"); err == nil {
				hasFido2 = true
			}

			if JSONOutput {
				printJSON(map[string]interface{}{
					"name":     name,
					"path":     basePath,
					"hardware": hasFido2,
				})
			} else {
				fmt.Printf("Identity: %s\n", name)
				fmt.Printf("Path:     %s\n", basePath)
				fmt.Printf("Hardware: %v\n", hasFido2)
			}
			return nil
		},
	}
	return cmd
}

func identityRenameCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "rename [old_name] [new_name]",
		Short: "Rename an identity",
		Args:  cobra.ExactArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			oldName, newName := args[0], args[1]
			home, _ := os.UserHomeDir()
			keysDir := filepath.Join(home, crypto.MaknoonDir, crypto.KeysDir)

			var oldBase, newBase string
			if strings.Contains(oldName, string(os.PathSeparator)) {
				if JSONOutput {
					absPath, _ := filepath.Abs(oldName)
					evalPath, _ := filepath.EvalSymlinks(absPath)
					evalHome, _ := filepath.EvalSymlinks(home)
					if !strings.HasPrefix(evalPath, evalHome) {
						return fmt.Errorf("security policy: arbitrary key paths outside home are prohibited in JSON mode")
					}
				}
				oldBase = oldName
			} else {
				ob := filepath.Base(oldName)
				if ob == ".." || ob == "." || ob == "/" {
					return fmt.Errorf("invalid old identity name")
				}
				oldBase = filepath.Join(keysDir, ob)
			}

			if strings.Contains(newName, string(os.PathSeparator)) {
				if JSONOutput {
					absPath, _ := filepath.Abs(newName)
					evalPath, _ := filepath.EvalSymlinks(absPath)
					evalHome, _ := filepath.EvalSymlinks(home)
					if !strings.HasPrefix(evalPath, evalHome) {
						return fmt.Errorf("security policy: arbitrary key paths outside home are prohibited in JSON mode")
					}
				}
				newBase = newName
			} else {
				nb := filepath.Base(newName)
				if nb == ".." || nb == "." || nb == "/" {
					return fmt.Errorf("invalid new identity name")
				}
				newBase = filepath.Join(keysDir, nb)
			}

			suffixes := []string{".kem.key", ".kem.pub", ".sig.key", ".sig.pub", ".fido2"}
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

			if JSONOutput {
				printJSON(map[string]string{"status": "success", "from": oldName, "to": newName})
			} else {
				fmt.Printf("Successfully renamed identity '%s' to '%s'\n", oldName, newName)
			}
			return nil
		},
	}
	return cmd
}
