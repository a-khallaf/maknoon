package commands

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/spf13/cobra"
	"go.etcd.io/bbolt"
	"golang.org/x/term"
)

const (
	vaultBucket = "secrets"
	metaBucket  = "metadata"
	saltKey     = "salt"
	fido2Key    = "fido2"
)

var vaultName string
var vaultPassphrase string
var useFido2 bool

// JSONOutput triggers JSON-formatted output and suppresses all interactive prompts.
var JSONOutput bool

// VaultCmd returns the cobra command for managing secure vaults.
func VaultCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "vault",
		Short: "Manage secure password vaults",
	}

	cmd.PersistentFlags().StringVarP(&vaultName, "vault", "v", "default", "Name or full path of the vault to use")
	cmd.PersistentFlags().StringVarP(&vaultPassphrase, "passphrase", "s", "", "Master passphrase for the vault")
	cmd.PersistentFlags().BoolVarP(&useFido2, "fido2", "f", false, "Use FIDO2 security key for authentication")
	cmd.PersistentFlags().BoolVar(&JSONOutput, "json", false, "Output results in JSON format")

	cmd.AddCommand(vaultSetCmd())
	cmd.AddCommand(vaultGetCmd())
	cmd.AddCommand(vaultListCmd())
	cmd.AddCommand(vaultRenameCmd())
	cmd.AddCommand(vaultDeleteCmd())

	return cmd
}

func checkJSONMode(cmd *cobra.Command) {
	if JSONOutput || os.Getenv("MAKNOON_JSON") == "1" {
		JSONOutput = true
		if cmd != nil {
			cmd.SilenceUsage = true
			cmd.SilenceErrors = true
		}
	}
}

func printJSON(v interface{}) {
	raw, _ := json.Marshal(v)
	fmt.Println(string(raw))
}

func printErrorJSON(err error) {
	raw, _ := json.Marshal(map[string]string{"error": err.Error()})
	fmt.Fprintln(os.Stderr, string(raw))
}

func resolveVaultPath(name string) (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("failed to get home directory: %w", err)
	}
	defaultDir := filepath.Join(home, crypto.MaknoonDir, crypto.VaultsDir)

	if strings.Contains(name, string(os.PathSeparator)) {
		if JSONOutput {
			absPath, _ := filepath.Abs(name)
			if !strings.HasPrefix(absPath, defaultDir) {
				return "", fmt.Errorf("security policy: arbitrary vault paths are prohibited in JSON mode")
			}
		}
		return name, nil
	}
	return filepath.Join(defaultDir, name+".db"), nil
}

func openVault() (*bbolt.DB, []byte, error) {
	checkJSONMode(nil)
	if err := crypto.EnsureMaknoonDirs(); err != nil {
		return nil, nil, err
	}

	dbPath, err := resolveVaultPath(vaultName)
	if err != nil {
		return nil, nil, err
	}

	db, err := bbolt.Open(dbPath, 0600, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open vault database: %w", err)
	}

	var salt []byte
	var fido2Raw []byte
	var fido2Secret []byte

	err = db.Update(func(tx *bbolt.Tx) error {
		b, err := tx.CreateBucketIfNotExists([]byte(metaBucket))
		if err != nil {
			return err
		}
		if _, err := tx.CreateBucketIfNotExists([]byte(vaultBucket)); err != nil {
			return err
		}
		salt = b.Get([]byte(saltKey))
		if salt == nil {
			salt = make([]byte, 32)
			if _, err := rand.Read(salt); err != nil {
				return err
			}
			if err := b.Put([]byte(saltKey), salt); err != nil {
				return err
			}
		}
		fido2Raw = b.Get([]byte(fido2Key))

		if useFido2 && fido2Raw == nil {
			meta, secret, err := crypto.Fido2Enroll("maknoon.io", "vault-user")
			if err != nil {
				return err
			}
			raw, err := json.Marshal(meta)
			if err != nil {
				return err
			}
			if err := b.Put([]byte(fido2Key), raw); err != nil {
				return err
			}
			fido2Raw = raw
			fido2Secret = secret
		}
		return nil
	})
	if err != nil {
		_ = db.Close()
		return nil, nil, err
	}

	var passphrase []byte
	if len(fido2Secret) > 0 {
		passphrase = fido2Secret
	} else if fido2Raw != nil {
		var meta crypto.Fido2Metadata
		if err := json.Unmarshal(fido2Raw, &meta); err != nil {
			_ = db.Close()
			return nil, nil, err
		}
		secret, err := crypto.Fido2Derive(meta.RPID, meta.CredentialID)
		if err != nil {
			_ = db.Close()
			return nil, nil, err
		}
		passphrase = secret
	} else if vaultPassphrase != "" {
		passphrase = []byte(vaultPassphrase)
	} else if env := os.Getenv("MAKNOON_PASSPHRASE"); env != "" {
		passphrase = []byte(env)
	} else if JSONOutput {
		_ = db.Close()
		return nil, nil, fmt.Errorf("master passphrase required via MAKNOON_PASSPHRASE or -s")
	} else {
		fmt.Print("Enter Vault Master Passphrase: ")
		p, err := term.ReadPassword(int(os.Stdin.Fd()))
		fmt.Println()
		if err != nil {
			_ = db.Close()
			return nil, nil, err
		}
		passphrase = p
	}

	masterKey := crypto.DeriveVaultKey(passphrase, salt)
	crypto.SafeClear(passphrase)

	return db, masterKey, nil
}

func vaultSetCmd() *cobra.Command {
	var user, note string
	cmd := &cobra.Command{
		Use:   "set [service]",
		Short: "Store a secret in the vault",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			service := args[0]
			var password []byte
			var err error

			if env := os.Getenv("MAKNOON_PASSWORD"); env != "" {
				password = []byte(env)
			} else if JSONOutput {
				err := fmt.Errorf("password required via MAKNOON_PASSWORD environment variable in JSON mode")
				printErrorJSON(err)
				return nil
			} else {
				fmt.Print("Enter password for ", service, ": ")
				password, err = term.ReadPassword(int(os.Stdin.Fd()))
				fmt.Println()
				if err != nil {
					return err
				}
			}
			defer crypto.SafeClear(password)

			db, key, err := openVault()
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			defer func() { _ = db.Close() }()
			defer crypto.SafeClear(key)

			entry := &crypto.VaultEntry{Service: service, Username: user, Password: password, Note: note}
			ciphertext, err := crypto.SealEntry(entry, key)
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			h := sha256.Sum256([]byte(strings.ToLower(service)))
			err = db.Update(func(tx *bbolt.Tx) error {
				return tx.Bucket([]byte(vaultBucket)).Put([]byte(hex.EncodeToString(h[:])), ciphertext)
			})
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			if JSONOutput {
				printJSON(map[string]string{"status": "success", "service": service})
			}
			return nil
		},
	}
	cmd.Flags().StringVarP(&user, "user", "u", "", "Username")
	cmd.Flags().StringVarP(&note, "note", "n", "", "Optional note")
	return cmd
}

func vaultGetCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "get [service]",
		Short: "Retrieve a secret from the vault",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			checkJSONMode(cmd)
			service := args[0]
			db, key, err := openVault()
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			defer func() { _ = db.Close() }()
			defer crypto.SafeClear(key)

			h := sha256.Sum256([]byte(strings.ToLower(service)))
			var ciphertext []byte
			err = db.View(func(tx *bbolt.Tx) error {
				b := tx.Bucket([]byte(vaultBucket))
				if b == nil {
					return fmt.Errorf("vault bucket not found")
				}
				ciphertext = b.Get([]byte(hex.EncodeToString(h[:])))
				return nil
			})
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			if ciphertext == nil {
				err := fmt.Errorf("service not found")
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			entry, err := crypto.OpenEntry(ciphertext, key)
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			defer crypto.SafeClear(entry.Password)

			if JSONOutput {
				type jsonEntry struct {
					Service  string `json:"service"`
					Username string `json:"username"`
					Password string `json:"password"`
					Note     string `json:"note"`
				}
				printJSON(jsonEntry{
					Service:  entry.Service,
					Username: entry.Username,
					Password: string(entry.Password),
					Note:     entry.Note,
				})
			} else {
				fmt.Printf("Service:  %s\nUsername: %s\nPassword: %s\n", entry.Service, entry.Username, string(entry.Password))
			}
			return nil
		},
	}
	return cmd
}

func vaultListCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "list",
		Short: "List all services stored in the vault",
		RunE: func(_ *cobra.Command, _ []string) error {
			db, key, err := openVault()
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			defer func() { _ = db.Close() }()
			defer crypto.SafeClear(key)

			var services []string
			err = db.View(func(tx *bbolt.Tx) error {
				b := tx.Bucket([]byte(vaultBucket))
				if b == nil {
					return nil
				}
				return b.ForEach(func(_ []byte, v []byte) error {
					entry, err := crypto.OpenEntry(v, key)
					if err == nil {
						defer crypto.SafeClear(entry.Password)
						if JSONOutput {
							services = append(services, entry.Service)
						} else {
							fmt.Printf(" - %s (%s)\n", entry.Service, entry.Username)
						}
					}
					return nil
				})
			})

			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			if JSONOutput {
				printJSON(services)
			}
			return nil
		},
	}
	return cmd
}

func vaultRenameCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "rename [old_name] [new_name]",
		Short: "Rename a local vault file",
		Args:  cobra.ExactArgs(2),
		RunE: func(_ *cobra.Command, args []string) error {
			oldPath, err := resolveVaultPath(args[0])
			if err != nil {
				return err
			}
			newPath, err := resolveVaultPath(args[1])
			if err != nil {
				return err
			}

			if _, err := os.Stat(oldPath); err != nil {
				return fmt.Errorf("vault '%s' not found", args[0])
			}
			if _, err := os.Stat(newPath); err == nil {
				return fmt.Errorf("target vault '%s' already exists", args[1])
			}

			if err := os.Rename(oldPath, newPath); err != nil {
				return err
			}

			if JSONOutput {
				printJSON(map[string]string{"status": "success", "from": args[0], "to": args[1]})
			} else {
				fmt.Printf("Vault '%s' renamed to '%s'\n", args[0], args[1])
			}
			return nil
		},
	}
	return cmd
}

func vaultDeleteCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "delete [name]",
		Short: "Permanently delete a vault file",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			path, err := resolveVaultPath(args[0])
			if err != nil {
				return err
			}

			if _, err := os.Stat(path); err != nil {
				return fmt.Errorf("vault '%s' not found", args[0])
			}

			if !JSONOutput {
				fmt.Printf("ARE YOU SURE you want to delete vault '%s'? This cannot be undone. (y/N): ", args[0])
				var confirm string
				fmt.Scanln(&confirm)
				if strings.ToLower(confirm) != "y" {
					return fmt.Errorf("deletion cancelled")
				}
			}

			if err := os.Remove(path); err != nil {
				return err
			}

			if JSONOutput {
				printJSON(map[string]string{"status": "success", "deleted": args[0]})
			} else {
				fmt.Printf("Vault '%s' deleted successfully.\n", args[0])
			}
			return nil
		},
	}
	return cmd
}
