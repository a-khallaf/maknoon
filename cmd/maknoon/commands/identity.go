package commands

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/spf13/cobra"
)

// IdentityCmd returns the cobra command for managing cryptographic identities.
func IdentityCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "identity",
		Short: "Manage Post-Quantum cryptographic identities",
	}

	cmd.PersistentFlags().BoolVar(&JSONOutput, "json", false, "Output results in JSON format")

	cmd.AddCommand(identityListCmd())
	cmd.AddCommand(identityActiveCmd())
	cmd.AddCommand(identityShowCmd())
	cmd.AddCommand(identityRenameCmd())
	cmd.AddCommand(identitySplitCmd())
	cmd.AddCommand(identityCombineCmd())
	cmd.AddCommand(identityPublishCmd())

	return cmd
}

func identityPublishCmd() *cobra.Command {
	var passphrase string
	cmd := &cobra.Command{
		Use:   "publish [handle]",
		Short: "Anchor your active identity to the global registry (dPKI)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			checkJSONMode(cmd)
			handle := args[0]
			if !strings.HasPrefix(handle, "@") {
				return fmt.Errorf("handle must start with @ (e.g., @alice)")
			}

			// 1. Get active identity
			name := "default" // Simplified for POC

			m := crypto.NewIdentityManager()
			if _, _, err := m.ResolveBaseKeyPath(name); err != nil {
				return err
			}

			id, err := m.LoadIdentity(name, []byte(passphrase), false)
			if err != nil {
				return err
			}
			defer id.Wipe()

			// 3. Create and sign the record
			record := &crypto.IdentityRecord{
				Handle:    handle,
				KEMPubKey: id.KEMPub,
				SIGPubKey: id.SIGPub,
				Timestamp: time.Now(),
			}

			if err := record.Sign(id.SIGPriv); err != nil {
				return fmt.Errorf("failed to sign identity record: %w", err)
			}

			// 4. Publish to Registry
			if err := crypto.GlobalRegistry.Publish(context.Background(), record); err != nil {
				return err
			}

			if JSONOutput {
				printJSON(crypto.IdentityResult{
					Status:   "success",
					Handle:   handle,
					Registry: "bolt",
				})
			} else {
				fmt.Printf("🚀 Identity published to Global Registry as %s\n", handle)
				fmt.Println("Note: This is currently using a local persistent bbolt database.")
			}

			return nil
		},
	}
	cmd.Flags().StringVarP(&passphrase, "passphrase", "s", "", "Passphrase to unlock your signing key")
	return cmd
}

func identitySplitCmd() *cobra.Command {
	var threshold, shares int
	var passphrase string

	cmd := &cobra.Command{
		Use:   "split [name]",
		Short: "Shard a private identity using Shamir's Secret Sharing",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			checkJSONMode(cmd)
			name := args[0]
			m := crypto.NewIdentityManager()
			id, err := m.LoadIdentity(name, []byte(passphrase), false)
			if err != nil {
				return err
			}
			defer id.Wipe()

			// Combine keys into a single blob: [len_kem_4_bytes][kem_priv][len_sig_4_bytes][sig_priv]
			blob := make([]byte, 8+len(id.KEMPriv)+len(id.SIGPriv))
			binary.BigEndian.PutUint32(blob[0:4], uint32(len(id.KEMPriv)))
			copy(blob[4:4+len(id.KEMPriv)], id.KEMPriv)
			binary.BigEndian.PutUint32(blob[4+len(id.KEMPriv):8+len(id.KEMPriv)], uint32(len(id.SIGPriv)))
			copy(blob[8+len(id.KEMPriv):], id.SIGPriv)
			defer crypto.SafeClear(blob)

			shards, err := crypto.SplitSecret(blob, threshold, shares)
			if err != nil {
				return err
			}

			if JSONOutput {
				var jsonShards []string
				for _, s := range shards {
					jsonShards = append(jsonShards, s.ToMnemonic())
				}
				printJSON(crypto.IdentityResult{
					Status:    "success",
					Identity:  name,
					Threshold: threshold,
					Shares:    jsonShards,
				})
			} else {
				fmt.Printf("🛡️  Identity '%s' sharded into %d parts (Threshold: %d)\n", name, shares, threshold)
				fmt.Println("CRITICAL: Keep these mnemonics safe and separated.")
				for i, s := range shards {
					fmt.Printf("\nShare %d:\n%s\n", i+1, s.ToMnemonic())
				}
			}

			return nil
		},
	}

	cmd.Flags().IntVarP(&threshold, "threshold", "m", 2, "Minimum shares required for reconstruction")
	cmd.Flags().IntVarP(&shares, "shares", "n", 3, "Total number of shares to generate")
	cmd.Flags().StringVarP(&passphrase, "passphrase", "s", "", "Passphrase to unlock the identity")

	return cmd
}

func identityCombineCmd() *cobra.Command {
	var output string
	var protectPassphrase string
	var noPassword bool

	cmd := &cobra.Command{
		Use:   "combine [mnemonics...]",
		Short: "Reconstruct a private identity from shards",
		RunE: func(cmd *cobra.Command, args []string) error {
			checkJSONMode(cmd)
			if len(args) == 0 {
				return fmt.Errorf("at least one shard mnemonic is required")
			}

			var shards []crypto.Share
			for _, mn := range args {
				s, err := crypto.FromMnemonic(mn)
				if err != nil {
					return fmt.Errorf("invalid mnemonic: %w", err)
				}
				shards = append(shards, *s)
			}

			blob, err := crypto.CombineShares(shards)
			if err != nil {
				return fmt.Errorf("failed to reconstruct secret: %w", err)
			}
			defer crypto.SafeClear(blob)

			if len(blob) < 8 {
				return fmt.Errorf("reconstructed blob too short")
			}

			kemLen := binary.BigEndian.Uint32(blob[0:4])
			if uint32(len(blob)) < 8+kemLen {
				return fmt.Errorf("reconstructed blob corrupted (invalid KEM length)")
			}
			kemPriv := blob[4 : 4+kemLen]

			sigLen := binary.BigEndian.Uint32(blob[4+kemLen : 8+kemLen])
			if uint32(len(blob)) != 8+kemLen+sigLen {
				return fmt.Errorf("reconstructed blob corrupted (invalid SIG length)")
			}
			sigPriv := blob[8+kemLen:]

			// Derive public keys from private keys.
			kemPub, err := crypto.DeriveKEMPublic(kemPriv)
			if err != nil {
				return fmt.Errorf("failed to derive KEM public key: %w", err)
			}
			sigPub, err := crypto.DeriveSIGPublic(sigPriv)
			if err != nil {
				return fmt.Errorf("failed to derive SIG public key: %w", err)
			}

			m := crypto.NewIdentityManager()
			basePath, baseName, err := m.ResolveBaseKeyPath(output)
			if err != nil {
				return err
			}

			pass, err := getInitialPassphrase(noPassword, protectPassphrase)
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			defer crypto.SafeClear(pass)

			if err := writeIdentityKeys(basePath, baseName, kemPub, kemPriv, sigPub, sigPriv, pass, 1); err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			if JSONOutput {
				printJSON(crypto.IdentityResult{
					Status:   "success",
					BasePath: basePath,
				})
			} else {
				fmt.Printf("Successfully reconstructed and saved identity to %s\n", basePath)
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "restored_id", "Name for the restored identity")
	cmd.Flags().StringVarP(&protectPassphrase, "passphrase", "s", "", "Passphrase to protect the restored identity")
	cmd.Flags().BoolVarP(&noPassword, "no-password", "n", false, "Save unprotected (Automation Mode)")

	return cmd
}

func identityActiveCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "active",
		Short: "List absolute paths of available public keys for encryption",
		RunE: func(_ *cobra.Command, _ []string) error {
			m := crypto.NewIdentityManager()
			keys, err := m.ListActiveIdentities()
			if err != nil {
				return err
			}

			if JSONOutput {
				printJSON(map[string]interface{}{
					"active_keys": keys,
				})
			} else {
				fmt.Println("🛡️  Active Public Keys (Absolute Paths):")
				if len(keys) == 0 {
					fmt.Println("  No identities found.")
				}
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
			m := crypto.NewIdentityManager()
			if _, err := os.Stat(m.KeysDir); os.IsNotExist(err) {
				if JSONOutput {
					printJSON([]string{})
					return nil
				}
				fmt.Println("No identities found.")
				return nil
			}

			files, err := os.ReadDir(m.KeysDir)
			if err != nil {
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
				if len(identities) == 0 {
					fmt.Println("  No identities found.")
				}
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
			m := crypto.NewIdentityManager()

			basePath, _, err := m.ResolveBaseKeyPath(name)
			if err != nil {
				return err
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
			m := crypto.NewIdentityManager()

			oldBase, _, err := m.ResolveBaseKeyPath(oldName)
			if err != nil {
				return err
			}
			newBase, _, err := m.ResolveBaseKeyPath(newName)
			if err != nil {
				return err
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
				printJSON(crypto.IdentityResult{
					Status: "success",
					From:   oldName,
					To:     newName,
				})
			} else {
				fmt.Printf("Successfully renamed identity '%s' to '%s'\n", oldName, newName)
			}
			return nil
		},
	}
	return cmd
}
