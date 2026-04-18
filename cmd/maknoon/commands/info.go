package commands

import (
	"fmt"
	"io"
	"os"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/spf13/cobra"
)

// InfoCmd returns the cobra command for inspecting Maknoon files.
func InfoCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "info [file]",
		Short: "Inspect a Maknoon encrypted file's metadata",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			filePath := args[0]
			f, err := os.Open(filePath)
			if err != nil {
				return err
			}
			defer f.Close()

			// Read enough to cover the fixed header
			header := make([]byte, 6)
			if _, err := io.ReadFull(f, header); err != nil {
				return fmt.Errorf("invalid file: header too short")
			}

			magic := string(header[:4])
			profileID := header[4]
			flags := header[5]

			if JSONOutput {
				return printInfoJSON(magic, profileID, flags, filePath)
			}

			fmt.Printf("File: %s\n", filePath)
			fmt.Printf("----------------------------------------\n")
			
			switch magic {
			case crypto.MagicHeader:
				fmt.Println("Type:           Symmetric (Passphrase Protected)")
			case crypto.MagicHeaderAsym:
				fmt.Println("Type:           Asymmetric (Public Key Protected)")
			default:
				return fmt.Errorf("not a valid Maknoon file (invalid magic: %s)", magic)
			}

			fmt.Printf("Profile ID:     %d\n", profileID)
			
			// Flags
			isCompressed := flags&crypto.FlagCompress != 0
			isArchive := flags&crypto.FlagArchive != 0
			
			fmt.Printf("Compression:    %v\n", isCompressed)
			fmt.Printf("Archive:        %v\n", isArchive)

			// Try to get profile name
			profile, err := crypto.GetProfile(profileID, f)
			if err == nil {
				fmt.Printf("KEM Algorithm:  %s\n", profile.KEMName())
				fmt.Printf("SIG Algorithm:  %s\n", profile.SIGName())
			}

			return nil
		},
	}
	return cmd
}

func printInfoJSON(magic string, profileID byte, flags byte, path string) error {
	type info struct {
		Path        string `json:"path"`
		Type        string `json:"type"`
		ProfileID   byte   `json:"profile_id"`
		Compressed  bool   `json:"compressed"`
		IsArchive   bool   `json:"is_archive"`
	}

	res := info{
		Path:       path,
		ProfileID:  profileID,
		Compressed: flags&crypto.FlagCompress != 0,
		IsArchive:  flags&crypto.FlagArchive != 0,
	}

	if magic == crypto.MagicHeader {
		res.Type = "symmetric"
	} else if magic == crypto.MagicHeaderAsym {
		res.Type = "asymmetric"
	}

	printJSON(res)
	return nil
}
