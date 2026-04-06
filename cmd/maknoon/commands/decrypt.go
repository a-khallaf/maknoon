package commands

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/klauspost/compress/zstd"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
	"github.com/a-khallaf/maknoon/pkg/crypto"
	"golang.org/x/term"
)

func DecryptCmd() *cobra.Command {
	var output string
	var keyPath string
	var passphrase string

	cmd := &cobra.Command{
		Use:   "decrypt [file]",
		Short: "Decrypt a .makn file or directory",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			inputFile := args[0]
			in, err := os.Open(inputFile)
			if err != nil {
				return fmt.Errorf("failed to open input file: %w", err)
			}
			defer in.Close()

			info, err := in.Stat()
			if err != nil { return err }

			// 1. Peek at the header to determine encryption type and flags
			header := make([]byte, 6)
			if _, err := io.ReadFull(in, header); err != nil {
				return fmt.Errorf("failed to read file header: %w", err)
			}
			in.Seek(0, 0)

			magic := string(header[:4])
			flags := header[5]
			
			// 2. Handle Passphrase/Identity logic BEFORE initializing UI
			var password []byte
			if passphrase != "" {
				password = []byte(passphrase)
			} else if env := os.Getenv("MAKNOON_PASSPHRASE"); env != "" {
				password = []byte(env)
			}

			var finalKey []byte
			if magic == crypto.MagicHeader {
				// Symmetric: ensure we have the file password
				if len(password) == 0 {
					fmt.Print("Enter passphrase: ")
					p, err := term.ReadPassword(int(os.Stdin.Fd()))
					fmt.Println()
					if err != nil { return err }
					password = p
				}
				finalKey = password
			} else if magic == crypto.MagicHeaderAsym {
				// Asymmetric: load and potentially unlock private key
				resolvedPath := crypto.ResolveKeyPath(keyPath)
				keyBytes, err := os.ReadFile(resolvedPath)
				if err != nil { return fmt.Errorf("failed to read private key: %w", err) }

				if len(keyBytes) > 4 && string(keyBytes[:4]) == crypto.MagicHeader {
					// Unlock private key
					if len(password) == 0 {
						fmt.Print("Enter passphrase to unlock your private key: ")
						p, err := term.ReadPassword(int(os.Stdin.Fd()))
						fmt.Println()
						if err != nil { return err }
						password = p
					}
					var unlockedKey bytes.Buffer
					if _, err := crypto.DecryptStream(bytes.NewReader(keyBytes), &unlockedKey, password); err != nil {
						return fmt.Errorf("failed to unlock private key: %w", err)
					}
					finalKey = unlockedKey.Bytes()
				} else {
					finalKey = keyBytes
				}
			} else {
				return fmt.Errorf("unsupported or invalid maknoon file header: %s", magic)
			}

			// Clean RAM on exit
			defer func() {
				if len(password) > 0 { crypto.SafeClear(password) }
				if magic == crypto.MagicHeaderAsym { crypto.SafeClear(finalKey) }
			}()

			// 3. Now that we have everything, initialize the Progress Bar and Pipe
			pr, pw := io.Pipe()
			bar := progressbar.DefaultBytes(info.Size(), "restoring")
			proxyIn := io.TeeReader(in, bar)

			fmt.Printf("Decrypting '%s'...\n", inputFile)

			go func() {
				var dErr error
				if magic == crypto.MagicHeader {
					_, dErr = crypto.DecryptStream(proxyIn, pw, finalKey)
				} else {
					_, dErr = crypto.DecryptStreamWithPrivateKey(proxyIn, pw, finalKey)
				}
				pw.CloseWithError(dErr)
			}()

			return finalizeDecryption(pr, flags, output, inputFile)
		},
	}

	cmd.Flags().StringVarP(&output, "output", "o", "", "Output file path or directory")
	cmd.Flags().StringVarP(&keyPath, "private-key", "k", "", "Path to your private key")
	cmd.Flags().StringVarP(&passphrase, "passphrase", "s", "", "Passphrase for decryption")
	return cmd
}

func finalizeDecryption(pr io.Reader, flags byte, output, inputFile string) error {
	var decryptedReader io.Reader = pr
	if flags&crypto.FlagCompress != 0 {
		zr, err := zstd.NewReader(pr)
		if err != nil { return err }
		defer zr.Close()
		decryptedReader = zr
	}

	if flags&crypto.FlagArchive != 0 {
		return crypto.ExtractArchive(decryptedReader, output)
	}

	outPath := output
	if outPath == "" {
		if strings.HasSuffix(inputFile, ".makn") {
			outPath = strings.TrimSuffix(inputFile, ".makn")
		} else {
			outPath = inputFile + ".dec"
		}
	}
	out, err := os.Create(outPath)
	if err != nil { return err }
	defer out.Close()
	_, err = io.Copy(out, decryptedReader)
	return err
}
