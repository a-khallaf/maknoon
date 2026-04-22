package commands

import (
	"fmt"
	"os"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/spf13/cobra"
)

// VerifyCmd returns the cobra command for verifying digital signatures.
func VerifyCmd() *cobra.Command {
	var pubKeyPath string
	var signaturePath string

	cmd := &cobra.Command{
		Use:   "verify [file]",
		Short: "Verify a file's integrity and signature",
		Args:  cobra.ExactArgs(1),
		RunE: func(_ *cobra.Command, args []string) error {
			filePath := args[0]
			if err := validatePath(filePath); err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}
			data, err := os.ReadFile(filePath)
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return err
				}
				return err
			}

			if signaturePath == "" {
				signaturePath = filePath + ".sig"
			}
			sigBytes, err := os.ReadFile(signaturePath)
			if err != nil {
				err := fmt.Errorf("signature file not found: %w", err)
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			im := crypto.NewIdentityManager()
			pubKeyBytes, err := im.ResolvePublicKey(pubKeyPath, false)
			if err != nil {
				if JSONOutput {
					printErrorJSON(err)
					return nil
				}
				return err
			}

			valid := crypto.VerifySignature(data, sigBytes, pubKeyBytes)
			if valid {
				if JSONOutput {
					printJSON(crypto.CommonResult{Status: "success", Message: "Signature Verified"})
				} else {
					fmt.Println("✅ Signature Verified! The data is authentic and has not been tampered with.")
				}
			} else {

				err := fmt.Errorf("❌ Signature Verification FAILED! The data might be corrupted or from an untrusted source")
				if JSONOutput {
					printErrorJSON(err)
					return err
				}
				return err
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&pubKeyPath, "public-key", "p", "", "Path to the ML-DSA public key")
	cmd.Flags().StringVarP(&signaturePath, "signature", "g", "", "Path to the signature file (defaults to file.sig)")
	return cmd
}
