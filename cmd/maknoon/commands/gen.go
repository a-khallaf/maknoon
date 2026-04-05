package commands

import (
	"crypto/rand"
	"fmt"
	"math/big"

	"github.com/spf13/cobra"
)

const (
	lowerLetters = "abcdefghijklmnopqrstuvwxyz"
	upperLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	digits       = "0123456789"
	symbols      = "!@#$%^&*()-_=+[]{}|;:,.<>?"
)

func GenCmd() *cobra.Command {
	var length int
	var noSymbols bool

	cmd := &cobra.Command{
		Use:   "gen",
		Short: "Generate a high-entropy secure password",
		RunE: func(cmd *cobra.Command, args []string) error {
			charset := lowerLetters + upperLetters + digits
			if !noSymbols {
				charset += symbols
			}

			password := make([]byte, length)
			for i := 0; i < length; i++ {
				num, err := rand.Int(rand.Reader, big.NewInt(int64(len(charset))))
				if err != nil {
					return fmt.Errorf("entropy failure: %w", err)
				}
				password[i] = charset[num.Int64()]
			}

			fmt.Println(string(password))

			// Memory Hygiene: Zero out the password bytes immediately after printing
			defer func() {
				for i := range password {
					password[i] = 0
				}
			}()

			return nil
		},
	}

	cmd.Flags().IntVarP(&length, "length", "l", 32, "Length of the generated password")
	cmd.Flags().BoolVarP(&noSymbols, "no-symbols", "n", false, "Exclude symbols from the password")
	
	return cmd
}
