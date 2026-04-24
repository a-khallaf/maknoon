package commands

import (
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

// ManCmd returns a command to verify the man page integrity.
func ManCmd() *cobra.Command {
	var verify bool

	cmd := &cobra.Command{
		Use:    "man",
		Short:  "Verify the man page integrity",
		Hidden: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			if verify {
				return verifyManPage(cmd.Root())
			}
			return nil
		},
	}

	cmd.Flags().BoolVar(&verify, "verify", false, "Verify that the man page is up to date with the CLI structure")
	return cmd
}

func verifyManPage(root *cobra.Command) error {
	manPath := "maknoon.1"
	content, err := os.ReadFile(manPath)
	if err != nil {
		return fmt.Errorf("failed to read man page: %w", err)
	}

	manContent := string(content)
	missing := []string{}

	// Helper to check command and its subcommands
	var checkCmd func(*cobra.Command)
	checkCmd = func(c *cobra.Command) {
		if c.Hidden || c.Name() == "help" || c.Name() == "completion" {
			return
		}

		// Simple check: Is the command name present in the man page?
		// We search for ".B <name>" to ensure it's documented as a command
		if !strings.Contains(manContent, ".B "+c.Name()) {
			missing = append(missing, c.CommandPath())
		}

		for _, sub := range c.Commands() {
			checkCmd(sub)
		}
	}

	// We start from the commands added to root
	for _, sub := range root.Commands() {
		checkCmd(sub)
	}

	if len(missing) > 0 {
		fmt.Fprintf(os.Stderr, "❌ Man page out of sync! The following commands are missing from %s:\n", manPath)
		for _, m := range missing {
			fmt.Fprintf(os.Stderr, "  - %s\n", m)
		}
		return fmt.Errorf("man page verification failed")
	}

	fmt.Println("✅ Man page is up to date.")
	return nil
}
