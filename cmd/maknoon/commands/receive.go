package commands

import (
	"context"
	"fmt"
	"os"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/schollz/progressbar/v3"
	"github.com/spf13/cobra"
)

var (
	recvOutput     string
	recvPassphrase string
	recvStealth    bool
	quietRecv      bool
	recvPrivateKey string
	recvP2PMode    bool
	recvIdentity   string
)

// ReceiveCmd returns the cobra command for receiving files via secure P2P.
func ReceiveCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "receive [peer_id]",
		Short: "Receive a file, directory, or text via secure P2P",
		Long:  `Downloads and decrypts data directly from a peer via libp2p.`,
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Fprintf(os.Stderr, "DEBUG: receive command starting\n")
			if err := InitEngine(); err != nil {
				return err
			}
			p := GlobalContext.UI.GetPresenter()

			var code string
			if len(args) > 0 {
				code = args[0]
			}

			if GlobalContext.UI.JSON {
				quietRecv = true
			}

			opts := crypto.P2PReceiveOptions{
				Passphrase: []byte(recvPassphrase),
				Stealth:    recvStealth,
				OutputDir:  recvOutput,
				P2PMode:    true,
			}

			if recvPrivateKey != "" {
				m := crypto.NewIdentityManager()
				resolvedPriv := m.ResolveKeyPath(recvPrivateKey, "MAKNOON_PRIVATE_KEY")
				if resolvedPriv == "" {
					p.RenderError(fmt.Errorf("private key required for identity-based P2P"))
					return nil
				}
				privBytes, err := m.LoadPrivateKey(resolvedPriv, []byte(recvPassphrase), "", false)
				if err != nil {
					p.RenderError(err)
					return nil
				}
				opts.PrivateKey = privBytes
			}

			status, err := GlobalContext.Engine.P2PReceive(&crypto.EngineContext{Context: context.Background()}, recvIdentity, code, opts)
			if err != nil {
				p.RenderError(err)
				return nil
			}

			// We need to peek into the first status message to get Multiaddrs for logging
			// But the loop below handles it. The issue might be that s.Addrs is empty.

			var bar *progressbar.ProgressBar
			for s := range status {
				fmt.Fprintf(os.Stderr, "DEBUG: receive phase: %s\n", s.Phase)
				if s.Error != nil {
					p.RenderError(s.Error)
					return nil
				}
				if s.Phase == "connecting" && s.Code != "" {
					// Always print multiaddrs to stderr for mission visibility
					for _, addr := range s.Addrs {
						fmt.Fprintf(os.Stderr, "📍 Multiaddr: %s\n", addr)
					}
					if !quietRecv {
						p.RenderMessage(fmt.Sprintf("🕳️  Waiting for peer. Share your PeerID: %s", s.Code))
					}
				}
				if !quietRecv {
					if bar == nil && s.BytesTotal > 0 {
						bar = progressbar.DefaultBytes(s.BytesTotal, "receiving")
					}
					if bar != nil && s.BytesDone > 0 {
						_ = bar.Set64(s.BytesDone)
					}
				}
				if s.Phase == "success" {
					if !quietRecv {
						p.RenderMessage(fmt.Sprintf("\n✨ Success! Data saved to: %s", s.FileName))
					}
					if GlobalContext.UI.JSON {
						p.RenderSuccess(map[string]string{"status": "success", "file": s.FileName})
					}
					break
				}
			}

			return nil
		},
	}

	cmd.Flags().StringVarP(&recvOutput, "output", "o", "", "Output path or directory")
	cmd.Flags().StringVarP(&recvPassphrase, "passphrase", "s", "", "Session passphrase")
	cmd.Flags().BoolVar(&recvStealth, "stealth", false, "Decrypt as stealth mode")
	cmd.Flags().BoolVarP(&quietRecv, "quiet", "q", false, "Suppress informational messages")
	cmd.Flags().StringVarP(&recvPrivateKey, "private-key", "k", "", "Path to your private key")
	cmd.Flags().BoolVar(&recvP2PMode, "p2p", true, "Use identity-first P2P (libp2p)")
	cmd.Flags().StringVar(&recvIdentity, "identity", "", "Identity name to use (default: active identity)")

	return cmd
}
