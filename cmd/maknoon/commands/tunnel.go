package commands

import (
	"context"
	"crypto/tls"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/al-Zamakhshari/maknoon/pkg/tunnel"
	"github.com/libp2p/go-libp2p"
	"github.com/multiformats/go-multiaddr"
	"github.com/spf13/cobra"
)

// TunnelCmd returns the root command for L4 gateway operations.
func TunnelCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "tunnel",
		Short: "Post-Quantum L4 Tunnel Gateway",
		Long:  `Manage secure network perimeters via PQC-QUIC, TCP-Yamux, or libp2p P2P.`,
	}

	cmd.AddCommand(tunnelListenCmd())
	cmd.AddCommand(tunnelStartCmd())

	return cmd
}

func tunnelListenCmd() *cobra.Command {
	var addr string
	var certFile, keyFile string
	var useYamux bool
	var useP2P bool

	cmd := &cobra.Command{
		Use:   "listen",
		Short: "Start a Post-Quantum Tunnel Server (Gateway Receiver)",
		RunE: func(cmd *cobra.Command, args []string) error {
			if useP2P {
				// Initialize libp2p with specific port if provided
				var opts []libp2p.Option
				if addr != "" {
					// Convert :port to multiaddr
					ma, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%s", strings.TrimPrefix(addr, ":")))
					if err == nil {
						opts = append(opts, libp2p.ListenAddrs(ma))
					}
				}

				h, err := tunnel.NewLibp2pHost(opts...)
				if err != nil {
					return err
				}
				defer h.Close()

				fmt.Printf("🚀 P2P Tunnel Server active!\n")
				fmt.Printf("🆔 Peer ID: %s\n", h.ID())
				fmt.Println("📍 Multiaddrs:")
				for _, addr := range h.Addrs() {
					fmt.Printf("  - %s/p2p/%s\n", addr, h.ID())
				}

				server := &tunnel.TunnelServer{P2PHost: h}
				return server.StartLibp2p(cmd.Context())
			}

			if useYamux {
				// Setup PQC TLS for Yamux
				tlsConf := tunnel.GetPQCConfig()
				cert, err := tunnel.GenerateTestCertificate()
				if err != nil {
					return fmt.Errorf("failed to generate ephemeral cert: %w", err)
				}
				tlsConf.Certificates = []tls.Certificate{cert}

				l, err := tls.Listen("tcp", addr, tlsConf)
				if err != nil {
					return fmt.Errorf("failed to start PQC-TCP listener: %w", err)
				}
				defer l.Close()

				fmt.Printf("🚀 PQC-Yamux Tunnel Server listening on %s (TCP)\n", addr)
				for {
					conn, err := l.Accept()
					if err != nil {
						return err
					}
					sess, err := tunnel.WrapYamux(conn, true)
					if err != nil {
						conn.Close()
						continue
					}
					server := &tunnel.TunnelServer{Session: sess}
					go server.StartYamux(cmd.Context())
				}
			}

			// 1. Setup PQC TLS
			tlsConf := tunnel.GetPQCConfig()

			if certFile != "" && keyFile != "" {
				// In v3.0, loading custom certs would happen here
				return fmt.Errorf("loading custom certificates not yet implemented in CLI")
			} else {
				fmt.Println("⚠️  Warning: Using ephemeral self-signed certificate for tunnel")
				cert, err := tunnel.GenerateTestCertificate()
				if err != nil {
					return fmt.Errorf("failed to generate ephemeral cert: %w", err)
				}
				tlsConf.Certificates = []tls.Certificate{cert}
			}

			// 2. Start Listener
			srv, err := tunnel.Listen(addr, tlsConf, GlobalContext.Engine.GetConfig().Tunnel)
			if err != nil {
				return fmt.Errorf("failed to start listener: %w", err)
			}
			defer srv.Listener.Close()

			server := &tunnel.TunnelServer{Listener: srv.Listener}
			fmt.Printf("🚀 PQC Tunnel Server listening on %s (UDP)\n", addr)

			return server.Start(cmd.Context())
		},
	}

	cmd.Flags().StringVar(&addr, "address", ":4433", "Address to listen on")
	cmd.Flags().StringVar(&certFile, "tls-cert", "", "Path to TLS certificate")
	cmd.Flags().StringVar(&keyFile, "tls-key", "", "Path to TLS private key")
	cmd.Flags().BoolVar(&useYamux, "yamux", false, "Use TCP+Yamux mode")
	cmd.Flags().BoolVar(&useP2P, "p2p", false, "Use libp2p for P2P/NAT traversal")

	return cmd
}

func tunnelStartCmd() *cobra.Command {
	var remote string
	var localPort int
	var useYamux bool
	var useP2P bool
	var p2pAddr string

	cmd := &cobra.Command{
		Use:   "start",
		Short: "Start a local Post-Quantum SOCKS5 Gateway",
		RunE: func(cmd *cobra.Command, args []string) error {
			if err := InitEngine(); err != nil {
				return err
			}

			opts := tunnel.TunnelOptions{
				RemoteEndpoint: remote,
				LocalProxyPort: localPort,
				UseYamux:       useYamux,
				P2PMode:        useP2P,
				P2PAddr:        p2pAddr,
			}

			status, err := GlobalContext.Engine.TunnelStart(&crypto.EngineContext{Context: context.Background()}, opts)
			if err != nil {
				return err
			}

			fmt.Printf("🔒 PQC L4 Tunnel Active\n")
			fmt.Printf("📡 Local Proxy: %s\n", status.LocalAddress)
			fmt.Printf("🌍 Remote Peer: %s\n", status.RemoteEndpoint)

			sig := make(chan os.Signal, 1)
			signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
			<-sig

			fmt.Println("\n🛑 Tearing down tunnel...")
			return GlobalContext.Engine.TunnelStop(&crypto.EngineContext{Context: context.Background()})
		},
	}

	cmd.Flags().StringVar(&remote, "remote", "", "Remote PQC Tunnel endpoint (host:port)")
	cmd.Flags().IntVar(&localPort, "port", 1080, "Local SOCKS5 proxy port")
	cmd.Flags().BoolVar(&useYamux, "yamux", false, "Use TCP+Yamux mode")
	cmd.Flags().BoolVar(&useP2P, "p2p", false, "Use libp2p for P2P mode")
	cmd.Flags().StringVar(&p2pAddr, "p2p-addr", "", "Remote P2P Multiaddr")

	return cmd
}
