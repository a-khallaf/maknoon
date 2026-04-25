package tunnel

import (
	"context"
	"crypto/tls"
	"fmt"

	"github.com/quic-go/quic-go"
)

// QUICClient manages a post-quantum QUIC tunnel.
type QUICClient struct {
	Session *quic.Conn
}

// QUICServer represents the receiving end of a post-quantum tunnel.
type QUICServer struct {
	Listener *quic.Listener
}

// Listen starts a post-quantum QUIC listener.
func Listen(address string, tlsConf *tls.Config) (*QUICServer, error) {
	quicConf := &quic.Config{
		MaxIdleTimeout:  0,
		KeepAlivePeriod: 30,
	}

	ln, err := quic.ListenAddr(address, tlsConf, quicConf)
	if err != nil {
		return nil, fmt.Errorf("failed to start QUIC listener: %w", err)
	}

	return &QUICServer{Listener: ln}, nil
}

// GetPQCConfig returns a TLS configuration optimized for Post-Quantum Hybrid security.
func GetPQCConfig() *tls.Config {
	return &tls.Config{
		MinVersion: tls.VersionTLS13,
		// Prioritize ML-KEM-768 hybrid key exchange (Go 1.23+)
		CurvePreferences: []tls.CurveID{
			tls.X25519MLKEM768, // Post-Quantum Hybrid
			tls.X25519,
			tls.CurveP256,
		},
		NextProtos: []string{"maknoon-pqc-tunnel"},
	}
}

// Dial establishes a secure QUIC connection to the remote endpoint.
func Dial(ctx context.Context, address string, tlsConf *tls.Config) (*QUICClient, error) {
	quicConf := &quic.Config{
		MaxIdleTimeout:  0,
		KeepAlivePeriod: 30,
	}

	conn, err := quic.DialAddr(ctx, address, tlsConf, quicConf)
	if err != nil {
		return nil, fmt.Errorf("failed to dial QUIC endpoint: %w", err)
	}

	return &QUICClient{Session: conn}, nil
}

// OpenStream initiates a new multiplexed stream through the tunnel.
func (c *QUICClient) OpenStream(ctx context.Context) (*quic.Stream, error) {
	return c.Session.OpenStreamSync(ctx)
}

// Close gracefully shuts down the tunnel.
func (c *QUICClient) Close() error {
	return c.Session.CloseWithError(0, "graceful shutdown")
}
