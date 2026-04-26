package tunnel

import (
	"context"
	"io"

	"github.com/hashicorp/yamux"
	"github.com/quic-go/quic-go"
)

// TunnelMux defines the strategy for multiplexing multiple virtual streams 
// through a single secure transport tunnel.
type TunnelMux interface {
	OpenStream(ctx context.Context) (io.ReadWriteCloser, error)
	AcceptStream(ctx context.Context) (io.ReadWriteCloser, error)
	Close() error
}

// TunnelListener defines the interface for accepting new multiplexed tunnels.
type TunnelListener interface {
	Accept(ctx context.Context) (TunnelMux, error)
	Close() error
	Addr() string
}

// QUICMux implements TunnelMux using the native QUIC protocol.
type QUICMux struct {
	Session *quic.Conn
}

func (m *QUICMux) OpenStream(ctx context.Context) (io.ReadWriteCloser, error) {
	return m.Session.OpenStreamSync(ctx)
}

func (m *QUICMux) AcceptStream(ctx context.Context) (io.ReadWriteCloser, error) {
	return m.Session.AcceptStream(ctx)
}

func (m *QUICMux) Close() error {
	return m.Session.CloseWithError(0, "graceful shutdown")
}

// YamuxMux implements TunnelMux using Hashicorp Yamux.
type YamuxMux struct {
	Session *yamux.Session
}

func (m *YamuxMux) OpenStream(ctx context.Context) (io.ReadWriteCloser, error) {
	return m.Session.OpenStream()
}

func (m *YamuxMux) AcceptStream(ctx context.Context) (io.ReadWriteCloser, error) {
	return m.Session.AcceptStream()
}

func (m *YamuxMux) Close() error {
	return m.Session.Close()
}
