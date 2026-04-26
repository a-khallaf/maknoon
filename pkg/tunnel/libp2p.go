package tunnel

import (
	"context"
	"fmt"
	"net"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/net/connmgr"
	"github.com/multiformats/go-multiaddr"
)

// MaknoonProtocol is the libp2p protocol ID for Maknoon L4 tunnels.
const MaknoonProtocol = "/maknoon/l4/1.0.0"

// Libp2pSession implements MuxSession for go-libp2p.
type Libp2pSession struct {
	Host   host.Host
	PeerID peer.ID
}

// OpenStream initiates a new multiplexed stream to the target peer.
func (s *Libp2pSession) OpenStream(ctx context.Context) (net.Conn, error) {
	stream, err := s.Host.NewStream(ctx, s.PeerID, MaknoonProtocol)
	if err != nil {
		return nil, fmt.Errorf("failed to open libp2p stream: %w", err)
	}
	return &libp2pConn{Stream: stream}, nil
}

// Close gracefully shuts down the libp2p host.
func (s *Libp2pSession) Close() error {
	return s.Host.Close()
}

// NewLibp2pHost initializes a minimal libp2p host for Maknoon.
func NewLibp2pHost(extraOpts ...libp2p.Option) (host.Host, error) {
	cmgr, err := connmgr.NewConnManager(10, 20)
	if err != nil {
		return nil, err
	}

	opts := []libp2p.Option{
		libp2p.ListenAddrStrings(
			"/ip4/0.0.0.0/tcp/0",
			"/ip4/0.0.0.0/udp/0/quic-v1",
		),
		libp2p.ConnectionManager(cmgr),
		libp2p.EnableRelay(),
		libp2p.EnableHolePunching(),
		libp2p.FallbackDefaults,
	}
	opts = append(opts, extraOpts...)

	return libp2p.New(opts...)
}

// DialLibp2p connects to a remote peer and returns a MuxSession.
func DialLibp2p(ctx context.Context, h host.Host, targetAddr string) (*Libp2pSession, error) {
	ma, err := multiaddr.NewMultiaddr(targetAddr)
	if err != nil {
		return nil, fmt.Errorf("invalid multiaddr: %w", err)
	}

	info, err := peer.AddrInfoFromP2pAddr(ma)
	if err != nil {
		return nil, fmt.Errorf("failed to get addr info: %w", err)
	}

	if err := h.Connect(ctx, *info); err != nil {
		return nil, fmt.Errorf("failed to connect to peer: %w", err)
	}

	return &Libp2pSession{
		Host:   h,
		PeerID: info.ID,
	}, nil
}
