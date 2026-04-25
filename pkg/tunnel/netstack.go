package tunnel

import (
	"context"
	"fmt"
	"net"

	"tailscale.com/wgengine/netstack"
)

// Netstack represents a user-space TCP/IP stack instance.
type Netstack struct {
	NS *netstack.Impl
}

// NewNetstack initializes a full Tailscale user-space network stack.
func NewNetstack() (*Netstack, error) {
	// Implementation pending
	return nil, fmt.Errorf("tailscale netstack implementation pending")
}

// DialContext provides a way to initiate outbound connections.
func (n *Netstack) DialContext(ctx context.Context, network, address string) (net.Conn, error) {
	return nil, fmt.Errorf("dialing not implemented yet")
}
