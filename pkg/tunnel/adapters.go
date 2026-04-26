package tunnel

import (
	"io"
	"net"
	"time"

	"github.com/libp2p/go-libp2p/core/network"
	"github.com/multiformats/go-multiaddr"
	"github.com/quic-go/quic-go"
)

// quicConn wraps *quic.Stream to satisfy net.Conn.
type quicConn struct {
	rawStream *quic.Stream
	session   *quic.Conn
}

func (c *quicConn) Read(b []byte) (n int, err error)  { return c.rawStream.Read(b) }
func (c *quicConn) Write(b []byte) (n int, err error) { return c.rawStream.Write(b) }
func (c *quicConn) Close() error                      { return c.rawStream.Close() }
func (c *quicConn) LocalAddr() net.Addr               { return c.session.LocalAddr() }
func (c *quicConn) RemoteAddr() net.Addr              { return c.session.RemoteAddr() }
func (c *quicConn) SetDeadline(t time.Time) error {
	c.rawStream.SetReadDeadline(t)
	c.rawStream.SetWriteDeadline(t)
	return nil
}
func (c *quicConn) SetReadDeadline(t time.Time) error  { return c.rawStream.SetReadDeadline(t) }
func (c *quicConn) SetWriteDeadline(t time.Time) error { return c.rawStream.SetWriteDeadline(t) }

// libp2pConn wraps network.Stream to satisfy net.Conn.
type libp2pConn struct {
	network.Stream
}

func (c *libp2pConn) LocalAddr() net.Addr {
	return &multiaddrAddr{ma: c.Stream.Conn().LocalMultiaddr()}
}
func (c *libp2pConn) RemoteAddr() net.Addr {
	return &multiaddrAddr{ma: c.Stream.Conn().RemoteMultiaddr()}
}

type multiaddrAddr struct {
	ma multiaddr.Multiaddr
}

func (a *multiaddrAddr) Network() string { return "libp2p" }
func (a *multiaddrAddr) String() string  { return a.ma.String() }

// connAdapter wraps an io.ReadWriteCloser to satisfy net.Conn (used by Yamux).
type connAdapter struct {
	io.ReadWriteCloser
}

func (c *connAdapter) LocalAddr() net.Addr                { return &net.IPAddr{IP: net.IPv4zero} }
func (c *connAdapter) RemoteAddr() net.Addr               { return &net.IPAddr{IP: net.IPv4zero} }
func (c *connAdapter) SetDeadline(t time.Time) error      { return nil }
func (c *connAdapter) SetReadDeadline(t time.Time) error  { return nil }
func (c *connAdapter) SetWriteDeadline(t time.Time) error { return nil }
