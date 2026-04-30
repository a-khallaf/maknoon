package tunnel

import (
	"context"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
)

// TunnelGateway implements a SOCKS5 proxy that routes traffic through a multiplexed tunnel.
type TunnelGateway struct {
	BindAddr string // e.g. "0.0.0.0" or "127.0.0.1"
	Port     int
	Session  MuxSession
	ln       net.Listener
}

// Start launches the SOCKS5 gateway on the local address.
func (g *TunnelGateway) Start() error {
	addr := g.BindAddr
	if addr == "" {
		addr = "127.0.0.1"
	}
	l, err := net.Listen("tcp", addr+":"+strconv.Itoa(g.Port))
	if err != nil {
		return err
	}
	g.ln = l
	fmt.Fprintf(os.Stderr, "socks5: listener started on %s\n", l.Addr())

	// Update port if 0 was used
	if g.Port == 0 {
		_, portStr, _ := net.SplitHostPort(l.Addr().String())
		g.Port, _ = strconv.Atoi(portStr)
	}

	go func() {
		for {
			conn, err := l.Accept()
			if err != nil {
				fmt.Fprintf(os.Stderr, "socks5: accept error: %v\n", err)
				return
			}
			fmt.Fprintf(os.Stderr, "socks5: accepted connection from %s\n", conn.RemoteAddr())
			go g.handleConnection(conn)
		}
	}()

	return nil
}

func (g *TunnelGateway) handleConnection(conn net.Conn) {
	defer conn.Close()

	fmt.Fprintf(os.Stderr, "socks5: handling connection from %s\n", conn.RemoteAddr())

	// Use a standard buffer for the handshake (non-sensitive control data)
	buf := make([]byte, 256)
	if _, err := io.ReadFull(conn, buf[:2]); err != nil || buf[0] != 0x05 {
		fmt.Fprintf(os.Stderr, "socks5: failed to read version/nmethods: %v\n", err)
		return
	}
	nMethods := int(buf[1])
	if _, err := io.ReadFull(conn, buf[:nMethods]); err != nil {
		fmt.Fprintf(os.Stderr, "socks5: failed to read methods: %v\n", err)
		return
	}
	conn.Write([]byte{0x05, 0x00})

	fmt.Fprintf(os.Stderr, "socks5: method negotiation complete, waiting for request\n")

	// Clear buffer before next read
	for i := range buf[:4] {
		buf[i] = 0
	}

	n, err := io.ReadAtLeast(conn, buf[:4], 4)
	if err != nil {
		fmt.Fprintf(os.Stderr, "socks5: failed to read request header: %v (read %d bytes, buf: %v)\n", err, n, buf[:4])
		return
	}
	if buf[0] != 0x05 {
		fmt.Fprintf(os.Stderr, "socks5: invalid request version: %d\n", buf[0])
		return
	}
	if buf[1] != 0x01 {
		fmt.Fprintf(os.Stderr, "socks5: unsupported command: %d\n", buf[1])
		return
	}

	var address string
	switch buf[3] {
	case 0x01: // IPv4
		io.ReadFull(conn, buf[:4])
		address = net.IP(buf[:4]).String()
	case 0x03: // Domain
		io.ReadFull(conn, buf[:1])
		dLen := int(buf[0])
		io.ReadFull(conn, buf[:dLen])
		address = string(buf[:dLen])
	case 0x04: // IPv6
		io.ReadFull(conn, buf[:16])
		address = net.IP(buf[:16]).String()
	default:
		fmt.Fprintf(os.Stderr, "socks5: unknown address type: %d\n", buf[3])
		return
	}
	io.ReadFull(conn, buf[:2])
	port := int(buf[0])<<8 | int(buf[1])
	dest := net.JoinHostPort(address, strconv.Itoa(port))

	fmt.Fprintf(os.Stderr, "socks5: dialing target %s through tunnel\n", dest)

	stream, err := g.Session.OpenStream(context.Background())
	if err != nil {
		fmt.Fprintf(os.Stderr, "socks5: failed to open tunnel stream: %v\n", err)
		conn.Write([]byte{0x05, 0x05, 0x00, 0x01, 0, 0, 0, 0, 0, 0})
		return
	}
	defer stream.Close()

	if _, err := stream.Write([]byte{byte(len(dest))}); err != nil {
		fmt.Fprintf(os.Stderr, "socks5: failed to write stream header: %v\n", err)
		return
	}
	if _, err := stream.Write([]byte(dest)); err != nil {
		fmt.Fprintf(os.Stderr, "socks5: failed to write stream address: %v\n", err)
		return
	}

	conn.Write([]byte{0x05, 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0})

	// HARDENED PIPE: Use GlobalPool enclaves for traffic processing
	pipe := func(dst io.Writer, src io.Reader, done chan struct{}) {
		// Get a locked buffer from the pool
		lb := GlobalPool.Get()
		defer GlobalPool.Put(lb)

		for {
			n, err := src.Read(lb.Bytes())
			if n > 0 {
				if _, werr := dst.Write(lb.Bytes()[:n]); werr != nil {
					break
				}
			}
			if err != nil {
				break
			}
		}
		done <- struct{}{}
	}

	done := make(chan struct{}, 2)
	go pipe(stream, conn, done)
	go pipe(conn, stream, done)
	<-done
}

// Stop shuts down the gateway listener.
func (g *TunnelGateway) Stop() error {
	if g.ln != nil {
		return g.ln.Close()
	}
	return nil
}
