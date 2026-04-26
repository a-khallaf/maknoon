package tunnel

import (
	"context"
	"io"
	"log/slog"
	"net"
)

// TunnelServer handles incoming PQC QUIC or Yamux connections and forwards them to internal targets.
type TunnelServer struct {
	Mux TunnelMux
}

// Start begins accepting streams through the multiplexer.
func (s *TunnelServer) Start(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	for {
		stream, err := s.Mux.AcceptStream(ctx)
		if err != nil {
			return err
		}
		go s.handleStream(stream)
	}
}

func (s *TunnelServer) handleStream(stream io.ReadWriteCloser) {
	defer stream.Close()

	// 1. Read Destination Header [1 byte len][address string]
	lb := GlobalPool.Get()
	defer GlobalPool.Put(lb)

	if _, err := io.ReadFull(stream, lb.Bytes()[:1]); err != nil {
		return
	}
	addrLen := int(lb.Bytes()[0])
	
	if _, err := io.ReadFull(stream, lb.Bytes()[:addrLen]); err != nil {
		return
	}
	targetAddr := string(lb.Bytes()[:addrLen])

	slog.Info("tunnel server: forwarding stream", "target", targetAddr)

	// 2. Dial the internal target
	target, err := net.Dial("tcp", targetAddr)
	if err != nil {
		slog.Error("tunnel server: failed to dial target", "target", targetAddr, "err", err)
		return
	}
	defer target.Close()

	// 3. Bi-directional PQC-to-Plaintext Bridge with Memory Hygiene
	done := make(chan struct{}, 2)
	
	go func() {
		lbIn := GlobalPool.Get()
		defer GlobalPool.Put(lbIn)
		io.CopyBuffer(target, stream, lbIn.Bytes())
		done <- struct{}{}
	}()

	go func() {
		lbOut := GlobalPool.Get()
		defer GlobalPool.Put(lbOut)
		io.CopyBuffer(stream, target, lbOut.Bytes())
		done <- struct{}{}
	}()

	<-done
}
