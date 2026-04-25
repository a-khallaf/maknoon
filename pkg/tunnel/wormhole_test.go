package tunnel

import (
	"bytes"
	"net"
	"testing"
)

// MockStream simulates a Wormhole Transit stream (io.ReadWriteCloser)
type MockStream struct {
	*bytes.Buffer
	Closed bool
}

func (m *MockStream) Close() error {
	m.Closed = true
	return nil
}

func TestWormholeFramingIntegrity(t *testing.T) {
	buf := new(bytes.Buffer)
	stream := &MockStream{Buffer: buf}
	
	pconn := &WormholePacketConn{Stream: stream}

	// 1. Test standard packet
	payload := []byte("quantum-safe-payload")
	n, err := pconn.WriteTo(payload, &net.UDPAddr{})
	if err != nil || n != len(payload) {
		t.Fatalf("WriteTo failed: %v", err)
	}

	if buf.Len() != 2+len(payload) {
		t.Errorf("expected framed size %d, got %d", 2+len(payload), buf.Len())
	}

	readBuf := make([]byte, 65536)
	rn, _, err := pconn.ReadFrom(readBuf)
	if err != nil || rn != len(payload) {
		t.Fatalf("ReadFrom failed: %v", err)
	}
	if !bytes.Equal(readBuf[:rn], payload) {
		t.Error("payload corruption detected in framing")
	}

	// 2. Stress Test: Edge sizes
	testSizes := []int{1, 1499, 1500, 2048, 65535}
	for _, size := range testSizes {
		buf.Reset()
		stressPayload := make([]byte, size)
		for i := range stressPayload {
			stressPayload[i] = byte(i % 256)
		}

		pconn.WriteTo(stressPayload, nil)
		rn, _, err := pconn.ReadFrom(readBuf)
		if err != nil || rn != size {
			t.Errorf("failed stress size %d: %v", size, err)
		}
		if !bytes.Equal(readBuf[:rn], stressPayload) {
			t.Errorf("corruption at size %d", size)
		}
	}
}
