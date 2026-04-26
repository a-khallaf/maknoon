package tunnel

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"io"
	"math/big"
	"net"
	"strconv"
	"testing"
	"time"
)

func TestQUICLoopbackTunnel(t *testing.T) {
	conf := TunnelConfig{
		MaxStreams:       10,
		IdleTimeout:      5,
		HandshakeTimeout: 5,
	}

	serverTLS := GetPQCConfig()
	cert, err := generateSelfSignedCert()
	if err != nil {
		t.Fatalf("failed to generate cert: %v", err)
	}
	serverTLS.Certificates = []tls.Certificate{cert}

	// Start QUIC Listener
	ln, err := Listen("127.0.0.1:0", serverTLS, conf)
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	defer ln.Close()
	serverAddr := ln.Addr()

	// Handle server connections in background
	go func() {
		mux, err := ln.Accept(context.Background())
		if err != nil { return }
		defer mux.Close()
		
		for {
			stream, err := mux.AcceptStream(context.Background())
			if err != nil { return }
			
			// Echo handshake protocol: [1 byte len][address]
			buf := make([]byte, 1)
			if _, err := io.ReadFull(stream, buf); err != nil { return }
			dLen := int(buf[0])
			destBuf := make([]byte, dLen)
			if _, err := io.ReadFull(stream, destBuf); err != nil { return }
			
			stream.Write([]byte("PQC-ACK: " + string(destBuf)))
			stream.Close()
		}
	}()

	// 2. Setup the Client Gateway
	clientTLS := GetPQCConfig()
	clientTLS.InsecureSkipVerify = true

	mux, err := Dial(context.Background(), serverAddr, clientTLS, conf)
	if err != nil {
		t.Fatalf("failed to dial: %v", err)
	}
	defer mux.Close()

	gw := &TunnelGateway{
		Port: 0,
		Mux:  mux,
	}
	if err := gw.Start(); err != nil {
		t.Fatalf("failed to start gateway: %v", err)
	}
	defer gw.Stop()

	// 3. Simulate a SOCKS5 Client request
	proxyConn, err := net.Dial("tcp", "127.0.0.1:"+strconv.Itoa(gw.Port))
	if err != nil {
		t.Fatalf("failed to connect to proxy: %v", err)
	}
	defer proxyConn.Close()

	// SOCKS5 Handshake
	proxyConn.Write([]byte{0x05, 0x01, 0x00})
	authResp := make([]byte, 2)
	io.ReadFull(proxyConn, authResp)

	// Connect to mock-host:443
	targetHost := "mock-host"
	proxyConn.Write([]byte{0x05, 0x01, 0x00, 0x03, byte(len(targetHost))})
	proxyConn.Write([]byte(targetHost))
	proxyConn.Write([]byte{0x01, 0xBB}) // Port 443

	resp := make([]byte, 10)
	io.ReadFull(proxyConn, resp)

	// 4. Verify End-to-End communication
	expected := "PQC-ACK: " + net.JoinHostPort(targetHost, "443")
	received := make([]byte, len(expected))
	if _, err := io.ReadFull(proxyConn, received); err != nil {
		t.Fatalf("failed to read from proxy: %v", err)
	}

	if string(received) != expected {
		t.Errorf("mismatch! expected %q, got %q", expected, string(received))
	}
}

func generateSelfSignedCert() (tls.Certificate, error) {
	priv, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return tls.Certificate{}, err
	}
	template := x509.Certificate{
		SerialNumber: big.NewInt(1),
		Subject: pkix.Name{Organization: []string{"Maknoon Test"}},
		NotBefore: time.Now(), NotAfter: time.Now().Add(time.Hour),
		KeyUsage: x509.KeyUsageKeyEncipherment | x509.KeyUsageDigitalSignature,
		ExtKeyUsage: []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}
	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &priv.PublicKey, priv)
	if err != nil {
		return tls.Certificate{}, err
	}
	return tls.Certificate{Certificate: [][]byte{derBytes}, PrivateKey: priv}, nil
}
