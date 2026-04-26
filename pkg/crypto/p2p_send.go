package crypto

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
)

const P2PSendProtocol = "/maknoon/send/1.0.0"

// SendHeader is the first message in a libp2p file transfer.
type SendHeader struct {
	FileName string `json:"name"`
	FileSize int64  `json:"size"`
}

// runLibp2pSend handles the sender side of a libp2p file transfer.
func (e *Engine) runLibp2pSend(ectx *EngineContext, inputName string, r io.Reader, h host.Host, target string, opts P2PSendOptions, status chan P2PStatus) {
	defer close(status)

	pID, err := peer.Decode(target)
	if err != nil {
		status <- P2PStatus{Phase: "error", Error: fmt.Errorf("invalid PeerID: %w", err)}
		return
	}

	status <- P2PStatus{Phase: "connecting"}
	stream, err := h.NewStream(ectx.Context, pID, P2PSendProtocol)
	if err != nil {
		status <- P2PStatus{Phase: "error", Error: fmt.Errorf("failed to open stream: %w", err)}
		return
	}
	defer stream.Close()

	// 1. Prepare encrypted payload
	status <- P2PStatus{Phase: "encrypting"}
	tmpEnc, err := os.CreateTemp("", "maknoon-p2p-send-*.makn")
	if err != nil {
		status <- P2PStatus{Phase: "error", Error: err}
		return
	}
	defer os.Remove(tmpEnc.Name())
	defer tmpEnc.Close()

	protectOpts := Options{
		Passphrase:  opts.Passphrase,
		PublicKey:   opts.PublicKey,
		Stealth:     opts.Stealth,
		Compress:    true,
		IsArchive:   opts.IsDirectory,
		Concurrency: e.Config.AgentLimits.MaxWorkers,
	}

	_, err = e.Protect(ectx, inputName, r, tmpEnc, protectOpts)
	if err != nil {
		status <- P2PStatus{Phase: "error", Error: err}
		return
	}

	fi, _ := tmpEnc.Stat()
	totalBytes := fi.Size()
	if _, err := tmpEnc.Seek(0, 0); err != nil {
		status <- P2PStatus{Phase: "error", Error: err}
		return
	}

	// 2. Send Header
	header := SendHeader{
		FileName: filepath.Base(inputName) + ".makn",
		FileSize: totalBytes,
	}
	if err := json.NewEncoder(stream).Encode(header); err != nil {
		status <- P2PStatus{Phase: "error", Error: fmt.Errorf("failed to send header: %w", err)}
		return
	}

	// 3. Stream Data
	status <- P2PStatus{Phase: "transferring", BytesTotal: totalBytes}
	if _, err := io.Copy(stream, tmpEnc); err != nil {
		status <- P2PStatus{Phase: "error", Error: fmt.Errorf("transfer failed: %w", err)}
		return
	}

	status <- P2PStatus{Phase: "success"}
}

// runLibp2pReceive handles the receiver side of a libp2p file transfer.
func (e *Engine) runLibp2pReceive(ectx *EngineContext, h host.Host, opts P2PReceiveOptions, status chan P2PStatus) {
	defer close(status)

	h.SetStreamHandler(P2PSendProtocol, func(stream network.Stream) {
		defer stream.Close()
		slog.Info("p2p: incoming file transfer", "from", stream.Conn().RemotePeer())

		// 1. Read Header
		var header SendHeader
		if err := json.NewDecoder(stream).Decode(&header); err != nil {
			slog.Error("p2p: failed to read header", "err", err)
			return
		}

		status <- P2PStatus{
			Phase:      "transferring",
			FileName:   header.FileName,
			BytesTotal: header.FileSize,
		}

		// 2. Download to temp file
		tmpFile, err := os.CreateTemp("", "maknoon-p2p-recv-*.makn")
		if err != nil {
			status <- P2PStatus{Phase: "error", Error: err}
			return
		}
		defer os.Remove(tmpFile.Name())
		defer tmpFile.Close()

		if _, err := io.Copy(tmpFile, stream); err != nil {
			status <- P2PStatus{Phase: "error", Error: err}
			return
		}

		// 3. Decrypt
		if _, err := tmpFile.Seek(0, 0); err != nil {
			status <- P2PStatus{Phase: "error", Error: err}
			return
		}

		status <- P2PStatus{Phase: "decrypting"}

		finalOut := opts.OutputDir
		if finalOut == "" {
			finalOut = strings.TrimSuffix(filepath.Base(header.FileName), ".makn")
		}

		unprotectOpts := Options{
			Passphrase:      opts.Passphrase,
			LocalPrivateKey: opts.PrivateKey,
			Stealth:         opts.Stealth,
			Concurrency:     e.Config.AgentLimits.MaxWorkers,
		}

		_, err = e.Unprotect(ectx, tmpFile, nil, finalOut, unprotectOpts)
		if err != nil {
			status <- P2PStatus{Phase: "error", Error: err}
			return
		}

		status <- P2PStatus{Phase: "success", FileName: finalOut}
	})

	// Wait for context to end
	<-ectx.Done()
}
