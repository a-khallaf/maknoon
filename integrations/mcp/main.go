package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/al-Zamakhshari/maknoon/pkg/crypto"
	"github.com/al-Zamakhshari/maknoon/pkg/tunnel"
	"github.com/mark3labs/mcp-go/mcp"
	"github.com/mark3labs/mcp-go/server"
)

var engine crypto.MaknoonEngine

func main() {
	if err := setupEngine(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	s := setupServer()

	// Default to stdio for container use cases
	if err := server.ServeStdio(s); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func setupEngine() error {
	policy := &crypto.AgentPolicy{}
	e, err := crypto.NewEngine(policy)
	if err != nil {
		return err
	}
	engine = e
	return nil
}

func setupServer() *server.MCPServer {
	s := server.NewMCPServer("Maknoon", "3.1.0")

	// Tool: encrypt_file
	encryptFile := mcp.NewTool("encrypt_file",
		mcp.WithDescription("Securely encrypt a file or directory for a recipient"),
	)
	encryptFile.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"path":       map[string]interface{}{"type": "string", "description": "Path to the input file or directory"},
			"output":     map[string]interface{}{"type": "string", "description": "Path for the encrypted output"},
			"public_key": map[string]interface{}{"type": "string", "description": "Optional path to the recipient's public key"},
			"stealth":    map[string]interface{}{"type": "boolean", "description": "Enable headerless mode"},
		},
		Required: []string{"path", "output"},
	}
	s.AddTool(encryptFile, encryptHandler)

	// Tool: decrypt_file
	decryptFile := mcp.NewTool("decrypt_file",
		mcp.WithDescription("Decrypt a Maknoon encrypted file"),
	)
	decryptFile.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"path":    map[string]interface{}{"type": "string", "description": "Path to the encrypted file"},
			"output":  map[string]interface{}{"type": "string", "description": "Output path for the decrypted content"},
			"pass":    map[string]interface{}{"type": "string", "description": "Passphrase for symmetric decryption"},
			"stealth": map[string]interface{}{"type": "boolean", "description": "Enable stealth mode detection"},
		},
		Required: []string{"path", "output"},
	}
	s.AddTool(decryptFile, decryptHandler)

	// Tool: inspect_file
	inspectFile := mcp.NewTool("inspect_file",
		mcp.WithDescription("Inspect a Maknoon encrypted file's metadata"),
	)
	inspectFile.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"path":    map[string]interface{}{"type": "string", "description": "Path to the encrypted file"},
			"stealth": map[string]interface{}{"type": "boolean", "description": "Enable stealth mode detection"},
		},
		Required: []string{"path"},
	}
	s.AddTool(inspectFile, inspectHandler)

	// Tool: p2p_send
	p2pSend := mcp.NewTool("p2p_send",
		mcp.WithDescription("Send a file, directory, or raw text via secure P2P"),
	)
	p2pSend.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"path":       map[string]interface{}{"type": "string", "description": "Path to the file or directory to send"},
			"text":       map[string]interface{}{"type": "string", "description": "Raw text to send instead of a file"},
			"to":         map[string]interface{}{"type": "string", "description": "Recipient @petname or PeerID"},
			"public_key": map[string]interface{}{"type": "string", "description": "Encrypt for a specific recipient's identity"},
			"stealth":    map[string]interface{}{"type": "boolean", "description": "Enable stealth mode"},
		},
		Required: []string{"to"},
	}
	s.AddTool(p2pSend, sendHandler)

	// Tool: p2p_receive
	p2pRecv := mcp.NewTool("p2p_receive",
		mcp.WithDescription("Wait for and receive a secure P2P file transfer"),
	)
	p2pRecv.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"peer_id":     map[string]interface{}{"type": "string", "description": "The PeerID of the sender (optional if listening)"},
			"passphrase":  map[string]interface{}{"type": "string", "description": "The session passphrase from the sender"},
			"private_key": map[string]interface{}{"type": "string", "description": "Path to your private key"},
			"output":      map[string]interface{}{"type": "string", "description": "Output path or directory"},
			"stealth":     map[string]interface{}{"type": "boolean", "description": "Enable headerless mode"},
		},
	}
	s.AddTool(p2pRecv, receiveHandler)

	// Tool: chat_start
	chatStart := mcp.NewTool("chat_start",
		mcp.WithDescription("Initiate an identity-bound P2P Chat session"),
	)
	chatStart.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"target": map[string]interface{}{"type": "string", "description": "Recipient @petname or PeerID"},
		},
	}
	s.AddTool(chatStart, chatStartHandler)

	// Tool: tunnel_start
	tunnelStart := mcp.NewTool("tunnel_start",
		mcp.WithDescription("Provision a Post-Quantum L4 tunnel and SOCKS5 gateway"),
	)
	tunnelStart.InputSchema = mcp.ToolInputSchema{
		Type: "object",
		Properties: map[string]interface{}{
			"remote":    map[string]interface{}{"type": "string", "description": "Remote PQC Tunnel endpoint (host:port)"},
			"port":      map[string]interface{}{"type": "integer", "description": "Local SOCKS5 proxy port (default 1080)"},
			"use_yamux": map[string]interface{}{"type": "boolean", "description": "Use TCP+Yamux mode"},
			"p2p_mode":  map[string]interface{}{"type": "boolean", "description": "Use libp2p for P2P mode"},
			"p2p_addr":  map[string]interface{}{"type": "string", "description": "Remote P2P Multiaddr"},
		},
	}
	s.AddTool(tunnelStart, tunnelStartHandler)

	return s
}

func encryptHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path := request.GetString("path", "")
	output := request.GetString("output", "")
	pubKeyPath := request.GetString("public_key", "")
	stealth := request.GetBool("stealth", false)

	f, err := os.Open(path)
	if err != nil {
		return formatError(err, "encrypt_file")
	}
	defer f.Close()

	out, err := os.Create(output)
	if err != nil {
		return formatError(err, "encrypt_file")
	}
	defer out.Close()

	opts := crypto.Options{
		Stealth: stealth,
	}

	if pubKeyPath != "" {
		im := crypto.NewIdentityManager()
		pk, _ := im.ResolvePublicKey(pubKeyPath, false)
		opts.PublicKey = pk
	}

	_, err = engine.Protect(&crypto.EngineContext{Context: ctx}, filepath.Base(path), f, out, opts)
	if err != nil {
		return formatError(err, "encrypt_file")
	}

	return mcp.NewToolResultText(fmt.Sprintf(`{"status":"success","path":"%s"}`, output)), nil
}

func decryptHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path := request.GetString("path", "")
	output := request.GetString("output", "")
	pass := request.GetString("pass", "")
	stealth := request.GetBool("stealth", false)

	f, err := os.Open(path)
	if err != nil {
		return formatError(err, "decrypt_file")
	}
	defer f.Close()

	opts := crypto.Options{
		Passphrase: []byte(pass),
		Stealth:    stealth,
	}

	_, err = engine.Unprotect(&crypto.EngineContext{Context: ctx}, f, nil, output, opts)
	if err != nil {
		return formatError(err, "decrypt_file")
	}

	return mcp.NewToolResultText(fmt.Sprintf(`{"status":"success","path":"%s"}`, output)), nil
}

func inspectHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path := request.GetString("path", "")
	f, err := os.Open(path)
	if err != nil {
		return formatError(err, "inspect_file")
	}
	defer f.Close()

	info, err := engine.Inspect(&crypto.EngineContext{Context: ctx}, f)
	if err != nil {
		return formatError(err, "inspect_file")
	}

	raw, _ := json.Marshal(info)
	return mcp.NewToolResultText(string(raw)), nil
}

func sendHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	path := request.GetString("path", "")
	text := request.GetString("text", "")
	to := request.GetString("to", "")
	pubKeyPath := request.GetString("public_key", "")
	stealth := request.GetBool("stealth", false)

	var r io.Reader
	var name string
	if text != "" {
		r = strings.NewReader(text)
		name = "text-message"
	} else if path != "" {
		f, err := os.Open(path)
		if err != nil {
			return formatError(err, "send_file")
		}
		defer f.Close()
		r = f
		name = filepath.Base(path)
	}

	opts := crypto.P2PSendOptions{
		Passphrase: []byte(os.Getenv("MAKNOON_PASSPHRASE")),
		Stealth:    stealth,
		P2PMode:    true,
		To:         to,
	}

	if pubKeyPath != "" {
		im := crypto.NewIdentityManager()
		pk, _ := im.ResolvePublicKey(pubKeyPath, false)
		opts.PublicKey = pk
	}

	peerID, _, err := engine.P2PSend(&crypto.EngineContext{Context: ctx}, name, r, opts)
	if err != nil {
		return formatError(err, "send_file")
	}

	// We return PeerID immediately for established status.
	// In a real system, the transfer happens in background.
	return mcp.NewToolResultText(fmt.Sprintf(`{"status":"established","peer_id":"%s"}`, peerID)), nil
}

func receiveHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	peerID := request.GetString("peer_id", "")
	output := request.GetString("output", "")
	stealth := request.GetBool("stealth", false)

	opts := crypto.P2PReceiveOptions{
		Passphrase: []byte(os.Getenv("MAKNOON_PASSPHRASE")),
		OutputDir:  output,
		Stealth:    stealth,
		P2PMode:    true,
	}

	statusChan, err := engine.P2PReceive(&crypto.EngineContext{Context: ctx}, peerID, opts)
	if err != nil {
		return formatError(err, "receive_file")
	}

	var lastStatus crypto.P2PStatus
	for s := range statusChan {
		lastStatus = s
		if s.Phase == "success" || s.Phase == "error" {
			break
		}
	}

	if lastStatus.Error != nil {
		return formatError(lastStatus.Error, "receive_file")
	}

	return mcp.NewToolResultText(fmt.Sprintf(`{"status":"success","path":"%s"}`, lastStatus.FileName)), nil
}

func chatStartHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	target := request.GetString("target", "")

	sess, err := engine.ChatStart(&crypto.EngineContext{Context: ctx}, target)
	if err != nil {
		return formatError(err, "chat_start")
	}

	res := map[string]string{
		"status":  "established",
		"peer_id": sess.Host.ID().String(),
	}
	raw, _ := json.Marshal(res)
	return mcp.NewToolResultText(string(raw)), nil
}

func tunnelStartHandler(ctx context.Context, request mcp.CallToolRequest) (*mcp.CallToolResult, error) {
	remote := request.GetString("remote", "")
	port := request.GetInt("port", 1080)
	useYamux := request.GetBool("use_yamux", false)
	p2pMode := request.GetBool("p2p_mode", false)
	p2pAddr := request.GetString("p2p_addr", "")

	opts := tunnel.TunnelOptions{
		RemoteEndpoint: remote,
		LocalProxyPort: port,
		UseYamux:       useYamux,
		P2PMode:        p2pMode,
		P2PAddr:        p2pAddr,
	}

	status, err := engine.TunnelStart(&crypto.EngineContext{Context: ctx}, opts)
	if err != nil {
		return formatError(err, "tunnel_start")
	}

	raw, _ := json.Marshal(status)
	return mcp.NewToolResultText(string(raw)), nil
}

func formatError(err error, tool string) (*mcp.CallToolResult, error) {
	resp := map[string]interface{}{
		"error": err.Error(),
		"tool":  tool,
	}
	raw, _ := json.Marshal(resp)
	return mcp.NewToolResultError(string(raw)), nil
}
